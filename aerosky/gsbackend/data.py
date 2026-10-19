# -*- coding: utf-8 -*-
"""多视角影像获取：从公开数据集下载可以直接喂给 3DGS 的真实影像。

系统本身不做无人机采集，所以这里内置了一批**公开可下载的多视角数据集**
（HuggingFace，无需注册；自带 hf-mirror.com 备用镜像）：

  * 带相机位姿的：NeRF 合成数据集、真实摄影测量数据集（含 COLMAP 稀疏模型）
  * 只有照片的：航拍/地面实拍序列，需要用 pycolmap 自己算位姿（见 aerosky.sfm）
"""
from __future__ import annotations

import fnmatch
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from ..utils.logger import get_logger

LOG = get_logger("gsdata")

MIRRORS = ["https://huggingface.co", "https://hf-mirror.com"]
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


class _Redirect308(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if code == 308:
            return urllib.request.Request(newurl, headers=dict(req.headers),
                                          method=req.get_method())
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_Redirect308())

PRESETS: Dict[str, dict] = {
    # ---- 带位姿：NeRF 合成数据集，小、稳，先跑通用 ----
    "lego": dict(repo="pablovela5620/nerf-synthetic-mirror",
                 allow=["lego/train/*", "lego/transforms_*.json"], kind="blender",
                 note="NeRF 合成：乐高推土机（100 视角，约 43 MB，自带位姿）"),
    "ship": dict(repo="pablovela5620/nerf-synthetic-mirror",
                 allow=["ship/train/*", "ship/transforms_*.json"], kind="blender",
                 note="NeRF 合成：帆船"),
    "drums": dict(repo="pablovela5620/nerf-synthetic-mirror",
                  allow=["drums/train/*", "drums/transforms_*.json"], kind="blender",
                  note="NeRF 合成：架子鼓"),
    # ---- 带位姿：真实照片 + 现成 COLMAP ----
    "south-building": dict(repo="alexmkwizu/colmap-testing-dataset",
                           allow=["south-building/images/*", "south-building/sparse/*"],
                           kind="colmap",
                           note="真实照片：南楼外景（128 张 + COLMAP 位姿）"),
    "gerrard-hall": dict(repo="alexmkwizu/colmap-testing-dataset",
                         allow=["gerrard-hall/images/*", "gerrard-hall/sparse/*"],
                         kind="colmap", note="真实照片：Gerrard Hall（100 张 + COLMAP 位姿）"),
    # ---- 只有照片：需要自己跑 SfM ----
    "bigsur": dict(repo="Jamesbass/bigsur-360-colmap", allow=["images/*"], kind="images",
                   note="真实航拍照片：Big Sur 海岸（需要跑 SfM，建议 --max-images 30 --consecutive）"),
}


def _api(mirror: str, path: str, timeout: int = 60):
    req = urllib.request.Request(mirror + path, headers={"User-Agent": "aerosky/1.0"})
    with _OPENER.open(req, timeout=timeout) as r:
        return json.load(r)


def list_remote(repo: str, mirror: Optional[str] = None) -> List[dict]:
    last = None
    for m in ([mirror] if mirror else MIRRORS):
        try:
            out, stack = [], [""]
            while stack:
                p = stack.pop()
                for e in _api(m, f"/api/datasets/{repo}/tree/main/{p}"):
                    if e["type"] == "directory":
                        stack.append(e["path"])
                    else:
                        size = e.get("size") or (e.get("lfs") or {}).get("size") or 0
                        out.append({"path": e["path"], "size": int(size)})
            return sorted(out, key=lambda x: x["path"])
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"列举 {repo} 失败：{last}")


def download_file(repo: str, path: str, dest: Path, mirror: Optional[str] = None,
                  timeout: int = 180) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    last = None
    for m in ([mirror] if mirror else MIRRORS):
        try:
            url = f"{m}/datasets/{repo}/resolve/main/{urllib.parse.quote(path)}"
            req = urllib.request.Request(url, headers={"User-Agent": "aerosky/1.0"})
            with _OPENER.open(req, timeout=timeout) as r:
                data = r.read()
            dest.write_bytes(data)
            return dest
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"下载 {path} 失败：{last}")


def _match(path: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(path, p) for p in patterns)


def _shrink(src: Path, max_side: int, quality: int = 90) -> None:
    try:
        import cv2
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            return
        h, w = img.shape[:2]
        if max(h, w) <= max_side:
            return
        k = max_side / max(h, w)
        small = cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k))),
                           interpolation=cv2.INTER_AREA)
        if src.suffix.lower() in (".jpg", ".jpeg"):
            cv2.imwrite(str(src), small, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            cv2.imwrite(str(src), small)
    except Exception:  # noqa: BLE001
        pass


def fetch_preset(name: str, out_root: str | Path, max_images: Optional[int] = None,
                 resize: Optional[int] = None, mirror: Optional[str] = None,
                 consecutive: bool = False, max_bytes: float = 3000.0,
                 progress=None) -> dict:
    """下载一个预设数据集，返回 {name, root, kind, images_dir, sparse_dir(或 None)}。"""
    if name not in PRESETS:
        raise KeyError(f"未知预设 {name}，可选：{', '.join(PRESETS)}")
    p = PRESETS[name]
    repo, allow, kind = p["repo"], p["allow"], p["kind"]
    LOG.info("[%s] %s", name, p["note"])

    files = list_remote(repo, mirror=mirror)
    picked = [f for f in files if _match(f["path"], allow)]
    images = [f for f in picked if f["path"].lower().endswith(IMG_EXT)]
    others = [f for f in picked if not f["path"].lower().endswith(IMG_EXT)]
    if max_images and len(images) > max_images:
        if consecutive and kind == "images":
            images = images[:max_images]
        else:
            step = max(1, len(images) // max_images)
            images = images[::step][:max_images]
    est = sum(f["size"] for f in images + others) / 1e6
    if est > max_bytes:
        raise RuntimeError(f"预计下载 {est:.0f} MB，超过上限 {max_bytes:.0f} MB；"
                           f"用 --max-images / --resize 缩小规模，或放宽 max_bytes")

    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    LOG.info("下载 %d 个文件（约 %.0f MB）-> %s", len(images) + len(others), est, out)
    done = 0
    for i, f in enumerate(images + others):
        rel = f["path"]
        if rel.startswith(name + "/"):
            rel = rel[len(name) + 1:]
        try:
            dest = download_file(repo, f["path"], out / rel, mirror=mirror)
            done += 1
            if resize and dest.suffix.lower() in IMG_EXT:
                _shrink(dest, resize)
        except Exception as e:  # noqa: BLE001
            LOG.warning("跳过 %s：%s", rel, str(e).splitlines()[0])
        if progress and (i % 5 == 0 or i == len(images + others) - 1):
            progress(f"下载 {i + 1}/{len(images + others)}")

    images_dir = out / "images" if (out / "images").is_dir() else out
    sparse_dir = None
    for cand in ("sparse/0", "sparse"):
        if (out / cand / "cameras.bin").exists() or (out / cand / "cameras.txt").exists():
            sparse_dir = out / cand
            break
    info = {"name": name, "root": str(out), "kind": kind, "repo": repo,
            "images_dir": str(images_dir),
            "sparse_dir": str(sparse_dir) if sparse_dir else None,
            "files": done, "note": p["note"]}
    (out / "dataset.json").write_text(json.dumps(info, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    LOG.info("下载完成：%d 个文件 -> %s", done, out)
    return info
