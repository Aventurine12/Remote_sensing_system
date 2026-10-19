# -*- coding: utf-8 -*-
"""权重下载与本地缓存：把 HuggingFace / GitHub 上的预训练权重落到 weights/ 目录。

系统设计成"权重在就用模型、权重不在就降级"，因此这里的失败不会中断主流程。
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Optional

from .logger import get_logger

LOG = get_logger("utils.weights")

# 语义分割（卫星影像建筑提取，实例分割）—— 由小到大，默认取 s
BUILDING_SEG_REPOS = [
    ("keremberke/yolov8s-building-segmentation", "best.pt", "yolov8s-building-segmentation.pt"),
    ("keremberke/yolov8n-building-segmentation", "best.pt", "yolov8n-building-segmentation.pt"),
    ("keremberke/yolov8m-building-segmentation", "best.pt", "yolov8m-building-segmentation.pt"),
]

# 目标检测：DOTAv1 航拍 15 类（plane/ship/.../large vehicle/small vehicle/tennis court...）
DOTA_OBB_ASSETS = [
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-obb.pt",
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-obb.pt",
]


def ensure_building_segmentation(weights_dir: str = "weights",
                                 prefer: str = "s") -> Optional[str]:
    """确保有一份建筑分割权重，返回本地路径；失败返回 None。"""
    Path(weights_dir).mkdir(parents=True, exist_ok=True)
    for name in ("yolov8s-building-segmentation.pt", "yolov8n-building-segmentation.pt",
                 "yolov8m-building-segmentation.pt"):
        p = os.path.join(weights_dir, name)
        if os.path.isfile(p) and os.path.getsize(p) > 1e6:
            return p
    order = BUILDING_SEG_REPOS
    if prefer == "n":
        order = [BUILDING_SEG_REPOS[1], BUILDING_SEG_REPOS[0], BUILDING_SEG_REPOS[2]]
    elif prefer == "m":
        order = [BUILDING_SEG_REPOS[2], BUILDING_SEG_REPOS[0], BUILDING_SEG_REPOS[1]]
    # 国内外网络环境差异很大：huggingface.co 不通时自动改用 hf-mirror.com
    endpoints = [os.environ.get("HF_ENDPOINT") or "https://huggingface.co",
                 "https://hf-mirror.com"]
    for endpoint in endpoints:
        os.environ["HF_ENDPOINT"] = endpoint
        for repo, fname, local in order:
            dst = os.path.join(weights_dir, local)
            try:
                LOG.info("从 %s 拉取 %s/%s ...", endpoint, repo, fname)
                from huggingface_hub import hf_hub_download
                src = hf_hub_download(repo_id=repo, filename=fname,
                                      cache_dir=os.path.join(weights_dir, ".hf"))
                shutil.copy2(src, dst)
                LOG.info("已保存 %s (%.1f MB)", dst, os.path.getsize(dst) / 1e6)
                return dst
            except Exception as e:  # noqa: BLE001
                LOG.warning("拉取 %s 失败：%s", repo, str(e).split("\n")[0])
    return None


def ensure_dota_obb(weights_dir: str = "weights") -> Optional[str]:
    """确保有一份 DOTAv1 旋转框检测权重（YOLO11-OBB / YOLOv8-OBB）。"""
    Path(weights_dir).mkdir(parents=True, exist_ok=True)
    for name in ("yolo11n-obb.pt", "yolov8n-obb.pt"):
        p = os.path.join(weights_dir, name)
        if os.path.isfile(p) and os.path.getsize(p) > 1e6:
            return p
    for url in DOTA_OBB_ASSETS:
        name = url.rsplit("/", 1)[-1]
        dst = os.path.join(weights_dir, name)
        try:
            LOG.info("下载 %s ...", url)
            _download(url, dst)
            return dst
        except Exception as e:  # noqa: BLE001
            LOG.warning("下载 %s 失败：%s", url, e)
    return None


def ensure_yolo_coco(weights_dir: str = "weights") -> Optional[str]:
    """COCO 通用检测权重，作为航拍权重的兜底。"""
    p = os.path.join(weights_dir, "yolov8n.pt")
    if os.path.isfile(p) and os.path.getsize(p) > 1e6:
        return p
    try:
        _download("https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt", p)
        return p
    except Exception as e:  # noqa: BLE001
        LOG.warning("下载 COCO 权重失败：%s", e)
        return None


def _download(url: str, dst: str, timeout: int = 600) -> str:
    import requests
    tmp = dst + ".part"
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
    os.replace(tmp, dst)
    return dst


def list_local_weights(weights_dir: str = "weights") -> List[dict]:
    p = Path(weights_dir)
    if not p.exists():
        return []
    out = []
    for f in sorted(p.glob("*")):
        if f.is_file() and f.suffix in (".pt", ".pth", ".onnx", ".ckpt"):
            out.append({"name": f.name, "size_mb": round(f.stat().st_size / 1e6, 2)})
    return out
