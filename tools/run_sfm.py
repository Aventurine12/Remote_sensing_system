# -*- coding: utf-8 -*-
"""运动恢复结构（SfM）：把一批普通照片算成相机位姿 + 稀疏点云。

没有无人机也没关系——任何一组同一场景/物体的多视角照片都能用。
内置 pycolmap（COLMAP 官方 Python 绑定，pip 装、免编译）。

    python tools/run_sfm.py --images D:\我的照片 --out D:\我的照片\colmap
    python tools/run_sfm.py --images data/bigsur/images --out data/bigsur/colmap --matcher sequential
    python tools/run_sfm.py --images D:\我的照片 --check-only     # 先看看照片合不合适
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aerosky.utils.logger import get_logger, setup  # noqa: E402

LOG = get_logger("run_sfm")
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def check_images(folder: Path) -> dict:
    import cv2
    import numpy as np
    files = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in IMG_EXT]
    sizes, ok = [], 0
    for p in files[:300]:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            continue
        ok += 1
        sizes.append(im.shape[:2])
    if not sizes:
        return {"images": len(files), "readable": 0}
    a = np.array(sizes)
    return {"images": len(files), "readable": ok,
            "median_hw": [int(np.median(a[:, 0])), int(np.median(a[:, 1]))],
            "min_hw": [int(a[:, 0].min()), int(a[:, 1].min())]}


def run_sfm(images_dir: Path, out_dir: Path, matcher: str, camera_model: str,
            max_image_size: int, max_images=None, gpu: bool = False) -> dict:
    import pycolmap

    images_dir = images_dir.resolve()
    out_dir = out_dir.resolve()
    imgs = [p for p in sorted(images_dir.rglob("*")) if p.suffix.lower() in IMG_EXT]
    if len(imgs) < 3:
        raise RuntimeError(f"照片太少（{len(imgs)} 张），SfM 至少 3 张，建议 20 张以上")
    if max_images and len(imgs) > max_images:
        sub = images_dir.parent / (images_dir.name + "_subset")
        sub.mkdir(parents=True, exist_ok=True)
        step = max(1, len(imgs) // max_images)
        for p in imgs[::step][:max_images]:
            shutil.copy2(p, sub / p.name)
        LOG.info("照片较多，抽 %d 张到 %s", len(list(sub.glob('*'))), sub)
        images_dir = sub

    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)
    db = work / "database.db"
    sparse = out_dir / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    LOG.info("[1/3] 提取特征 …")
    try:
        ro = pycolmap.ImageReaderOptions()
    except Exception:  # noqa: BLE001
        ro = None
    try:
        eo = pycolmap.FeatureExtractionOptions()
        for k, v in (("max_image_size", int(max_image_size)), ("num_threads", -1)):
            if hasattr(eo, k):
                setattr(eo, k, v)
    except Exception:  # noqa: BLE001
        eo = None
    pycolmap.extract_features(str(db), str(images_dir), camera_mode=pycolmap.CameraMode.AUTO,
                              camera_model=camera_model, reader_options=ro,
                              extraction_options=eo)
    LOG.info("    %.1fs", time.time() - t0)

    LOG.info("[2/3] 特征匹配（%s）…", matcher)
    mo = None
    try:
        mo = pycolmap.FeatureMatchingOptions()
        if hasattr(mo, "use_gpu"):
            mo.use_gpu = bool(gpu)
    except Exception:  # noqa: BLE001
        mo = None
    fn = pycolmap.match_sequential if matcher == "sequential" else pycolmap.match_exhaustive
    try:
        fn(str(db), matching_options=mo)
    except TypeError:
        fn(str(db))

    LOG.info("[3/3] 增量式重建 …")
    try:
        recs = pycolmap.incremental_mapping(str(db), str(images_dir), str(sparse),
                                            options=pycolmap.IncrementalPipelineOptions())
    except Exception:  # noqa: BLE001
        recs = pycolmap.incremental_mapping(str(db), str(images_dir), str(sparse))
    if isinstance(recs, dict):
        recs = list(recs.values())
    if not recs:
        raise RuntimeError("SfM 失败：没有重建出模型。常见原因：照片重叠不足、场景纹理太弱"
                           "（白墙/水面）、照片太少。建议同一定点绕物体拍 20~60 张、重叠 60% 以上。")
    best = max(recs, key=lambda r: len(r.images))
    dst = sparse / "0"
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)
    best.write(str(dst))

    link = out_dir / "images"
    if not link.exists():
        try:
            link.symlink_to(images_dir, target_is_directory=True)
        except Exception:  # noqa: BLE001
            shutil.copytree(images_dir, link, dirs_exist_ok=True)

    info = {"images_dir": str(images_dir), "model_dir": str(dst), "out": str(out_dir),
            "num_reg_images": len(best.images), "total_photos": len(imgs),
            "seconds": round(time.time() - t0, 1)}
    try:
        info["num_points3D"] = int(best.num_points3D())
    except Exception:  # noqa: BLE001
        info["num_points3D"] = 0
    if info["num_reg_images"] < max(3, len(imgs) * 0.4):
        LOG.warning("注册率偏低（%d/%d），重建质量可能不佳", info["num_reg_images"], len(imgs))
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description="照片 -> 相机位姿（SfM）")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default=None, help="默认 <images>/../colmap")
    ap.add_argument("--matcher", default="exhaustive", choices=["exhaustive", "sequential"])
    ap.add_argument("--camera-model", default="SIMPLE_RADIAL")
    ap.add_argument("--max-image-size", type=int, default=2000)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    setup()

    imgs = Path(a.images)
    info = check_images(imgs)
    print("照片检查：", json.dumps(info, ensure_ascii=False))
    if info.get("images", 0) < 3:
        LOG.error("可用照片不足")
        return 2
    if a.check_only:
        return 0
    out = Path(a.out) if a.out else (imgs.parent / "colmap")
    res = run_sfm(imgs, out, a.matcher, a.camera_model, a.max_image_size,
                  a.max_images, a.gpu)
    (out / "sfm_info.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print("\n" + "=" * 66)
    print(f"  稀疏模型：{res['model_dir']}")
    print(f"  注册影像：{res['num_reg_images']}/{res['total_photos']}")
    print(f"  下一步：python tools/run_3dgs.py --data \"{out}\" --out outputs/3dgs")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
