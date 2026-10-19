# -*- coding: utf-8 -*-
"""预下载模型权重到 weights/ 目录。

    python tools/download_weights.py            # 下载全部
    python tools/download_weights.py --seg      # 只下载建筑分割
    python tools/download_weights.py --det      # 只下载 DOTA 检测
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aerosky.config import load_config
from aerosky.utils.weights import (ensure_building_segmentation, ensure_dota_obb,
                                   ensure_yolo_coco, list_local_weights)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", action="store_true")
    ap.add_argument("--det", action="store_true")
    ap.add_argument("--prefer", default="s", choices=["n", "s", "m"],
                    help="建筑分割模型大小：n 最快 / s 平衡 / m 最准")
    a = ap.parse_args()
    do_all = not (a.seg or a.det)

    wdir = load_config().get_path("paths.weights")
    print(f"权重目录：{wdir}\n")
    if a.seg or do_all:
        p = ensure_building_segmentation(wdir, prefer=a.prefer)
        print("建筑分割：", p or "失败（系统会自动降级到传统算法）")
    if a.det or do_all:
        p = ensure_dota_obb(wdir)
        print("DOTA 航拍检测：", p or "失败，尝试 COCO 通用权重…")
        if not p:
            print("COCO 通用检测：", ensure_yolo_coco(wdir) or "失败")
    print("\n本地权重：")
    for w in list_local_weights(wdir):
        print(f"  {w['name']:<44} {w['size_mb']:>8.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
