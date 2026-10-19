# -*- coding: utf-8 -*-
"""命令行跑 3D 高斯泼溅重建（走系统的后端适配层）。

    # 看当前有哪些后端可用
    python tools/run_3dgs.py --list

    # 用带位姿的数据集重建
    python tools/run_3dgs.py --images data/south-building/images \
                             --sparse data/south-building/sparse/0 \
                             --out outputs/3dgs --iters 7000
    # 或者直接给 COLMAP 目录
    python tools/run_3dgs.py --data data/south-building --out outputs/3dgs

    # 指定后端
    python tools/run_3dgs.py --data data/lego --backend gsplat
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aerosky.config import load_config  # noqa: E402
from aerosky.gsbackend.registry import detect_backends  # noqa: E402
from aerosky.reconstruction import reconstruct  # noqa: E402
from aerosky.utils.logger import setup  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="3D 高斯泼溅重建")
    ap.add_argument("--list", action="store_true", help="列出可用后端")
    ap.add_argument("--data", default=None, help="COLMAP 数据目录（含 images/ 与 sparse/0/）")
    ap.add_argument("--images", default=None)
    ap.add_argument("--sparse", default=None)
    ap.add_argument("--out", default="outputs/3dgs")
    ap.add_argument("--backend", default=None,
                    choices=[None, "gsplat", "official", "opensplat", "taichi", "builtin"])
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--sh", type=int, default=3)
    ap.add_argument("--downscale", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--time-budget", type=float, default=0, help="训练时间上限（分钟）")
    a = ap.parse_args()
    setup()
    cfg = load_config()

    if a.list:
        print("3DGS 后端：\n")
        for d in detect_backends(cfg, verbose=False):
            print(f"  [{'✓' if d['available'] else '×'}] {d['name']:<9} {d['title']}")
            print(f"      {d['reason']}")
        return 0

    images, sparse = a.images, a.sparse
    if a.data:
        root = Path(a.data)
        images = images or str(root / "images" if (root / "images").is_dir() else root)
        for cand in ("sparse/0", "sparse"):
            if (root / cand / "cameras.bin").exists() or (root / cand / "cameras.txt").exists():
                sparse = sparse or str(root / cand)
                break
    if not images or not sparse:
        ap.error("请给 --data（COLMAP 目录）或同时给 --images 与 --sparse")

    gs = dict(cfg.get_path("gsplat", {}))
    if a.iters:
        gs["iterations"] = a.iters
    gs["sh_degree"] = a.sh
    gs["time_budget"] = a.time_budget * 60 if a.time_budget else None

    def prog(line):
        print(line.rstrip(), flush=True)

    res = reconstruct(a.out, gs, device=a.device, images_dir=images, sparse_dir=sparse,
                      iterations=a.iters, backend=a.backend, app_cfg=cfg,
                      downscale=a.downscale, progress=prog)
    print("\n" + json.dumps(res.summary(), ensure_ascii=False, indent=2))
    print(f"\n模型：{res.ply}\n查看：python tools/view_3dgs.py --model \"{res.ply}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
