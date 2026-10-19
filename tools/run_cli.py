# -*- coding: utf-8 -*-
"""命令行版端到端流程（不依赖 GUI）。

    # 语义分割
    python tools/run_cli.py segment --image outputs/demo/satellite_before.tif
    # 目标检测
    python tools/run_cli.py detect  --image outputs/demo/satellite_before.tif
    # 变化检测
    python tools/run_cli.py change  --before a.tif --after b.tif
    # 三维重建（无 COLMAP 结果时自动用合成场景）
    python tools/run_cli.py recon   --out outputs/3d
    # 一次性跑完全流程并出报告
    python tools/run_cli.py all     --out outputs/cli_run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from aerosky.config import load_config
from aerosky.pipeline import Pipeline
from aerosky.reconstruction import load_renderer, reconstruct
from aerosky.registration import register_to_raster
from aerosky.report.builder import ReportContext, export_report
from aerosky.report.uav_task import plan_flight
from aerosky.utils.geo import Raster, make_demo_geotiff, save_geojson


def main() -> int:
    ap = argparse.ArgumentParser(description="aerosky 命令行")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("segment"); p.add_argument("--image", required=True)
    p.add_argument("--out", default=None); p.add_argument("--device", default=None)
    p = sub.add_parser("detect"); p.add_argument("--image", required=True)
    p.add_argument("--out", default=None); p.add_argument("--device", default=None)
    p = sub.add_parser("change"); p.add_argument("--before", required=True)
    p.add_argument("--after", required=True); p.add_argument("--out", default=None)
    p.add_argument("--device", default=None)
    p = sub.add_parser("recon"); p.add_argument("--out", required=True)
    p.add_argument("--images", default=None); p.add_argument("--sparse", default=None)
    p.add_argument("--iters", type=int, default=None); p.add_argument("--device", default=None)
    p = sub.add_parser("all"); p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=None); p.add_argument("--device", default=None)
    a = ap.parse_args()

    cfg = load_config()
    if a.cmd == "recon":
        res = reconstruct(a.out, cfg.gsplat, device=a.device or _dev(),
                          images_dir=a.images, sparse_dir=a.sparse, iterations=a.iters)
        print(json.dumps(res.summary(), ensure_ascii=False, indent=2))
        return 0

    if a.cmd == "all":
        from aerosky.demo import run_demo
        return run_demo(cfg, out_root=a.out, device=a.device)

    pipe = Pipeline(cfg, device=a.device)
    out = Path(a.out or Path(cfg.get_path("paths.output")) / "cli")
    out.mkdir(parents=True, exist_ok=True)

    if a.cmd == "segment":
        r = Raster.open(a.image)
        res = pipe.segment(r)
        print(json.dumps(res.summary(), ensure_ascii=False, indent=2))
        print("GeoJSON:", pipe.save_geojson(res, out / "segmentation.geojson"))
    elif a.cmd == "detect":
        r = Raster.open(a.image)
        res = pipe.detect(r)
        print(json.dumps(res.summary(), ensure_ascii=False, indent=2))
        print("GeoJSON:", pipe.save_geojson(res, out / "detection.geojson"))
    elif a.cmd == "change":
        ra, rb = Raster.open(a.before), Raster.open(a.after)
        res = pipe.change(ra, rb)
        print(json.dumps(res.summary(), ensure_ascii=False, indent=2))
        print("GeoJSON:", pipe.save_geojson(res, out / "change.geojson"))
    return 0


def _dev():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


if __name__ == "__main__":
    sys.exit(main())
