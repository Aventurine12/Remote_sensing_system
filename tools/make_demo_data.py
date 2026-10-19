# -*- coding: utf-8 -*-
"""生成离线演示数据（前后时相 GeoTIFF）。

    python tools/make_demo_data.py --out outputs/demo --size 2048 1536
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aerosky.utils.geo import Raster, make_demo_geotiff


def main() -> int:
    ap = argparse.ArgumentParser(description="生成带地理参考的演示用多光谱影像")
    ap.add_argument("--out", default="outputs/demo")
    ap.add_argument("--size", nargs=2, type=int, default=[2048, 1536])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--lon", type=float, default=126.6320)
    ap.add_argument("--lat", type=float, default=45.7520)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    size = (a.size[0], a.size[1])
    p1 = make_demo_geotiff(out / "satellite_before.tif", size=size, seed=a.seed,
                           with_change=False)
    p2 = make_demo_geotiff(out / "satellite_after.tif", size=size, seed=a.seed,
                           with_change=True)
    r = Raster.open(p1)
    print(f"前时相：{p1}")
    print(f"后时相：{p2}")
    print(f"尺寸 {r.width}×{r.height}，分辨率 {r.pixel_size_m()[0]:.3f} m/px，"
          f"范围 {r.bounds_lonlat()}")
    print("后时相相对前时相新增了若干建筑，可用于变化检测与三维重建联动演示。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
