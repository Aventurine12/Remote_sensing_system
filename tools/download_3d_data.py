# -*- coding: utf-8 -*-
"""下载公开的多视角影像数据集（供 3D 高斯泼溅重建使用）。

    python tools/download_3d_data.py --list
    python tools/download_3d_data.py lego
    python tools/download_3d_data.py south-building --max-images 40 --resize 1200
    python tools/download_3d_data.py bigsur --max-images 30 --consecutive --resize 1000

数据来源全部是 HuggingFace 上的公开数据集（无需注册；不通时自动改用 hf-mirror.com）。
下载后：
  * 自带相机位姿的（kind=blender/colmap）→ 直接 `--data <目录>` 拿去重建
  * 只有照片的（kind=images）→ 先 `python tools/run_sfm.py` 算位姿
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aerosky.gsbackend.data import PRESETS, fetch_preset  # noqa: E402
from aerosky.utils.logger import get_logger, setup  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="下载多视角影像数据集")
    ap.add_argument("name", nargs="?", help="预设名（见 --list）")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=str(ROOT.parent / "data"))
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--resize", type=int, default=None, help="下载后把最长边缩到该像素")
    ap.add_argument("--consecutive", action="store_true",
                    help="只取前 N 张连续帧（视频/航拍抽帧做 SfM 必须连续）")
    ap.add_argument("--mirror", default=None)
    ap.add_argument("--max-bytes", type=float, default=3000.0)
    a = ap.parse_args()
    setup()

    if a.list or not a.name:
        print("可选数据集：\n")
        for k, v in PRESETS.items():
            print(f"  {k:<16} {v['note']}")
            print(f"  {'':<16} repo={v['repo']}  kind={v['kind']}")
        print("\n示例：python tools/download_3d_data.py lego")
        return 0

    def prog(line):
        print("  " + line, flush=True)

    info = fetch_preset(a.name, a.out, max_images=a.max_images, resize=a.resize,
                        mirror=a.mirror, consecutive=a.consecutive,
                        max_bytes=a.max_bytes, progress=prog)
    print("\n" + json.dumps(info, ensure_ascii=False, indent=2))
    if info.get("sparse_dir"):
        print(f"\n下一步（已带相机位姿，直接重建）：\n"
              f"  python tools/run_3dgs.py --images \"{info['images_dir']}\" "
              f"--sparse \"{info['sparse_dir']}\" --out outputs/3dgs")
    else:
        print(f"\n下一步（只有照片，先算位姿）：\n"
              f"  python tools/run_sfm.py --images \"{info['images_dir']}\" "
              f"--out \"{Path(info['root']) / 'colmap'}\"\n"
              f"  python tools/run_3dgs.py --data \"{Path(info['root']) / 'colmap'}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
