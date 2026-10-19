# -*- coding: utf-8 -*-
"""交互式查看 3D 高斯模型（.ply）。

    python tools/view_3dgs.py --model outputs/3dgs/point_cloud.ply

左键拖拽旋转 · 滚轮缩放 · 右键平移 · R 重置视角 · S 存图。
渲染用的是系统内置的纯 numpy 前向光栅化预览器（只是"看图"，不影响训练用的实现）。
也可以把同一个 .ply 直接拖进 https://supersplat.dev 用浏览器看。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import PyQt5
    os.environ.setdefault(
        "QT_QPA_PLATFORM_PLUGIN_PATH",
        os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins", "platforms"))
except Exception:  # noqa: BLE001
    pass

from PyQt5.QtWidgets import QApplication  # noqa: E402

from aerosky.reconstruction import load_renderer  # noqa: E402
from aerosky.ui.viewer3d import SplatView  # noqa: E402
from aerosky.utils.logger import setup  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="查看 3D 高斯模型")
    ap.add_argument("--model", required=True, help="point_cloud.ply")
    ap.add_argument("--scale", type=float, default=0.6, help="渲染分辨率比例（0.3~1.0）")
    a = ap.parse_args()
    setup()
    r = load_renderer(a.model)
    app = QApplication(sys.argv[:1])
    w = SplatView()
    w.setWindowTitle("3D 高斯泼溅查看器 — " + Path(a.model).name)
    w.resize(1100, 760)
    w.render_scale = float(a.scale)
    w.set_renderer(r)
    w.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
