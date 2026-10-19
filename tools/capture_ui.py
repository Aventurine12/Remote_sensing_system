# -*- coding: utf-8 -*-
"""离屏渲染主界面截图（用于报告/答辩材料，也用来验证 GUI 能正常构建）。

    python tools/capture_ui.py --data outputs/demo_run --out outputs/ui_shot.png

不打开真实窗口：使用 Qt 的 offscreen 平台插件把窗口渲染成 PNG。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))
try:
    import PyQt5
    os.environ.setdefault(
        "QT_QPA_PLATFORM_PLUGIN_PATH",
        os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins", "platforms"))
except Exception:  # noqa: BLE001
    pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/demo_run")
    ap.add_argument("--out", default="outputs/ui_shot.png")
    ap.add_argument("--ply", default=None)
    ap.add_argument("--size", nargs=2, type=int, default=[1600, 950])
    ap.add_argument("--skip-inference", action="store_true")
    a = ap.parse_args()

    from PyQt5.QtWidgets import QApplication
    from aerosky.config import load_config
    from aerosky.ui.main_window import MainWindow
    from aerosky.utils.geo import Raster

    app = QApplication(sys.argv[:1])
    cfg = load_config()
    win = MainWindow(cfg)
    win.resize(a.size[0], a.size[1])

    data = Path(a.data)
    before = data / "real" / "satellite_before.tif"
    after = data / "real" / "satellite_after.tif"
    if not before.exists():
        before = data / "satellite_before.tif"
        after = data / "satellite_after.tif"
    if before.exists():
        win.raster_a = Raster.open(before)
        win._show_raster()
        win._update_data_label()
        print("已载入影像：", before)

    if not a.skip_inference and win.raster_a is not None:
        print("运行语义分割 …")
        res = win.pipe.segment(win.raster_a)
        win.results["segmentation"] = res
        print("  模型", res.model_name, "耗时 %.1fs" % res.elapsed,
              "图斑", res.stats.get("num_polygons"))

    if win.raster_a is not None and after.exists():
        win.raster_b = Raster.open(after)
        try:
            print("运行变化检测 …")
            chg = win.pipe.change(win.raster_a, win.raster_b)
            win.results["change_detection"] = chg
            print("  变化图斑", chg.stats.get("num_changes"))
        except Exception as e:  # noqa: BLE001
            print("  变化检测失败：", e)

    win._refresh_layers()
    win._refresh_table()
    win._refresh_info()

    ply = Path(a.ply) if a.ply else None
    if ply is None:
        for cand in (data / "3dgs" / "point_cloud.ply",):
            if cand.exists():
                ply = cand
    if ply and ply.exists():
        try:
            from aerosky.reconstruction import load_renderer
            win.view3d.set_renderer(load_renderer(str(ply)))
            win.view3d.reset_view()
            print("三维视图已载入：", ply)
        except Exception as e:  # noqa: BLE001
            print("三维载入失败：", e)

    win.log("界面截图已生成（离屏渲染）")
    app.processEvents()
    win.show()
    app.processEvents()
    # 让三维视图先完成一帧渲染
    win.view3d._dirty = True
    win.view3d.repaint()
    app.processEvents()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pix = win.grab()
    pix.save(str(out))
    print("已保存：", out, pix.width(), "x", pix.height())

    # 再截一张切到三维视图的
    win.view3d.parentWidget().parentWidget().setCurrentIndex(0) if hasattr(
        win.view3d.parentWidget().parentWidget(), "setCurrentIndex") else None
    app.processEvents()
    win.view3d._dirty = True
    win.view3d.repaint()
    app.processEvents()
    out3d = out.with_name(out.stem + "_3d" + out.suffix)
    pix3 = win.grab()
    pix3.save(str(out3d))
    print("已保存：", out3d, pix3.width(), "x", pix3.height())
    return 0


if __name__ == "__main__":
    sys.exit(main())
