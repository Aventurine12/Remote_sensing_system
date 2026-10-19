# -*- coding: utf-8 -*-
"""aerosky 启动入口。

    python run_app.py                # 启动图形界面
    python run_app.py --check        # 只做环境自检
    python run_app.py --demo         # 无人值守跑一遍端到端（生成成果图）
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# OpenCV 自带的 Qt 插件会与 PyQt5 抢平台插件，必须在导入 cv2 之前指定 PyQt5 的插件目录
try:
    import PyQt5
    os.environ.setdefault(
        "QT_QPA_PLATFORM_PLUGIN_PATH",
        os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins", "platforms"))
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="空天协同的多光谱遥感影像智能解译与三维监测系统")
    ap.add_argument("--check", action="store_true", help="只做环境自检")
    ap.add_argument("--demo", action="store_true", help="无人值守跑一遍端到端流程")
    ap.add_argument("--reuse-3d", action="store_true",
                    help="端到端演示时复用已有的三维重建结果（跳过训练，省时间）")
    ap.add_argument("--device", default=None, help="cpu / cuda")
    a = ap.parse_args()

    if a.check:
        from tools.check_env import main as check
        return check()

    from aerosky.config import load_config
    cfg = load_config()

    if a.demo:
        from aerosky.demo import run_demo
        return run_demo(cfg, device=a.device, reuse_3d=a.reuse_3d)

    from PyQt5.QtWidgets import QApplication
    from aerosky.ui.main_window import MainWindow
    app = QApplication(sys.argv)
    app.setApplicationName("aerosky")
    app.setStyleSheet(APP_QSS)
    win = MainWindow(cfg)
    win.show()
    return app.exec_()


APP_QSS = """
QMainWindow, QWidget { background:#f2f4f7; color:#1b1f24;
    font-family:"Microsoft YaHei","Segoe UI",Arial; font-size:12px; }
QGroupBox { border:1px solid #c8d2dd; border-radius:6px; margin-top:12px; padding-top:10px;
    background:#ffffff; font-weight:600; }
QGroupBox::title { subcontrol-origin: margin; left:10px; padding:0 4px; color:#1c4b7a; }
QPushButton { background:#2f6fb5; color:#fff; border:none; border-radius:5px; padding:6px 8px; }
QPushButton:hover { background:#3a82cf; }
QPushButton:pressed { background:#255a94; }
QPushButton:disabled { background:#b9c3cf; color:#eef2f6; }
QTableWidget { background:#fff; gridline-color:#dde4ec; }
QHeaderView::section { background:#eaf1f8; border:none; padding:4px; }
QPlainTextEdit, QTextBrowser { background:#ffffff; border:1px solid #ccd5df; border-radius:4px; }
QTabWidget::pane { border:1px solid #ccd5df; background:#fff; }
QTabBar::tab { background:#e6ecf3; padding:5px 12px; border-top-left-radius:4px;
    border-top-right-radius:4px; }
QTabBar::tab:selected { background:#ffffff; color:#1c4b7a; font-weight:600; }
QProgressBar { border:1px solid #ccd5df; border-radius:4px; background:#fff; }
QProgressBar::chunk { background:#2f6fb5; border-radius:3px; }
QMenuBar { background:#e9eef4; } QMenuBar::item:selected { background:#d3e0ee; }
"""


if __name__ == "__main__":
    sys.exit(main())
