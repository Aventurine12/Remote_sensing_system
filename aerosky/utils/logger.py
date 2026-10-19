# -*- coding: utf-8 -*-
"""统一日志：同时输出到控制台与 logs/aerosky.log。"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from ..config import ROOT

_FMT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return
    # Windows 控制台默认 GBK，中文/符号输出会报编码错，这里统一成 UTF-8
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    log_dir = Path(ROOT) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("aerosky")
    root.setLevel(logging.INFO)
    root.propagate = False

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
    root.addHandler(sh)

    fh = logging.FileHandler(log_dir / "aerosky.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(_FMT))
    root.addHandler(fh)
    _configured = True


def get_logger(name: str = "aerosky") -> logging.Logger:
    _configure_root()
    return logging.getLogger(name if name.startswith("aerosky") else f"aerosky.{name}")


def setup(level: int = logging.INFO, logfile: str | None = None) -> None:
    """显式初始化日志（工具脚本用；不调用也会在首次 get_logger 时自动初始化）。"""
    _configure_root()
    root = logging.getLogger("aerosky")
    root.setLevel(level)
    if logfile:
        p = Path(logfile)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(p, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FMT))
        root.addHandler(fh)
