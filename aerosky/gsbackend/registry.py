# -*- coding: utf-8 -*-
"""后端注册与自动选择。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from .backends import (BuiltinBackend, GSplatBackend, OfficialBackend,
                       OpenSplatBackend, TaichiBackend)
from .base import GSBackend
from ..utils.logger import get_logger

LOG = get_logger("gsbackend.registry")

BACKENDS: Dict[str, type] = {
    "gsplat": GSplatBackend,
    "official": OfficialBackend,
    "opensplat": OpenSplatBackend,
    "taichi": TaichiBackend,
    "builtin": BuiltinBackend,
}

# 自动选择时的优先级：越靠前越优先（真实实现优先，内置兜底最后）
PRIORITY = ["gsplat", "official", "opensplat", "taichi", "builtin"]

# 各后端默认使用的 Python 解释器（可以指向不同的环境）
DEFAULT_PYTHON_KEYS = {
    "gsplat": ("gsplat_python", "AEROSKY_GSPLAT_PYTHON"),
    "official": ("official_python", "AEROSKY_OFFICIAL_PYTHON"),
    "taichi": ("taichi_python", "AEROSKY_TAICHI_PYTHON"),
}

# 常见候选解释器位置（自动探测用）
_KNOWN_PYTHONS = [
    r"D:\Code\dc\gs3d_env\Scripts\python.exe",      # 由 rs_seg 的 CUDA torch 派生
    r"D:\Code\dc\3dgs_env\python.exe",
    os.path.expanduser(r"~\miniconda3\envs\gs310\python.exe"),
    r"C:\miniconda3\envs\gs310\python.exe",
    r"C:\ProgramData\miniconda3\envs\gs310\python.exe",
]


def _resolve_python(name: str, cfg: dict | None) -> Optional[str]:
    cfg = cfg or {}
    keys = DEFAULT_PYTHON_KEYS.get(name, ())
    for k in keys:
        if cfg.get(k):
            return str(cfg[k])
        if k.startswith("AEROSKY_") and os.environ.get(k):
            return os.environ[k]
    if cfg.get("python_exe"):
        return str(cfg["python_exe"])
    for p in _KNOWN_PYTHONS:
        if p and Path(p).exists():
            return p
    import sys
    return sys.executable


def get_backend(name: str, cfg: dict | None = None,
                python_exe: Optional[str] = None) -> GSBackend:
    if name not in BACKENDS:
        raise KeyError(f"未知后端 {name}，可选：{list(BACKENDS)}")
    py = python_exe or _resolve_python(name, cfg)
    return BACKENDS[name](cfg, python_exe=py)


def detect_backends(cfg: dict | None = None, verbose: bool = True) -> List[dict]:
    """逐个探测后端可用性。返回 [{name, title, available, reason, python}]。"""
    out = []
    for name in PRIORITY:
        try:
            b = get_backend(name, cfg)
            ok, why = b.available()
        except Exception as e:  # noqa: BLE001
            ok, why, b = False, f"探测异常：{e}", None
        out.append({"name": name, "title": BACKENDS[name].title,
                    "available": bool(ok), "reason": why,
                    "python": getattr(b, "python_exe", "")})
        if verbose:
            LOG.info("[%s] %-6s %s", "✓" if ok else "×", name, why)
    return out


def autoselect(cfg: dict | None = None, prefer: Optional[str] = None,
               verbose: bool = True) -> GSBackend:
    """按优先级挑一个可用后端；prefer 指定则优先用它。"""
    order = ([prefer] if prefer and prefer in BACKENDS else []) + \
            [n for n in PRIORITY if n != prefer]
    for name in order:
        try:
            b = get_backend(name, cfg)
            ok, why = b.available()
        except Exception as e:  # noqa: BLE001
            ok, why = False, str(e)
        if ok:
            if verbose:
                LOG.info("选用 3DGS 后端：%s（%s）", name, why)
            return b
        if verbose and prefer == name:
            LOG.warning("指定的后端 %s 不可用：%s", name, why)
    if verbose:
        LOG.warning("没有任何 3DGS 后端可用，请运行 tools/setup_3dgs.py")
    return get_backend("builtin", cfg)
