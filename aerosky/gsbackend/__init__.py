# -*- coding: utf-8 -*-
"""3D 高斯泼溅后端适配层。

本系统**不自研 3DGS 训练器**，而是对接业界现成的真实实现，按可用性自动选择：

    gsplat      nerfstudio 团队的 gsplat（PyTorch + CUDA 扩展，有 Windows 预编译轮子）
    official    原论文官方实现 graphdeco-inria/gaussian-splatting（CUDA 扩展，需编译）
    opensplat   OpenSplat（C++/libtorch，支持 CPU / CUDA / Metal）
    taichi      taichi_3d_gaussian_splatting（纯 Python + Taichi，CUDA）
    builtin     内置极简实现（无 CUDA 环境下的兜底，仅用于流程演示与预览）

用 `tools/setup_3dgs.py` 一键克隆/安装这些后端；用 `registry.detect_backends()` 看当前
环境哪些可用。三维重建入口 `aerosky.reconstruction.reconstruct()` 会自动挑选。
"""
from .base import GSBackend, GSResult  # noqa: F401
from .backends import (BuiltinBackend, GSplatBackend, OfficialBackend,  # noqa: F401
                       OpenSplatBackend, TaichiBackend)
from .registry import detect_backends, get_backend  # noqa: F401

__all__ = ["GSBackend", "GSResult", "GSplatBackend", "OfficialBackend",
           "OpenSplatBackend", "TaichiBackend", "BuiltinBackend",
           "get_backend", "detect_backends"]
