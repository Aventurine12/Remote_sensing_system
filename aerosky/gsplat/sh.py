# -*- coding: utf-8 -*-
"""球谐函数（Spherical Harmonics）：3D 高斯用低阶球谐系数编码视角相关的颜色。

本项目把球谐阶数由 3DGS 原论文的 3 阶降到 1 阶（每高斯 4 组系数而不是 16 组），
在保持真实感的同时把参数量降到约 1/4——对应"参数精简"的轻量化策略。
"""
from __future__ import annotations

import torch

# 实球谐基的归一化常数（与 3D Gaussian Splatting 官方实现一致）
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
      -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
      -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def num_coeffs(degree: int) -> int:
    return (degree + 1) ** 2


def eval_sh(degree: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """求值球谐。

    Args:
        degree: 阶数 0~3
        sh:     (N, (degree+1)^2, 3) 球谐系数
        dirs:   (N, 3) 单位方向（由高斯指向相机）
    Returns:
        (N, 3) 的颜色（尚未加 0.5 偏置、未 clamp）
    """
    result = C0 * sh[:, 0]
    if degree > 0:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        result = result - C1 * y * sh[:, 1] + C1 * z * sh[:, 2] - C1 * x * sh[:, 3]
        if degree > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result
                      + C2[0] * xy * sh[:, 4] + C2[1] * yz * sh[:, 5]
                      + C2[2] * (2 * zz - xx - yy) * sh[:, 6]
                      + C2[3] * xz * sh[:, 7] + C2[4] * (xx - yy) * sh[:, 8])
            if degree > 2:
                result = (result
                          + C3[0] * y * (3 * xx - yy) * sh[:, 9]
                          + C3[1] * xy * z * sh[:, 10]
                          + C3[2] * y * (4 * zz - xx - yy) * sh[:, 11]
                          + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12]
                          + C3[4] * x * (4 * zz - xx - yy) * sh[:, 13]
                          + C3[5] * z * (xx - yy) * sh[:, 14]
                          + C3[6] * x * (xx - 3 * yy) * sh[:, 15])
    return result


def rgb_to_sh(rgb) -> torch.Tensor:
    """把 RGB(0~1) 转成 0 阶球谐系数（用于初始化）。"""
    return (torch.as_tensor(rgb) - 0.5) / C0


# --------------------------------------------------------------------- numpy 版
def eval_sh_np(degree: int, sh, dirs):
    """numpy 版本，供不依赖 torch 的实时预览渲染器使用。"""
    import numpy as np
    result = C0 * sh[:, 0]
    if degree > 0:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        result = result - C1 * y * sh[:, 1] + C1 * z * sh[:, 2] - C1 * x * sh[:, 3]
        if degree > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result + C2[0] * xy * sh[:, 4] + C2[1] * yz * sh[:, 5]
                      + C2[2] * (2 * zz - xx - yy) * sh[:, 6]
                      + C2[3] * xz * sh[:, 7] + C2[4] * (xx - yy) * sh[:, 8])
    return np.asarray(result)
