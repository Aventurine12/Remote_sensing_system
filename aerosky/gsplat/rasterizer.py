# -*- coding: utf-8 -*-
"""可微分 3D 高斯光栅化器（纯 PyTorch，CPU / CUDA 均可跑，无需编译 CUDA 扩展）。

与 3D Gaussian Splatting 原论文（Kerbl et al., SIGGRAPH 2023）一致的流程：

    1. 由 旋转四元数 R 与 尺度 S 组成三维协方差  Σ = R S Sᵀ Rᵀ
    2. 世界点投影到相机，用雅可比  J 做局部仿射近似，得到二维协方差
           Σ' = J W Σ Wᵀ Jᵀ      (W 为世界->相机的旋转)
    3. 二维高斯按不透明度沿深度排序，做前向 alpha 合成：
           C = Σ_i T_i α_i c_i + T_end · C_bg ,  T_i = Π_{j<i}(1-α_j)
    4. 颜色由球谐系数按视角方向求值得到

原实现用 CUDA 瓦片并行；这里用「包围盒展开 + 逐像素前缀积」的向量化写法，
在 CPU 上也能对十万级高斯做前向渲染，并且全程可导，可直接反向传播优化高斯参数。
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch

from .sh import eval_sh


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """四元数 (N,4) [w,x,y,z] -> 旋转矩阵 (N,3,3)。"""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(-1)
    N = q.shape[0]
    R = torch.empty((N, 3, 3), dtype=q.dtype, device=q.device)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_cov3d(scales: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    """Σ = R S Sᵀ Rᵀ，返回 (N,3,3)。scales 需为已激活的正数。"""
    R = quat_to_rotmat(quats)
    M = R * scales[:, None, :]                # R @ diag(s)
    return M @ M.transpose(1, 2)


def _pairs_from_bbox(px: torch.Tensor, py: torch.Tensor, radius: torch.Tensor,
                     width: int, height: int):
    """把每个高斯的包围盒展开成 (高斯下标, 像素x, 像素y) 的候选对（全向量化）。"""
    r = torch.clamp(radius.ceil().long(), min=0)
    side = 2 * r + 1
    counts = side * side
    total = int(counts.sum().item())
    if total == 0:
        empty = torch.zeros(0, dtype=torch.long, device=px.device)
        return empty, empty, empty
    gid = torch.repeat_interleave(torch.arange(px.numel(), device=px.device), counts)
    starts = torch.cumsum(counts, 0) - counts
    ar = torch.arange(total, device=px.device)
    local = ar - starts[gid]
    w = side[gid]
    dy = local // w - r[gid]
    dx = local % w - r[gid]
    gx = px[gid].round().long() + dx
    gy = py[gid].round().long() + dy
    # 越界裁剪的核心是把坐标 clamp 回画面内，再用有效掩码筛掉重复
    valid = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height)
    return gid[valid], gx[valid], gy[valid]


def render_gaussians(means: torch.Tensor, scales: torch.Tensor, quats: torch.Tensor,
                     opacities: torch.Tensor, shs: torch.Tensor,
                     viewmat: torch.Tensor, K: torch.Tensor,
                     width: int, height: int, sh_degree: int = 1,
                     bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                     radius_scale: float = 3.0, eps2d: float = 0.3,
                     max_pairs: int = 60_000_000
                     ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """渲染一帧。

    Args:
        means/scales/quats/opacities/shs: (N,3)/(N,3)/(N,4)/(N,)/(N,(d+1)^2,3)
        viewmat: (4,4) 世界->相机
        K: (3,3) 内参
        width/height: 输出尺寸
    Returns:
        image (H,W,3), alpha (H,W), info
    """
    device = means.device
    N = means.shape[0]
    viewmat = viewmat.to(device=means.device, dtype=means.dtype)
    K = K.to(device=means.device, dtype=means.dtype)
    bg = torch.tensor(bg_color, dtype=means.dtype, device=device)
    info = {"num_gaussians": int(N), "num_pairs": 0, "visible": 0}
    if N == 0:
        return bg.expand(height, width, 3).clone(), torch.zeros(height, width, device=device), info

    W3 = viewmat[:3, :3]
    t3 = viewmat[:3, 3]
    p_cam = means @ W3.transpose(0, 1) + t3                     # (N,3)
    z = p_cam[:, 2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 视锥剔除：深度为正且在画面外留有余量的
    keep = z > 1e-3
    u = fx * p_cam[:, 0] / z.clamp_min(1e-6) + cx
    v = fy * p_cam[:, 1] / z.clamp_min(1e-6) + cy
    keep &= (u > -200) & (u < width + 200) & (v > -200) & (v < height + 200)
    if keep.sum() == 0:
        return bg.expand(height, width, 3).clone(), torch.zeros(height, width, device=device), info

    means_k, scales_k, quats_k = means[keep], scales[keep], quats[keep]
    opac_k, shs_k = opacities[keep], shs[keep]
    p_cam, z, u, v = p_cam[keep], z[keep], u[keep], v[keep]
    info["visible"] = int(keep.sum().item())

    # ---- 三维协方差 -> 二维协方差
    cov3d = build_cov3d(scales_k, quats_k)
    J = torch.zeros((p_cam.shape[0], 2, 3), dtype=means.dtype, device=device)
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx * p_cam[:, 0] / z.clamp_min(1e-6) ** 2
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -fy * p_cam[:, 1] / z.clamp_min(1e-6) ** 2
    T = J @ W3                                                   # (N,2,3)
    cov2d = T @ cov3d @ T.transpose(1, 2)                        # (N,2,2)
    cov2d[:, 0, 0] += eps2d
    cov2d[:, 1, 1] += eps2d
    a = cov2d[:, 0, 0]
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1]
    det = (a * c - b * b).clamp_min(1e-9)
    # 二次型矩阵的逆： (a b; b c)^-1 = 1/det * (c -b; -b a)
    inv_a, inv_b, inv_c = c / det, -b / det, a / det
    mid = 0.5 * (a + c)
    lam = mid + torch.sqrt((mid * mid - det).clamp_min(0.0))
    radius = radius_scale * torch.sqrt(lam.clamp_min(1e-9))
    # 屏幕尺寸过大的高斯既费算力又几乎不影响画面（3DGS 原实现同样会跳过它们），
    # 这里统一夹到画面短边的一定比例，并把中心落在画面外太远的直接剔除。
    max_r = max(6.0, 0.14 * min(width, height))
    radius = radius.clamp(max=max_r)
    inside = (u > -max_r) & (u < width + max_r) & (v > -max_r) & (v < height + max_r)
    if inside.sum() == 0:
        return bg.expand(height, width, 3).clone(), torch.zeros(height, width, device=device), info

    # ---- 包围盒展开成候选像素对
    gid, gx, gy = _pairs_from_bbox(u, v, radius, width, height)
    if gid.numel():
        sel = inside[gid]                       # 丢弃中心远在画面外的高斯
        gid, gx, gy = gid[sel], gx[sel], gy[sel]
    if gid.numel() == 0:
        return bg.expand(height, width, 3).clone(), torch.zeros(height, width, device=device), info
    if gid.numel() > max_pairs:                                   # 保护性降采样
        sel = torch.randperm(gid.numel(), device=device)[:max_pairs]
        gid, gx, gy = gid[sel], gx[sel], gy[sel]
    info["num_pairs"] = int(gid.numel())

    dx = gx.to(means.dtype) + 0.5 - u[gid]
    dy = gy.to(means.dtype) + 0.5 - v[gid]
    power = -0.5 * (inv_a[gid] * dx * dx + 2 * inv_b[gid] * dx * dy + inv_c[gid] * dy * dy)
    alpha = opac_k[gid] * torch.exp(power)
    alpha = alpha.clamp(0.0, 0.99)

    # ---- 颜色（球谐按视角方向求值）
    campos = -W3.transpose(0, 1) @ t3
    dirs = means_k[gid] - campos[None, :]
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    color = eval_sh(sh_degree, shs_k[gid], dirs) + 0.5
    color = color.clamp_min(0.0)

    # ---- 按 (像素, 深度) 稳定排序
    zs = z[gid]
    o1 = torch.argsort(zs, stable=True)
    pix = (gy * width + gx)
    pix, alpha, color, zs = pix[o1], alpha[o1], color[o1], zs[o1]
    o2 = torch.argsort(pix, stable=True)
    pix, alpha, color = pix[o2], alpha[o2], color[o2]

    # ---- 逐像素前缀积得到透射率 T_i = Π_{j<i}(1-α_j)
    log1ma = torch.log1p(-alpha)
    E = torch.cumsum(log1ma.double(), 0).float() - log1ma           # 全局 exclusive 前缀和
    _, inv, counts = torch.unique_consecutive(pix, return_inverse=True, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    T = torch.exp(E - E[starts][inv])                               # 段内 exclusive 前缀积
    weights = T * alpha                                             # (P,)

    img = torch.zeros(height * width, 3, dtype=means.dtype, device=device)
    img.index_add_(0, pix, weights[:, None] * color)

    Eincl_last = (E + log1ma)[starts + counts - 1]
    # 未被任何高斯覆盖的像素透射率必须为 1，否则那些区域会渲染成纯黑
    Tbg = torch.ones(height * width, dtype=means.dtype, device=device)
    Tbg.index_copy_(0, pix[starts], torch.exp(Eincl_last).to(means.dtype))
    img = img + Tbg[:, None] * bg[None, :]

    acc = torch.zeros(height * width, dtype=means.dtype, device=device)
    acc.index_add_(0, pix, weights)
    return (img.reshape(height, width, 3).clamp(0, 1),
            acc.reshape(height, width).clamp(0, 1), info)


# --------------------------------------------------------------------- 简易辅助
def camera_viewmat(R, t) -> torch.Tensor:
    """把 (3,3) 旋转与 (3,) 平移拼成 4x4 世界->相机矩阵。"""
    R = torch.as_tensor(np.asarray(R, np.float32))
    t = torch.as_tensor(np.asarray(t, np.float32))
    M = torch.eye(4, dtype=torch.float32, device=R.device)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def camera_K(fx, fy, cx, cy, dtype=torch.float32, device="cpu") -> torch.Tensor:
    return torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=dtype, device=device)


def fov2focal(fov_rad: float, pixels: int) -> float:
    return pixels / (2 * math.tan(fov_rad / 2))
