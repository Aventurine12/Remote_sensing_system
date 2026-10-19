# -*- coding: utf-8 -*-
"""可训练的 3D 高斯模型：参数、激活、自适应密度控制、相似变换与 .ply 存取。

一个高斯单元由四类参数定义（与项目文档中的表述一致）：
    位置 μ ∈ R³      —— 决定高斯在场景中的中心
    协方差 Σ = R S Sᵀ Rᵀ —— 由尺度 S（3）与旋转四元数 R（4）共同决定形状与朝向
    不透明度 α ∈ (0,1)   —— 决定光线穿透程度
    球谐系数 SH          —— 编码视角相关颜色；本项目降到 1 阶以精简参数
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..utils.logger import get_logger
from . import ply_io
from .rasterizer import render_gaussians
from .sh import num_coeffs, rgb_to_sh

LOG = get_logger("gsplat.model")

SH_C0 = 0.28209479177387814


class GaussianModel(nn.Module):
    def __init__(self, sh_degree: int = 1, device: str = "cpu"):
        super().__init__()
        self.sh_degree = int(sh_degree)
        self.device_ = device
        self.max_gaussians = 1_000_000
        self._means = nn.Parameter(torch.zeros(0, 3))
        self._scales = nn.Parameter(torch.zeros(0, 3))
        self._quats = nn.Parameter(torch.zeros(0, 4))
        self._opac = nn.Parameter(torch.zeros(0, 1))
        self._shs = nn.Parameter(torch.zeros(0, num_coeffs(self.sh_degree), 3))
        self._xyz_grad_accum = torch.zeros(0)
        self._xyz_grad_count = torch.zeros(0)
        self.optimizer: Optional[torch.optim.Optimizer] = None

    # ------------------------------------------------------------------ 属性
    @property
    def num(self) -> int:
        return self._means.shape[0]

    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self._scales).clamp(max=1e3)

    def get_rotation(self) -> torch.Tensor:
        return self._quats / self._quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self._opac).squeeze(-1)

    def get_features(self) -> torch.Tensor:
        return self._shs

    def get_colors(self) -> torch.Tensor:
        """0 阶球谐对应的基础颜色（0~1）。"""
        return (self._shs[:, 0, :] * SH_C0 + 0.5).clamp(0, 1)

    # ------------------------------------------------------------------ 初始化
    @torch.no_grad()
    def init_from_points(self, pts: np.ndarray, colors: Optional[np.ndarray] = None,
                         scales: Optional[np.ndarray] = None, k: int = 12):
        """由稀疏点云（如 COLMAP 的 points3D）初始化高斯。"""
        pts = np.asarray(pts, np.float64).reshape(-1, 3)
        n = pts.shape[0]
        if n == 0:
            return self
        if n > self.max_gaussians:
            sel = np.random.default_rng(0).choice(n, self.max_gaussians, replace=False)
            pts = pts[sel]
            colors = None if colors is None else np.asarray(colors)[sel]
            n = pts.shape[0]
        if scales is None:
            dist = _nn_distance(pts, k=min(k, n - 1 if n > 1 else 1))
            # 初始尺度取近邻距离的一半：过大的初始高斯会覆盖太多像素，既拖慢训练
            # 又让自适应密度控制失去意义
            scale = np.repeat(np.maximum(np.asarray(dist, np.float64) * 0.5, 1e-4)[:, None],
                              3, axis=1)
        else:
            scale = np.asarray(scales, np.float64).reshape(n, 3)
        if colors is None:
            colors = np.full((n, 3), 0.55)
        colors = np.asarray(colors, np.float64).reshape(n, 3)[:, :3]
        colors = np.clip(colors / 255.0 if colors.max() > 1.5 else colors, 0.03, 0.97)

        dev = self.device_
        self._means = nn.Parameter(torch.tensor(pts, dtype=torch.float32, device=dev))
        self._scales = nn.Parameter(torch.tensor(np.log(np.clip(scale, 1e-4, None)),
                                                 dtype=torch.float32, device=dev))
        q = torch.zeros(n, 4, device=dev)
        q[:, 0] = 1.0
        self._quats = nn.Parameter(q)
        self._opac = nn.Parameter(torch.full((n, 1), _inv_sigmoid(0.25), device=dev))
        C = num_coeffs(self.sh_degree)
        shs = torch.zeros(n, C, 3, device=dev)
        sh0 = (torch.tensor(colors, dtype=torch.float32, device=dev) - 0.5) / SH_C0
        shs[:, 0, :] = sh0
        self._shs = nn.Parameter(shs)
        self._xyz_grad_accum = torch.zeros(n, device=dev)
        self._xyz_grad_count = torch.zeros(n, device=dev)
        return self

    def to_device(self, device: str) -> "GaussianModel":
        self.device_ = device
        self.to(device)
        self._xyz_grad_accum = self._xyz_grad_accum.to(device)
        self._xyz_grad_count = self._xyz_grad_count.to(device)
        return self

    # ------------------------------------------------------------------ 前向
    def forward(self, viewmat: torch.Tensor, K: torch.Tensor, width: int, height: int,
                bg: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        return render_gaussians(self._means, self.get_scaling(), self.get_rotation(),
                                self.get_opacity(), self._shs, viewmat, K,
                                width, height, sh_degree=self.sh_degree, bg_color=bg)

    # ------------------------------------------------------------------ 优化器
    def make_optimizer(self, lr: Dict[str, float]):
        self._lr = dict(lr or {})
        self.optimizer = torch.optim.Adam([
            {"params": [self._means], "lr": self._lr.get("position", 1.6e-4), "name": "xyz"},
            {"params": [self._shs], "lr": self._lr.get("feature", 2.5e-3), "name": "f_dc"},
            {"params": [self._opac], "lr": self._lr.get("opacity", 5e-2), "name": "opacity"},
            {"params": [self._scales], "lr": self._lr.get("scaling", 5e-3), "name": "scaling"},
            {"params": [self._quats], "lr": self._lr.get("rotation", 1e-3), "name": "rotation"},
        ], eps=1e-15)
        return self.optimizer

    def _after_param_change(self) -> None:
        """增删高斯会重建 Parameter 对象，必须同步重建优化器，否则优化器仍指向旧张量、
        训练会静默失效甚至发散（这是自研 3DGS 最容易踩的坑）。"""
        if self.optimizer is not None:
            self.make_optimizer(getattr(self, "_lr", {}))

    # ------------------------------------------------------------ 自适应密度控制
    def accumulate_grad(self):
        if self._means.grad is None:
            return
        n = self.num
        if self._xyz_grad_accum.numel() != n:
            self._xyz_grad_accum = torch.zeros(n, device=self.device_)
            self._xyz_grad_count = torch.zeros(n, device=self.device_)
        self._xyz_grad_accum += self._means.grad.detach().norm(dim=-1)
        self._xyz_grad_count += 1

    @torch.no_grad()
    def densify_and_prune(self, grad_threshold: float = 2e-4, min_opacity: float = 0.005,
                          extent: float = 1.0, max_screen: float = 0.5):
        """克隆/分裂"欠采样"高斯、剪枝"过透明/过大"高斯（平衡质量与效率）。"""
        if self.num == 0 or self._xyz_grad_count.max() == 0:
            return 0, 0
        avg = self._xyz_grad_accum / self._xyz_grad_count.clamp_min(1)
        self._xyz_grad_accum.zero_()
        self._xyz_grad_count.zero_()
        self.last_grad_stats = {
            "max": float(avg.max().item()),
            "q50": float(avg.median().item()),
            "q90": float(torch.quantile(avg.float(), 0.90).item()),
        }
        if grad_threshold <= 0:
            # 自动阈值：梯度的世界坐标量纲随场景尺度变化，写死一个常数并不通用。
            # 这里取梯度范数的 85 分位，稳定地让"最欠采样"的那部分高斯参与致密化。
            grad_threshold = float(torch.quantile(avg.float(), 0.85).item())
            self.last_grad_stats["auto_threshold"] = grad_threshold
        hot = avg >= grad_threshold
        if hot.sum() == 0:
            return 0, 0
        scales = self.get_scaling()
        big = scales.max(dim=-1).values > 0.01 * extent
        n_clone = int((hot & ~big).sum().item())
        n_split = int((hot & big).sum().item())

        new_means, new_scales, new_quats, new_opac, new_shs = [], [], [], [], []
        if n_clone:
            new_means.append(self._means[hot & ~big].detach())
            new_scales.append(self._scales[hot & ~big].detach())
            new_quats.append(self._quats[hot & ~big].detach())
            new_opac.append(self._opac[hot & ~big].detach())
            new_shs.append(self._shs[hot & ~big].detach())
        if n_split:
            sel = hot & big
            rep = self._means[sel].detach()
            sc = self.get_scaling()[sel].detach()
            qt = self.get_rotation()[sel].detach()
            op = self.get_opacity()[sel].detach()
            sh = self._shs[sel].detach()
            std = sc * 1.0
            noise = torch.randn_like(rep) * std
            new_means += [rep + noise, rep - noise]
            new_scales += [torch.log(sc / 1.6).clamp_min(-12),
                           torch.log(sc / 1.6).clamp_min(-12)]
            new_quats += [qt, qt]
            new_opac += [_inv_sigmoid(op).unsqueeze(-1), _inv_sigmoid(op).unsqueeze(-1)]
            new_shs += [sh, sh]

        if new_means:
            self._cat_params(new_means, new_scales, new_quats, new_opac, new_shs)

        # 剪枝
        keep = self.get_opacity() >= min_opacity
        keep &= self.get_scaling().max(dim=-1).values < 0.1 * extent
        if keep.sum() < self.num:
            self._select(keep)
        # 硬上限保护（轻量化策略：高斯总数 <= max_gaussians）
        if self.num > self.max_gaussians:
            op = self.get_opacity()
            idx = torch.topk(op, self.max_gaussians).indices
            mask = torch.zeros(self.num, dtype=torch.bool, device=self.device_)
            mask[idx] = True
            self._select(mask)
        self._after_param_change()
        return n_clone, n_split

    @torch.no_grad()
    def _cat_params(self, means, scales, quats, opac, shs):
        dev = self.device_
        self._means = nn.Parameter(torch.cat([self._means.detach()] + means, 0).to(dev))
        self._scales = nn.Parameter(torch.cat([self._scales.detach()] + scales, 0).to(dev))
        self._quats = nn.Parameter(torch.cat([self._quats.detach()] + quats, 0).to(dev))
        self._opac = nn.Parameter(torch.cat([self._opac.detach().reshape(-1, 1)] +
                                            [o.reshape(-1, 1) for o in opac], 0).to(dev))
        self._shs = nn.Parameter(torch.cat([self._shs.detach()] + shs, 0).to(dev))
        n = self.num
        self._xyz_grad_accum = torch.zeros(n, device=dev)
        self._xyz_grad_count = torch.zeros(n, device=dev)

    @torch.no_grad()
    def _select(self, mask: torch.Tensor):
        self._means = nn.Parameter(self._means.detach()[mask])
        self._scales = nn.Parameter(self._scales.detach()[mask])
        self._quats = nn.Parameter(self._quats.detach()[mask])
        self._opac = nn.Parameter(self._opac.detach()[mask])
        self._shs = nn.Parameter(self._shs.detach()[mask])
        self._xyz_grad_accum = torch.zeros(self.num, device=self.device_)
        self._xyz_grad_count = torch.zeros(self.num, device=self.device_)

    @torch.no_grad()
    def reset_opacity(self, value: float = 0.01):
        op = self.get_opacity()
        new = torch.minimum(op, torch.full_like(op, value))
        self._opac = nn.Parameter(_inv_sigmoid(new).unsqueeze(-1))
        self._after_param_change()

    # ------------------------------------------------------------------ 几何
    @torch.no_grad()
    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        m = self._means.detach().cpu().numpy()
        return m.min(0), m.max(0)

    @torch.no_grad()
    def centroid(self) -> np.ndarray:
        return self._means.detach().cpu().numpy().mean(0)

    @torch.no_grad()
    def transform(self, R: np.ndarray, s: float, t: np.ndarray) -> "GaussianModel":
        """对整场景施加相似变换 p' = s R p + t（地理配准就用它把模型摆到 WGS84）。"""
        R = np.asarray(R, np.float64)
        t = np.asarray(t, np.float64).reshape(3)
        dev = self.device_
        Rt = torch.tensor(R, dtype=torch.float32, device=dev)
        tt = torch.tensor(t, dtype=torch.float32, device=dev)
        self._means = nn.Parameter((self._means.detach() @ Rt.T) * float(s) + tt)
        dq = _rotmat_to_quat(R)
        q = torch.tensor(dq, dtype=torch.float32, device=dev)
        self._quats = nn.Parameter(_quat_mul(q[None, :].repeat(self.num, 1),
                                             self._quats.detach()))
        self._scales = nn.Parameter(self._scales.detach() + math.log(max(float(s), 1e-8)))
        self._after_param_change()
        return self

    @torch.no_grad()
    def translate(self, t) -> "GaussianModel":
        self._means = nn.Parameter(self._means.detach() +
                                   torch.tensor(t, dtype=torch.float32, device=self.device_))
        return self

    # ------------------------------------------------------------------ 存取
    def save_ply(self, path) -> str:
        return ply_io.gaussians_to_ply(
            path, self._means.detach().cpu().numpy(),
            self._scales.detach().cpu().numpy(), self._quats.detach().cpu().numpy(),
            self._opac.detach().cpu().numpy(), self._shs.detach().cpu().numpy(),
            self.sh_degree)

    def save_state(self, path) -> str:
        Path_ = __import__("pathlib").Path
        Path_(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"sh_degree": self.sh_degree,
                    "means": self._means.detach().cpu(),
                    "scales": self._scales.detach().cpu(),
                    "quats": self._quats.detach().cpu(),
                    "opac": self._opac.detach().cpu(),
                    "shs": self._shs.detach().cpu()}, path)
        return str(path)

    def load_state(self, path, device: Optional[str] = None) -> "GaussianModel":
        ck = torch.load(path, map_location="cpu")
        dev = device or self.device_
        self.sh_degree = int(ck.get("sh_degree", self.sh_degree))
        self._means = nn.Parameter(ck["means"].float().to(dev))
        self._scales = nn.Parameter(ck["scales"].float().to(dev))
        self._quats = nn.Parameter(ck["quats"].float().to(dev))
        self._opac = nn.Parameter(ck["opac"].float().to(dev))
        self._shs = nn.Parameter(ck["shs"].float().to(dev))
        self._xyz_grad_accum = torch.zeros(self.num, device=dev)
        self._xyz_grad_count = torch.zeros(self.num, device=dev)
        return self

    @classmethod
    def from_ply(cls, path, device: str = "cpu") -> "GaussianModel":
        d = ply_io.ply_to_gaussians(path)
        m = cls(sh_degree=int(round(math.sqrt(d["shs"].shape[1]))) - 1, device=device)
        m._means = nn.Parameter(torch.tensor(d["means"], dtype=torch.float32, device=device))
        m._scales = nn.Parameter(torch.tensor(np.log(np.clip(d["scales"], 1e-8, None)),
                                              dtype=torch.float32, device=device))
        m._quats = nn.Parameter(torch.tensor(d["quats"], dtype=torch.float32, device=device))
        m._opac = nn.Parameter(torch.tensor(_inv_sigmoid(d["opacities"]),
                                            dtype=torch.float32, device=device).reshape(-1, 1))
        m._shs = nn.Parameter(torch.tensor(d["shs"], dtype=torch.float32, device=device))
        m._xyz_grad_accum = torch.zeros(m.num, device=device)
        m._xyz_grad_count = torch.zeros(m.num, device=device)
        return m


# --------------------------------------------------------------------------- 工具
def _nn_distance(pts: np.ndarray, k: int = 3) -> np.ndarray:
    """每个点到最近 k 个邻居的平均距离（近似，用分块计算避免大矩阵）。"""
    n = pts.shape[0]
    if n <= 1:
        return np.full(n, 0.05)
    out = np.zeros(n)
    step = max(1, 2000)
    for i in range(0, n, step):
        chunk = pts[i:i + step]
        d = np.linalg.norm(chunk[:, None, :] - pts[None, :, :], axis=-1)
        d.sort(axis=1)
        kk = min(k, d.shape[1] - 1)
        out[i:i + step] = d[:, 1:kk + 1].mean(axis=1)
    return np.maximum(out, 1e-4)


def _inv_sigmoid(x):
    if isinstance(x, torch.Tensor):
        x = x.clamp(1e-6, 1 - 1e-6)
        return torch.log(x / (1 - x))
    x = float(min(max(x, 1e-6), 1 - 1e-6))
    return math.log(x / (1 - x))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    m = np.asarray(R, np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], np.float64)
    return q / np.linalg.norm(q)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)
