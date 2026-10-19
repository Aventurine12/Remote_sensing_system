# -*- coding: utf-8 -*-
"""纯 numpy 前向渲染器：给前端做交互式预览用，不依赖 torch，也不做反向传播。

算法与 rasterizer.render_gaussians 完全一致（同一条前向 alpha 合成公式），
只是把可微分实现换成更省内存的 numpy 版本，用来在拖动鼠标时实时出图。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .camera import Camera
from .sh import eval_sh_np


class SplatRenderer:
    """把一组 3D 高斯渲染成图像。"""

    def __init__(self, means: np.ndarray, scales: np.ndarray, quats: np.ndarray,
                 opacities: np.ndarray, shs: np.ndarray, sh_degree: int = 1,
                 bg_color: Tuple[float, float, float] = (0.12, 0.13, 0.16)):
        self.means = np.asarray(means, np.float32).reshape(-1, 3)
        self.scales = np.asarray(scales, np.float32).reshape(-1, 3)
        self.quats = np.asarray(quats, np.float32).reshape(-1, 4)
        self.opacities = np.asarray(opacities, np.float32).reshape(-1)
        self.shs = np.asarray(shs, np.float32)
        self.sh_degree = int(sh_degree)
        self.bg = np.asarray(bg_color, np.float32)
        self._cov3d: Optional[np.ndarray] = None
        self.n = self.means.shape[0]

    # ------------------------------------------------------------------ 预计算
    def cov3d(self) -> np.ndarray:
        if self._cov3d is None:
            R = quat_to_rotmat_np(self.quats)
            M = R * self.scales[:, None, :]
            self._cov3d = (M @ M.transpose(0, 2, 1)).astype(np.float32)
        return self._cov3d

    # ------------------------------------------------------------------ 渲染
    def render(self, cam: Camera, radius_scale: float = 3.0,
               eps2d: float = 0.3, max_pairs: int = 12_000_000,
               jitter: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        H, W = int(cam.height), int(cam.width)
        img = np.tile(self.bg, (H, W, 1)).astype(np.float32)
        acc = np.zeros((H, W), np.float32)
        if self.n == 0:
            return (img * 255).astype(np.uint8), acc

        R, t = cam.R.astype(np.float32), cam.t.astype(np.float32)
        pc = self.means @ R.T + t
        z = pc[:, 2]
        keep = z > 1e-3
        if not np.any(keep):
            return (img * 255).astype(np.uint8), acc

        fx, fy, cx, cy = float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy)
        u = fx * pc[:, 0] / np.maximum(z, 1e-6) + cx
        v = fy * pc[:, 1] / np.maximum(z, 1e-6) + cy
        margin = 256.0
        keep &= (u > -margin) & (u < W + margin) & (v > -margin) & (v < H + margin)
        if not np.any(keep):
            return (img * 255).astype(np.uint8), acc

        idx = np.nonzero(keep)[0]
        pc, z, u, v = pc[idx], z[idx], u[idx], v[idx]
        cov3d = self.cov3d()[idx]

        # 世界->相机旋转作用于协方差 + 雅可比投影
        J = np.zeros((pc.shape[0], 2, 3), np.float32)
        J[:, 0, 0] = fx / z
        J[:, 0, 2] = -fx * pc[:, 0] / np.maximum(z, 1e-6) ** 2
        J[:, 1, 1] = fy / z
        J[:, 1, 2] = -fy * pc[:, 1] / np.maximum(z, 1e-6) ** 2
        T = J @ R[None, :, :]
        cov2d = T @ cov3d @ T.transpose(0, 2, 1)
        a = cov2d[:, 0, 0] + eps2d
        b = cov2d[:, 0, 1]
        c = cov2d[:, 1, 1] + eps2d
        det = np.maximum(a * c - b * b, 1e-9)
        ia, ib, ic = c / det, -b / det, a / det
        mid = 0.5 * (a + c)
        lam = mid + np.sqrt(np.maximum(mid * mid - det, 0.0))
        radius = radius_scale * np.sqrt(np.maximum(lam, 1e-9))

        # ---- 包围盒展开
        r = np.maximum(np.ceil(radius).astype(np.int64), 0)
        counts = (2 * r + 1) ** 2
        # 控制候选对规模：过大时按比例缩小半径
        total = int(counts.sum())
        if total > max_pairs:
            k = np.sqrt(max_pairs / max(total, 1))
            r = np.maximum((r * k).astype(np.int64), 0)
            counts = (2 * r + 1) ** 2
        starts = np.cumsum(counts) - counts
        total = int(counts.sum())
        if total == 0:
            return (img * 255).astype(np.uint8), acc
        gid = np.repeat(np.arange(pc.shape[0]), counts)
        local = np.arange(total) - starts[gid]
        w = 2 * r[gid] + 1
        dy = local // w - r[gid]
        dx = local % w - r[gid]
        gx = np.round(u[gid]).astype(np.int64) + dx
        gy = np.round(v[gid]).astype(np.int64) + dy
        valid = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
        gid, gx, gy = gid[valid], gx[valid], gy[valid]
        if gid.size == 0:
            return (img * 255).astype(np.uint8), acc

        if jitter > 0:
            gx = np.clip(gx + np.random.randint(-jitter, jitter + 1, gx.shape), 0, W - 1)
            gy = np.clip(gy + np.random.randint(-jitter, jitter + 1, gy.shape), 0, H - 1)

        ddx = gx.astype(np.float32) + 0.5 - u[gid]
        ddy = gy.astype(np.float32) + 0.5 - v[gid]
        power = -0.5 * (ia[gid] * ddx * ddx + 2 * ib[gid] * ddx * ddy + ic[gid] * ddy * ddy)
        alpha = np.clip(self.opacities[idx][gid] * np.exp(power), 0.0, 0.99)

        # ---- 颜色
        campos = -R.T @ t
        dirs = self.means[idx][gid] - campos[None, :]
        dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-8)
        color = np.clip(eval_sh_np(self.sh_degree, self.shs[idx][gid], dirs) + 0.5, 0, None)

        # ---- 排序 + 逐像素 alpha 合成
        pix = gy * W + gx
        o1 = np.argsort(z[gid], kind="stable")
        pix, alpha, color = pix[o1], alpha[o1], color[o1]
        o2 = np.argsort(pix, kind="stable")
        pix, alpha, color = pix[o2], alpha[o2], color[o2]

        log1ma = np.log1p(-alpha)
        E = np.cumsum(log1ma, dtype=np.float64) - log1ma        # 全局 exclusive 前缀和
        change = np.empty(pix.shape, bool)
        change[0] = True
        np.not_equal(pix[1:], pix[:-1], out=change[1:])
        seg_id = np.cumsum(change) - 1
        seg_start_E = E[change]
        T = np.exp(E - seg_start_E[seg_id])
        wgt = (T * alpha).astype(np.float32)

        flat = np.zeros((H * W, 3), np.float32)
        np.add.at(flat, pix, wgt[:, None] * color)
        accflat = np.zeros(H * W, np.float32)
        np.add.at(accflat, pix, wgt)

        # 背景透射率（每段最后一个元素）；未被覆盖的像素保持 1
        last_idx = np.nonzero(np.append(pix[1:] != pix[:-1], True))[0]
        Tbg = np.ones(H * W, np.float32)
        Tbg[pix[last_idx]] = np.exp((E + log1ma)[last_idx]).astype(np.float32)
        flat += Tbg[:, None] * self.bg[None, :]

        img = flat.reshape(H, W, 3)
        acc = accflat.reshape(H, W)
        return (np.clip(img, 0, 1) * 255).astype(np.uint8), acc

    # ------------------------------------------------------------------ 统计
    def bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.means.min(0), self.means.max(0)

    def center_scale(self) -> Tuple[np.ndarray, float]:
        lo, hi = self.bbox()
        c = (lo + hi) / 2
        return c, float(np.linalg.norm(hi - lo))


def quat_to_rotmat_np(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = q.shape[0]
    R = np.empty((N, 3, 3), np.float32)
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
