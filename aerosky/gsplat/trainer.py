# -*- coding: utf-8 -*-
"""3D 高斯泼溅训练器：SfM 位姿 + 稀疏点云 -> 优化高斯参数 -> 导出 .ply。

损失 = (1-λ)·L1 + λ·(1−SSIM)，与 3DGS 原论文一致；训练中周期性执行自适应密度控制
（欠采样区域克隆/分裂、过透明高斯剪枝）以及不透明度重置，动态平衡重建质量与模型规模。
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..utils.logger import get_logger
from .camera import Camera
from .colmap_io import ColmapCamera, ColmapImage, read_reconstruction
from .model import GaussianModel
from .rasterizer import camera_K, camera_viewmat

LOG = get_logger("gsplat.trainer")


# --------------------------------------------------------------------------- SSIM
def _gaussian_kernel(size: int = 11, sigma: float = 1.5, device="cpu"):
    import torch
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def ssim_torch(img1, img2, size: int = 11) -> "object":
    """可微 SSIM（对 (H,W,3) 的张量逐通道计算后取均值）。"""
    import torch
    import torch.nn.functional as F
    k = _gaussian_kernel(size, 1.5, img1.device)[None, None]
    x = img1.permute(2, 0, 1).unsqueeze(0)
    y = img2.permute(2, 0, 1).unsqueeze(0)
    pad = size // 2
    mu_x = F.conv2d(x, k.expand(3, 1, size, size), padding=pad, groups=3)
    mu_y = F.conv2d(y, k.expand(3, 1, size, size), padding=pad, groups=3)
    sx = F.conv2d(x * x, k.expand(3, 1, size, size), padding=pad, groups=3) - mu_x ** 2
    sy = F.conv2d(y * y, k.expand(3, 1, size, size), padding=pad, groups=3) - mu_y ** 2
    sxy = F.conv2d(x * y, k.expand(3, 1, size, size), padding=pad, groups=3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / \
        ((mu_x ** 2 + mu_y ** 2 + c1) * (sx + sy + c2))
    return s.mean()


# --------------------------------------------------------------------------- 数据集
class ViewDataset:
    """一帧影像 + 一个相机。"""

    def __init__(self, name: str, image: np.ndarray, cam: Camera):
        self.name = name
        self.image = image
        self.cam = cam

    @property
    def size(self):
        return self.cam.width, self.cam.height


def load_dataset(images_dir: str | Path, sparse_dir: str | Path,
                 downscale: int = 1, max_views: Optional[int] = None) -> Tuple[List[ViewDataset], np.ndarray, np.ndarray]:
    """读取 COLMAP 稀疏模型与对应影像。"""
    import cv2
    rec = read_reconstruction(sparse_dir)
    views: List[ViewDataset] = []
    for img, cam in rec.named():
        p = Path(images_dir) / img.name
        if not p.exists():
            cand = list(Path(images_dir).rglob(Path(img.name).name))
            if not cand:
                LOG.warning("缺少影像 %s，跳过", img.name)
                continue
            p = cand[0]
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = bgr[:, :, ::-1].copy()
        h, w = rgb.shape[:2]
        fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
        if downscale > 1:
            rgb = cv2.resize(rgb, (w // downscale, h // downscale), interpolation=cv2.INTER_AREA)
            fx, fy, cx, cy = fx / downscale, fy / downscale, cx / downscale, cy / downscale
            h, w = rgb.shape[:2]
        c = Camera(img.R(), img.tvec, w, h, fx, fy, cx, cy, image_path=str(p))
        views.append(ViewDataset(img.name, rgb, c))
    if max_views:
        step = max(1, len(views) // max_views)
        views = views[::step][:max_views]
    LOG.info("载入 %d 个视角，稀疏点 %d 个", len(views), rec.points3D.shape[0])
    return views, rec.points3D, rec.colors


# --------------------------------------------------------------------------- 训练器
class GaussianTrainer:
    def __init__(self, cfg: dict, device: str = "cpu"):
        self.cfg = dict(cfg or {})
        self.device = device
        self.model: Optional[GaussianModel] = None
        self.views: List[ViewDataset] = []
        self.log: List[dict] = []

    # ------------------------------------------------------------------ 数据
    def prepare(self, images_dir, sparse_dir, downscale: Optional[int] = None,
                max_views: Optional[int] = None, points: Optional[np.ndarray] = None,
                colors: Optional[np.ndarray] = None) -> "GaussianTrainer":
        ds = int(downscale or self.cfg.get("downscale", 1))
        if points is None:
            self.views, points, colors = load_dataset(images_dir, sparse_dir,
                                                      downscale=ds, max_views=max_views)
        else:
            self.views, _, _ = load_dataset(images_dir, sparse_dir,
                                            downscale=ds, max_views=max_views)
        self.model = GaussianModel(sh_degree=int(self.cfg.get("sh_degree", 1)),
                                   device=self.device)
        self.model.max_gaussians = int(self.cfg.get("max_gaussians", 100000))
        if colors is not None and np.asarray(colors).max() > 1.5:
            colors = np.asarray(colors) / 255.0
        self.model.init_from_points(points, colors)
        self.model.to_device(self.device)
        self.model.make_optimizer(self.cfg.get("learning_rate", {}))
        LOG.info("高斯初始化：%d 个，设备 %s", self.model.num, self.device)
        return self

    # ------------------------------------------------------------------ 训练
    def train(self, iterations: Optional[int] = None,
              progress: Optional[Callable[[int, int, dict], None]] = None,
              preview_every: int = 0) -> Dict:
        assert self.model is not None and self.views, "请先 prepare()"
        iters = int(iterations or self.cfg.get("iterations", 700))
        densify_from = int(self.cfg.get("densify_from", 200))
        densify_until = int(self.cfg.get("densify_until", 500))
        grad_thr = float(self.cfg.get("densify_grad_threshold", 2e-4))
        reset_every = int(self.cfg.get("opacity_reset_interval", 300))
        extent = float(np.linalg.norm(self.model.bounding_box()[1] -
                                      self.model.bounding_box()[0]) + 1e-6)
        rng = np.random.default_rng(0)
        t0 = time.time()
        psnrs: List[float] = []
        bg = tuple(float(v) for v in self._estimate_bg())
        self._bg = bg
        import torch as _torch
        try:                                    # CPU 训练时把线程都用上
            _torch.set_num_threads(max(1, (os.cpu_count() or 4)))
            _torch.set_flush_denormal(True)     # 高斯尾部会出现大量非规格化浮点数，很拖 CPU
        except Exception:  # noqa: BLE001
            pass

        for it in range(1, iters + 1):
            view = self.views[int(rng.integers(0, len(self.views)))]
            gt = _to_tensor(view.image, self.device)
            vm = camera_viewmat(view.cam.R, view.cam.t).to(self.device)
            K = camera_K(view.cam.fx, view.cam.fy, view.cam.cx, view.cam.cy,
                         device=self.device)
            img, alpha, info = self.model(vm, K, view.cam.width, view.cam.height, bg=bg)
            l1 = (img - gt).abs().mean()
            ssim = ssim_torch(img, gt)
            loss = 0.8 * l1 + 0.2 * (1 - ssim)
            loss.backward()
            self.model.accumulate_grad()
            self.model.optimizer.step()
            self.model.optimizer.zero_grad(set_to_none=True)

            if densify_from <= it <= densify_until and it % 100 == 0:
                nc, ns = self.model.densify_and_prune(grad_thr, extent=extent)
                st = getattr(self.model, "last_grad_stats", {})
                LOG.info("  致密化 iter%d：克隆 %d / 分裂 %d -> %d 个高斯；"
                         "梯度 q50=%.2e q90=%.2e 阈值=%.2e",
                         it, nc, ns, self.model.num, st.get("q50", 0), st.get("q90", 0),
                         st.get("auto_threshold", grad_thr))
            if reset_every and it % reset_every == 0:
                self.model.reset_opacity()

            if it % 25 == 0 or it == iters:
                with _no_grad():
                    psnr = _psnr(img.detach(), gt)
                psnrs.append(psnr)
                rec = {"iter": it, "loss": float(loss.item()), "psnr": psnr,
                       "n_gauss": self.model.num, "pairs": info["num_pairs"],
                       "elapsed": round(time.time() - t0, 2)}
                self.log.append(rec)
                if progress:
                    progress(it, iters, rec)
                else:
                    LOG.info("iter %4d/%d | loss %.4f | PSNR %.2f | 高斯 %d | 候选对 %.2fM | %.1fs",
                             it, iters, rec["loss"], psnr, self.model.num,
                             info["num_pairs"] / 1e6, rec["elapsed"])
        total = time.time() - t0
        result = {"iterations": iters, "seconds": round(total, 2),
                  "final_psnr": psnrs[-1] if psnrs else None,
                  "best_psnr": max(psnrs) if psnrs else None,
                  "num_gaussians": self.model.num, "history": self.log}
        LOG.info("训练完成：%d 次迭代 / %.1fs / 最终 PSNR %.2f / 高斯 %d",
                 iters, total, result["final_psnr"] or 0, self.model.num)
        return result

    # ------------------------------------------------------------------ 评估与导出
    def evaluate(self, max_views: int = 8) -> Dict[str, float]:
        """在若干视角上算 PSNR / SSIM，作为重建保真度指标。"""
        from ..utils.vis import psnr as psnr_np, ssim as ssim_np
        ps, ss = [], []
        step = max(1, len(self.views) // max_views)
        with _no_grad():
            for v in self.views[::step][:max_views]:
                img = self.render_view(v)
                ps.append(psnr_np(img, v.image))
                ss.append(ssim_np(img, v.image))
        return {"PSNR": float(np.mean(ps)), "SSIM": float(np.mean(ss)), "views": len(ps)}

    def render_view(self, view: ViewDataset, bg=(0.0, 0.0, 0.0)) -> np.ndarray:
        vm = camera_viewmat(view.cam.R, view.cam.t).to(self.device)
        K = camera_K(view.cam.fx, view.cam.fy, view.cam.cx, view.cam.cy, device=self.device)
        with _no_grad():
            img, _, _ = self.model(vm, K, view.cam.width, view.cam.height, bg=bg)
        return (img.detach().cpu().numpy() * 255).astype(np.uint8)

    def _estimate_bg(self) -> np.ndarray:
        """用首帧影像四周边框像素的中位数估计背景色（3DGS 训练里背景色必须固定）。"""
        if not self.views:
            return np.zeros(3, np.float32)
        img = self.views[0].image.astype(np.float32) / 255.0
        h, w = img.shape[:2]
        b = max(1, min(h, w) // 24)
        border = np.concatenate([img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
                                 img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3)])
        return np.median(border, axis=0).astype(np.float32)

    def render_orbit(self, n: int = 8, width: int = 320, height: int = 240,
                     elev_deg: float = 20.0) -> List[np.ndarray]:
        """绕场景一圈渲染，用于生成"三维重建成果"图。"""
        lo, hi = self.model.bounding_box()
        center = (lo + hi) / 2
        radius = float(np.linalg.norm(hi - lo)) * 0.9 + 1.0
        out = []
        for i in range(n):
            ang = 2 * math.pi * i / n
            e = math.radians(elev_deg)
            eye = center + np.array([radius * math.cos(ang) * math.cos(e),
                                     radius * math.sin(ang) * math.cos(e),
                                     radius * math.sin(e)])
            cam = Camera.look_at(eye, center, up=(0, 0, -1), width=width, height=height,
                                 fov_y_deg=45.0)
            vm = camera_viewmat(cam.R, cam.t).to(self.device)
            K = camera_K(cam.fx, cam.fy, cam.cx, cam.cy, device=self.device)
            with _no_grad():
                img, _, _ = self.model(vm, K, width, height, bg=getattr(self, "_bg", (0, 0, 0)))
            out.append((img.detach().cpu().numpy() * 255).astype(np.uint8))
        return out

    def save(self, out_dir: str | Path, name: str = "point_cloud") -> Dict[str, str]:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        ply = self.model.save_ply(d / f"{name}.ply")
        ckpt = self.model.save_state(d / f"{name}.pt")
        import json
        meta = {"num_gaussians": self.model.num, "sh_degree": self.model.sh_degree,
                "history": self.log}
        with open(d / "train_log.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return {"ply": ply, "checkpoint": ckpt, "log": str(d / "train_log.json")}


# --------------------------------------------------------------------------- 工具
class _no_grad:
    def __enter__(self):
        import torch
        self._ctx = torch.no_grad()
        self._ctx.__enter__()
        return self

    def __exit__(self, *a):
        return self._ctx.__exit__(*a)


def _to_tensor(img: np.ndarray, device: str):
    import torch
    return torch.tensor(np.asarray(img, np.float32) / 255.0, device=device)


def _psnr(a, b) -> float:
    mse = float(((a - b) ** 2).mean().item())
    return 99.0 if mse <= 1e-12 else float(10 * math.log10(1.0 / mse))
