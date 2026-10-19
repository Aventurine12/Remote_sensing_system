# -*- coding: utf-8 -*-
"""三维重建门面：把「多视角影像 -> 位姿 -> 3D 高斯泼溅 -> 轻量化 .ply」封成一步。

**重建交给现成的 3DGS 实现**（gsplat / 官方实现 / OpenSplat / Taichi），
本模块只负责：准备数据、挑选后端、把产物收回来并统一成 ReconstructionResult，
再交给地理配准与前端三维视图。

没有可用 CUDA 后端时，系统会退回内置极简实现（仅用于把流程跑通和给前端预览），
并在结果里把 backend 标成 builtin，报告里也会如实写明。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .gsbackend.base import GSResult
from .gsbackend.registry import autoselect, detect_backends
from .utils.logger import get_logger

LOG = get_logger("reconstruction")


@dataclass
class ReconstructionResult:
    root: str = ""
    ply: str = ""
    checkpoint: str = ""
    num_gaussians: int = 0
    sh_degree: int = 1
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    seconds: float = 0.0
    device: str = ""
    images_dir: str = ""
    sparse_dir: str = ""
    previews: List[str] = field(default_factory=list)
    history: List[dict] = field(default_factory=list)
    source: str = "synthetic"          # synthetic | colmap
    origin_lonlat: Optional[List[float]] = None
    local_center: Optional[List[float]] = None
    # 新增：真正干活的是哪个 3DGS 实现
    backend: str = ""
    toolkit: str = ""
    log_file: str = ""
    iterations: Optional[int] = None

    def summary(self) -> dict:
        d = dict(self.__dict__)
        d.pop("history", None)
        return d

    @property
    def is_real_backend(self) -> bool:
        return self.backend not in ("", "builtin")


def reconstruct(out_dir: str | Path, cfg: dict, device: str = "cuda",
                images_dir: Optional[str] = None, sparse_dir: Optional[str] = None,
                iterations: Optional[int] = None, n_previews: int = 6,
                progress: Optional[Callable] = None,
                origin_lonlat=(126.6320, 45.7520), reuse: bool = False,
                backend: Optional[str] = None, app_cfg: Optional[dict] = None,
                downscale: int = 1) -> ReconstructionResult:
    """执行一次三维重建。

    Args:
        cfg: gsplat 相关配置（迭代数、球面谐波阶数、时间预算等）
        backend: 指定 3DGS 后端名；None 表示按可用性自动选择
        app_cfg: 全局配置（用来找各后端的 Python 解释器路径等）
    """
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if reuse:
        cached = out_dir / "reconstruction.json"
        if cached.exists():
            try:
                d = json.loads(cached.read_text(encoding="utf-8"))
                res = ReconstructionResult(**{k: v for k, v in d.items()
                                              if k in ReconstructionResult.__dataclass_fields__})
                if res.ply and Path(res.ply).exists():
                    LOG.info("复用已有三维重建结果：%s（%s，%d 个高斯）",
                             out_dir, res.backend or "?", res.num_gaussians)
                    res.seconds = 0.0
                    res.history = []
                    return res
            except Exception as e:  # noqa: BLE001
                LOG.warning("复用失败，改为重新重建：%s", e)

    # ---------------------------------------------------------------- 数据
    source = "colmap"
    local_center = None
    if not images_dir or not sparse_dir or not Path(sparse_dir).exists():
        LOG.info("未提供多视角影像，使用内置合成场景（仅用于流程演示）")
        from .gsplat.synthetic import make_dataset
        info = make_dataset(out_dir / "synthetic", n_views=24, width=320, height=240,
                            origin_lonlat=origin_lonlat)
        images_dir, sparse_dir = info["images_dir"], info["sparse_dir"]
        source = "synthetic"
        local_center = info["center"]

    # ---------------------------------------------------------------- 后端
    be = autoselect(app_cfg or {}, prefer=backend)
    ok, why = be.available()
    if not ok and be.name != "builtin":
        LOG.warning("后端 %s 不可用（%s），改用内置实现", be.name, why)
        be = autoselect(app_cfg or {}, prefer="builtin", verbose=False)
        ok, why = be.available()

    iters = iterations if iterations is not None else int(cfg.get("iterations", 7000))
    sh_degree = int(cfg.get("sh_degree", 3))
    extra = {"max_gaussians": cfg.get("max_gaussians"),
             "time_budget": cfg.get("time_budget")}
    LOG.info("三维重建：后端=%s，迭代=%d，球谐阶数=%d，设备=%s",
             be.name, iters, sh_degree, device)

    res_dir = out_dir / ("run_" + be.name)
    res_dir.mkdir(parents=True, exist_ok=True)

    def _p(line: str):
        s = line.rstrip()
        if s and progress:
            progress(s)
        elif s and ("iter" in s.lower() or "PSNR" in s):
            LOG.info("  %s", s[:160])

    try:
        gs: GSResult = be.train(str(images_dir), str(sparse_dir), str(res_dir),
                                iterations=iters, downscale=downscale,
                                sh_degree=sh_degree, device=device, extra=extra,
                                progress_cb=_p)
    except Exception as e:  # noqa: BLE001
        LOG.error("后端 %s 执行失败：%s", be.name, e)
        if be.name != "builtin":
            LOG.warning("回退到内置实现")
            be = autoselect(app_cfg or {}, prefer="builtin", verbose=False)
            gs = be.train(str(images_dir), str(sparse_dir), str(out_dir / "run_builtin"),
                          iterations=min(iters, 800), downscale=max(downscale, 2),
                          sh_degree=min(sh_degree, 1), device="cpu", extra=extra,
                          progress_cb=_p)
        else:
            raise

    res = ReconstructionResult(
        root=str(out_dir), ply=gs.ply, checkpoint="",
        num_gaussians=int(gs.num_gaussians or 0), sh_degree=sh_degree,
        psnr=gs.psnr, ssim=gs.ssim, seconds=time.time() - t0, device=gs.device or device,
        images_dir=str(images_dir), sparse_dir=str(sparse_dir),
        previews=list(gs.previews or []), history=list(gs.extra.get("history", []) or []),
        source=source, origin_lonlat=list(origin_lonlat), local_center=local_center,
        backend=be.name, toolkit=be.title, log_file=gs.log_file,
        iterations=iters,
    )
    if not res.previews:
        res.previews = [str(p) for p in sorted(res_dir.glob("orbit_*.png"))]
    with open(out_dir / "reconstruction.json", "w", encoding="utf-8") as f:
        json.dump(res.summary(), f, ensure_ascii=False, indent=2)
    LOG.info("三维重建完成：后端 %s，%d 个高斯，PSNR %s，耗时 %.1fs -> %s",
             res.backend, res.num_gaussians,
             f"{res.psnr:.2f}" if res.psnr is not None else "-",
             res.seconds, res.ply)
    return res


def backend_report(app_cfg: Optional[dict] = None) -> List[dict]:
    """列出所有 3DGS 后端的可用性（给界面/命令行用）。"""
    return detect_backends(app_cfg or {}, verbose=False)


def load_renderer(ply_path: str):
    """载入 .ply 得到前端可交互浏览用的实时渲染器（纯 numpy，仅用于预览显示）。"""
    from .gsplat.render_np import SplatRenderer
    from .gsplat.ply_io import ply_to_gaussians
    d = ply_to_gaussians(ply_path)
    deg = int(round(np.sqrt(d["shs"].shape[1]))) - 1
    return SplatRenderer(d["means"], d["scales"], d["quats"], d["opacities"], d["shs"], deg)


def ortho_footprint(means: np.ndarray, resolution: int = 256, pad: float = 0.06):
    """把高斯的水平投影做成占据图（供地理配准与二维叠加用）。"""
    pts = np.asarray(means, np.float64)
    if pts.size == 0:
        return np.zeros((resolution, resolution), np.uint8), (0, 1, 0, 1)
    xmin, ymin = pts[:, 0].min(), pts[:, 1].min()
    xmax, ymax = pts[:, 0].max(), pts[:, 1].max()
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    m = max(dx, dy)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = m / 2 * (1 + pad)
    img = np.zeros((resolution, resolution), np.uint8)
    gx = np.clip(((pts[:, 0] - (cx - half)) / (2 * half) * (resolution - 1)).astype(int),
                 0, resolution - 1)
    gy = np.clip(((pts[:, 1] - (cy - half)) / (2 * half) * (resolution - 1)).astype(int),
                 0, resolution - 1)
    img[resolution - 1 - gy, gx] = 255
    import cv2
    img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
    return img, (cx - half, cx + half, cy - half, cy + half)


# --------------------------------------------------------------------------- 内置兜底
def _train_builtin(images_dir, sparse_dir, out_dir, iterations=700, downscale=2,
                   sh_degree=1, device="cpu", progress_cb=None, cfg=None) -> GSResult:
    """内置极简实现（纯 PyTorch）。仅在没有 CUDA 后端时兜底，不是 3DGS 主路径。"""
    from .gsplat.trainer import GaussianTrainer
    from .gsplat.ply_io import ply_to_gaussians
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    c = dict(cfg or {})
    c.update({"sh_degree": sh_degree, "iterations": iterations, "downscale": downscale,
              "checkpoint_every": max(1, iterations // 4)})
    tr = GaussianTrainer(c, device=device)
    tr.checkpoint_dir = str(out_dir / "checkpoints")
    tr.prepare(images_dir, sparse_dir, downscale=downscale)
    state = {"last": 0}

    def prog(i, n, rec):
        step = max(1, n // 10)
        if i - state["last"] >= step or i == n:
            state["last"] = i
            line = (f"  iter {i:5d}/{n}  loss={rec['loss']:.4f}  "
                    f"PSNR={rec.get('psnr_avg', rec['psnr']):.2f}  高斯={rec['n_gauss']}")
            print(line, flush=True)
            if progress_cb:
                progress_cb(line)

    tr.train(iterations=iterations, progress=prog)
    q = tr.evaluate(max_views=8)
    saved = tr.save(out_dir)
    previews = []
    try:
        import cv2
        w, h = 640, 480
        for i, img in enumerate(tr.render_orbit(n=4, width=w, height=h)):
            p = out_dir / f"orbit_{i:02d}.png"
            cv2.imwrite(str(p), img[:, :, ::-1])
            previews.append(str(p))
    except Exception:  # noqa: BLE001
        pass
    d = ply_to_gaussians(saved["ply"])
    return GSResult(backend="builtin", out_dir=str(out_dir), ply=saved["ply"],
                    num_gaussians=int(d["means"].shape[0]), psnr=float(q["PSNR"]),
                    ssim=float(q["SSIM"]), device=device, previews=previews,
                    iterations=iterations, log_file="")
