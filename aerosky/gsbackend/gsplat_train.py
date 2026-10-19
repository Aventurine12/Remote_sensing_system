# -*- coding: utf-8 -*-
"""gsplat 训练入口（独立脚本，跑在装有 gsplat + CUDA 的解释器里）。

这个脚本刻意不依赖 aerosky 包，只用 numpy / torch / gsplat，
这样它可以放在任意 Python 环境里执行；aerosky 通过子进程调用它。

    python gsplat_train.py --images <照片目录> --sparse <COLMAP sparse/0> --out <输出目录>

产物：point_cloud.ply（标准 3DGS 格式）、metrics.json、orbit_*.png、train.log
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch


# ============================================================ COLMAP 读取（独立实现）
def read_colmap(sparse_dir: Path):
    """返回 cameras[dict], images[list], points(N,3), colors(N,3)。支持 bin / txt。"""
    cameras, images, pts, cols = {}, [], np.zeros((0, 3)), np.zeros((0, 3))
    if (sparse_dir / "cameras.bin").exists():
        cameras = _read_cameras_bin(sparse_dir / "cameras.bin")
        images = _read_images_bin(sparse_dir / "images.bin")
        p = sparse_dir / "points3D.bin"
        if p.exists():
            pts, cols = _read_points_bin(p)
    else:
        cameras = _read_cameras_txt(sparse_dir / "cameras.txt")
        images = _read_images_txt(sparse_dir / "images.txt")
        p = sparse_dir / "points3D.txt"
        if p.exists():
            pts, cols = _read_points_txt(p)
    return cameras, images, pts, cols


def _q2r(q):
    q = np.asarray(q, np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


CAMERA_MODELS = {0: ("SIMPLE_PINHOLE", ["f", "cx", "cy"]),
                 1: ("PINHOLE", ["fx", "fy", "cx", "cy"]),
                 2: ("SIMPLE_RADIAL", ["f", "cx", "cy", "k"]),
                 3: ("RADIAL", ["f", "cx", "cy", "k1", "k2"]),
                 4: ("OPENCV", ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"])}


def _cam_params(model, params):
    p = list(params)
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        return p[0], p[0], p[1], p[2]
    return p[0], p[1], p[2], p[3]


def _read_cameras_bin(path):
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cid, mid = struct.unpack("<ii", f.read(8))
            w, h = struct.unpack("<QQ", f.read(16))
            model, names = CAMERA_MODELS.get(mid, ("PINHOLE", ["fx", "fy", "cx", "cy"]))
            params = struct.unpack("<" + "d" * len(names), f.read(8 * len(names)))
            out[cid] = (model, int(w), int(h), params)
    return out


def _read_cameras_txt(path):
    """COLMAP 文本格式：CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]"""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        cid = int(f[0])
        model = f[1]                      # 字符串模型名，如 SIMPLE_RADIAL / PINHOLE
        w, h = int(f[2]), int(f[3])
        out[cid] = (model, w, h, tuple(float(x) for x in f[4:]))
    return out


def _read_images_bin(path):
    out = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            iid = struct.unpack("<i", f.read(4))[0]
            q = struct.unpack("<4d", f.read(32))
            t = struct.unpack("<3d", f.read(24))
            cid = struct.unpack("<i", f.read(4))[0]
            name = bytearray()
            while True:
                c = f.read(1)
                if c == b"\x00" or not c:
                    break
                name += c
            npts = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * npts)
            out.append((iid, q, t, cid, name.decode("utf-8", "replace")))
    return out


def _read_images_txt(path):
    out, lines, i = [], Path(path).read_text(encoding="utf-8").splitlines(), 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        f = line.split()
        out.append((int(f[0]), tuple(float(x) for x in f[1:5]),
                    tuple(float(x) for x in f[5:8]), int(f[8]), " ".join(f[9:])))
        i += 2
    return out


def _read_points_bin(path):
    pts, cols = [], []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(8)
            pts.append(struct.unpack("<3d", f.read(24)))
            cols.append(struct.unpack("<3B", f.read(3)))
            f.read(8)
            tl = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * tl)
    return np.array(pts, np.float64), np.array(cols, np.float64)


def _read_points_txt(path):
    pts, cols = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        pts.append([float(f[1]), float(f[2]), float(f[3])])
        cols.append([float(f[4]), float(f[5]), float(f[6])])
    return np.array(pts, np.float64).reshape(-1, 3), np.array(cols, np.float64).reshape(-1, 3)


# ============================================================ 数据集
class Scene:
    def __init__(self, images_dir: Path, sparse_dir: Path, downscale: int = 1,
                 max_views: int = None, white_bg: bool = False):
        import cv2
        cams, imgs, pts, cols = read_colmap(sparse_dir)
        self.views = []          # (name, image(HW3 uint8), R, t, fx, fy, cx, cy, w, h)
        for iid, q, t, cid, name in imgs:
            if cid not in cams:
                continue
            p = images_dir / name
            if not p.exists():
                cand = list(images_dir.rglob(Path(name).name))
                if not cand:
                    continue
                p = cand[0]
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = bgr[:, :, ::-1].copy()
            h, w = rgb.shape[:2]
            model, cw, ch, params = cams[cid]
            fx, fy, cx, cy = _cam_params(model, params)
            if downscale > 1:
                rgb = cv2.resize(rgb, (max(1, w // downscale), max(1, h // downscale)),
                                 interpolation=cv2.INTER_AREA)
                fx, fy, cx, cy = fx / downscale, fy / downscale, cx / downscale, cy / downscale
                h, w = rgb.shape[:2]
            self.views.append((name, rgb, _q2r(q), np.asarray(t, np.float64),
                               fx, fy, cx, cy, w, h))
        self.views.sort(key=lambda v: v[0])
        if max_views and len(self.views) > max_views:
            step = max(1, len(self.views) // max_views)
            self.views = self.views[::step][:max_views]
        if not self.views:
            raise RuntimeError("没有可用的影像，检查 images 目录与 sparse 模型是否匹配")
        self.points, self.colors = pts, cols
        self.white_bg = white_bg
        self._normalize()
        self._init_points()

    def _normalize(self, scale=1.0):
        c = np.stack([-v[2].T @ v[3] for v in self.views])
        mid = c.mean(0)
        span = float(np.linalg.norm(c.max(0) - c.min(0))) or 1.0
        k = scale / span
        new = []
        for (name, img, R, t, fx, fy, cx, cy, w, h) in self.views:
            ctr = -R.T @ t
            t2 = -R @ ((ctr - mid) * k)
            new.append((name, img, R, t2, fx, fy, cx, cy, w, h))
        self.views = new
        if self.points.shape[0]:
            self.points = (self.points - mid) * k

    def _init_points(self, grid: int = 40, bg_tol: float = 26.0):
        """没有稀疏点云时，用影像反投影造一层初始点。"""
        if self.points.shape[0] >= 50:
            return
        import cv2
        pts, cols = [], []
        pick = self.views[:: max(1, len(self.views) // 10)]
        centers = np.stack([-v[2].T @ v[3] for v in pick])
        sc = centers.mean(0)
        for v in pick:
            name, img, R, t, fx, fy, cx, cy, w, h = v
            depth = float(np.linalg.norm((-R.T @ t) - sc))
            b = max(1, min(h, w) // 32)
            border = np.concatenate([img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
                                     img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3)])
            bg = np.median(border, axis=0)
            ys, xs = np.linspace(0, h - 1, grid).astype(int), np.linspace(0, w - 1, grid).astype(int)
            gx, gy = np.meshgrid(xs, ys)
            gx, gy = gx.ravel(), gy.ravel()
            px = img[gy, gx].astype(np.float64)
            keep = np.linalg.norm(px - bg[None, :], axis=1) > bg_tol
            if keep.sum() < 8:
                keep = np.ones_like(keep, bool)
            gx, gy, px = gx[keep], gy[keep], px[keep]
            d = np.stack([(gx - cx) / fx, (gy - cy) / fy, np.ones_like(gx, np.float64)], 1)
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            pts.append((-R.T @ t) + (d @ R) * depth)
            cols.append(px)
        self.points = np.concatenate(pts, 0)
        self.colors = np.concatenate(cols, 0)


def nn_scale(points: np.ndarray, k: int = 8) -> np.ndarray:
    n = points.shape[0]
    out = np.zeros(n)
    step = 2000
    for i in range(0, n, step):
        chunk = points[i:i + step]
        d = np.linalg.norm(chunk[:, None, :] - points[None, :, :], axis=-1)
        d.sort(axis=1)
        kk = min(k, d.shape[1] - 1)
        out[i:i + step] = d[:, 1:kk + 1].mean(axis=1)
    return np.maximum(out, 1e-4)


# ============================================================ gsplat 适配
def rasterize(means, quats, scales, opacities, sh, viewmats, Ks, w, h, sh_degree, near=0.01):
    """兼容 gsplat 各版本的调用方式，返回 (rgb, alpha, info)。"""
    from gsplat import rasterization
    colors = sh
    attempts = [
        dict(),                                        # rasterization(means, quats, scales, opac, colors, ...)
    ]
    try:
        out = rasterization(means, quats, scales, opacities, colors, viewmats, Ks, w, h,
                            sh_degree=sh_degree, near_plane=near, render_mode="RGB")
        return out[0], out[1], (out[2] if len(out) > 2 else {})
    except TypeError as e1:
        # 1.5+ 的 splats 位置参数形式
        try:
            out = rasterization(dict(means=means, quats=quats, scales=scales,
                                     opacities=opacities, sh0=colors[..., :1, :],
                                     shN=colors[..., 1:, :]),
                                viewmats, Ks, w, h, sh_degree=sh_degree,
                                near_plane=near, render_mode="RGB")
            return out[0], out[1], (out[2] if len(out) > 2 else {})
        except Exception:  # noqa: BLE001
            raise e1


def make_strategy():
    """优先用 gsplat 官方的 DefaultStrategy 做自适应密度控制。"""
    for mod, attr in (("gsplat.strategy", "DefaultStrategy"), ("gsplat", "DefaultStrategy")):
        try:
            m = __import__(mod, fromlist=[attr])
            return getattr(m, attr)
        except Exception:  # noqa: BLE001
            continue
    return None


# ============================================================ 训练
def train(args) -> dict:
    import cv2
    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        print("[warn] 没有可用 CUDA，退回 CPU（会非常慢）")
        dev = "cpu"
    torch.manual_seed(0)
    np.random.seed(0)

    scene = Scene(Path(args.images), Path(args.sparse), downscale=args.downscale,
                  max_views=args.max_views)
    print(f"[data] {len(scene.views)} 个视角，{scene.points.shape[0]} 个初始点，"
          f"影像 {scene.views[0][8]}x{scene.views[0][9]}")

    gts = [torch.tensor(v[1], dtype=torch.float32, device=dev) / 255.0 for v in scene.views]
    viewmats, Ks, sizes = [], [], []
    for (name, img, R, t, fx, fy, cx, cy, w, h) in scene.views:
        vm = torch.eye(4, dtype=torch.float32)
        vm[:3, :3] = torch.tensor(R, dtype=torch.float32)
        vm[:3, 3] = torch.tensor(t, dtype=torch.float32)
        viewmats.append(vm)
        Ks.append(torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32))
        sizes.append((int(w), int(h)))
    viewmats = torch.stack(viewmats).to(dev)
    Ks = torch.stack(Ks).to(dev)

    # 背景色：取首帧边框像素中位数
    img0 = scene.views[0][1]
    b = max(1, min(img0.shape[0], img0.shape[1]) // 24)
    border = np.concatenate([img0[:b].reshape(-1, 3), img0[-b:].reshape(-1, 3),
                             img0[:, :b].reshape(-1, 3), img0[:, -b:].reshape(-1, 3)])
    bg = torch.tensor(np.median(border, axis=0) / 255.0, dtype=torch.float32, device=dev)
    print(f"[data] 背景色 RGB={bg.cpu().numpy().round(3).tolist()}")

    # ---- 初始化高斯
    pts = torch.tensor(scene.points, dtype=torch.float32, device=dev)
    if scene.colors.size and scene.colors.max() > 1.5:
        rgb = torch.tensor(scene.colors / 255.0, dtype=torch.float32, device=dev)
    else:
        rgb = torch.tensor(np.tile([0.5, 0.5, 0.5], (pts.shape[0], 1)),
                           dtype=torch.float32, device=dev)
    K = (args.sh + 1) ** 2
    sh0 = ((rgb - 0.5) / 0.28209479177387814).unsqueeze(1)
    shN = torch.zeros((pts.shape[0], K - 1, 3), dtype=torch.float32, device=dev)

    params = {
        "means": torch.nn.Parameter(pts.clone()),
        "scales": torch.nn.Parameter(torch.log(
            torch.tensor(nn_scale(scene.points) * 0.6, dtype=torch.float32, device=dev)
            .unsqueeze(-1).repeat(1, 3))),
        "quats": torch.nn.Parameter(torch.tensor(
            np.tile([1, 0, 0, 0], (pts.shape[0], 1)), dtype=torch.float32, device=dev)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((pts.shape[0],), 0.1,
                                                               device=dev))),
        "sh0": torch.nn.Parameter(sh0),
        "shN": torch.nn.Parameter(shN),
    }
    lrs = {"means": 1.6e-4, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
           "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {k: torch.optim.Adam([{"params": [v], "lr": lrs[k]}], eps=1e-15)
                  for k, v in params.items()}

    Strategy = make_strategy()
    strategy, state = None, None
    scene_scale = float(torch.linalg.norm(
        params["means"].detach().max(0).values - params["means"].detach().min(0).values)) + 1e-6
    if Strategy is not None:
        try:
            strategy = Strategy(verbose=False)
            strategy.check_sanity(params, optimizers)
            state = strategy.initialize_state(scene_scale=scene_scale)
            print("[gsplat] 使用 DefaultStrategy 做自适应密度控制")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] DefaultStrategy 初始化失败（{e}），改用内置简易致密化")
            strategy = None

    # 内置简易致密化（当 gsplat 不提供 strategy 时）
    grad_acc = torch.zeros(params["means"].shape[0], device=dev)
    grad_cnt = torch.zeros_like(grad_acc)
    extent = scene_scale

    def simple_densify(step):
        nonlocal grad_acc, grad_cnt
        if params["means"].grad is None or grad_cnt.max() == 0:
            return
        avg = grad_acc / grad_cnt.clamp_min(1)
        grad_acc = torch.zeros_like(avg)
        grad_cnt = torch.zeros_like(avg)
        thr = float(torch.quantile(avg.float(), 0.85)) if avg.numel() > 100 else float(avg.mean())
        hot = avg >= thr
        if hot.sum() == 0:
            return
        new = {k: [v.detach()] for k, v in params.items()}
        idx = torch.nonzero(hot).squeeze(1)
        new["means"].append(params["means"].detach()[idx] +
                            torch.randn(idx.numel(), 3, device=dev) *
                            torch.exp(params["scales"].detach()[idx]))
        new["scales"].append(params["scales"].detach()[idx] - 0.3)
        new["quats"].append(params["quats"].detach()[idx])
        new["opacities"].append(params["opacities"].detach()[idx])
        new["sh0"].append(params["sh0"].detach()[idx])
        new["shN"].append(params["shN"].detach()[idx])
        with torch.no_grad():
            for k in params:
                params[k] = torch.nn.Parameter(torch.cat(new[k], 0))
        for k in params:
            optimizers[k] = torch.optim.Adam([{"params": [params[k]], "lr": lrs[k]}], eps=1e-15)
        grad_acc = torch.zeros(params["means"].shape[0], device=dev)
        grad_cnt = torch.zeros_like(grad_acc)
        # 剪枝过透明 / 过大的
        keep = torch.sigmoid(params["opacities"].detach()) >= 0.005
        keep &= torch.exp(params["scales"].detach()).max(-1).values < 0.1 * extent
        if keep.sum() < params["means"].shape[0]:
            with torch.no_grad():
                for k in params:
                    params[k] = torch.nn.Parameter(params[k].detach()[keep])
            for k in params:
                optimizers[k] = torch.optim.Adam([{"params": [params[k]], "lr": lrs[k]}], eps=1e-15)
            grad_acc = torch.zeros(params["means"].shape[0], device=dev)
            grad_cnt = torch.zeros_like(grad_acc)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []
    t0 = time.time()
    densify_from = int(args.iters * 0.05)
    densify_until = int(args.iters * 0.6)
    reset_every = max(1, int(args.iters * 0.3))
    best = -1.0

    for step in range(1, args.iters + 1):
        i = int(np.random.randint(0, len(scene.views)))
        w, h = sizes[i]
        means = params["means"]
        quats = torch.nn.functional.normalize(params["quats"], dim=-1)
        scales = torch.exp(params["scales"])
        opac = torch.sigmoid(params["opacities"])
        sh = torch.cat([params["sh0"], params["shN"]], dim=1)

        if strategy is not None:
            try:
                strategy.step_pre_backward(params, optimizers, state, step, {})
            except Exception:  # noqa: BLE001
                pass
        rgb, alpha, info = rasterize(means, quats, scales, opac, sh,
                                     viewmats[i][None], Ks[i][None], w, h, args.sh)
        img = rgb[0]
        gt = gts[i]
        l1 = (img - gt).abs().mean()
        # D-SSIM（用简单可微实现，GS 官方是 0.2 权重）
        ssim = _ssim(img, gt)
        loss = 0.8 * l1 + 0.2 * (1 - ssim)
        loss.backward()
        if strategy is None:
            with torch.no_grad():
                grad_acc += params["means"].grad.detach().norm(dim=-1)
                grad_cnt += 1
        for o in optimizers.values():
            o.step()
            o.zero_grad(set_to_none=True)

        if densify_from <= step <= densify_until and step % 100 == 0:
            if strategy is not None:
                try:
                    strategy.step_post_backward(params, optimizers, state, step, info, packed=True)
                except TypeError:
                    try:
                        strategy.step_post_backward(params, optimizers, state, step, info)
                    except Exception:  # noqa: BLE001
                        pass
                except Exception:  # noqa: BLE001
                    pass
            else:
                simple_densify(step)
        if step % reset_every == 0:
            with torch.no_grad():
                params["opacities"].data = torch.logit(
                    torch.clamp(torch.sigmoid(params["opacities"].data), max=0.01))

        if step % 50 == 0 or step == args.iters:
            with torch.no_grad():
                p = float(10 * math.log10(1.0 / max(float(((img - gt) ** 2).mean()), 1e-12)))
            log.append({"iter": step, "loss": float(loss.item()), "psnr": p,
                        "n": int(params["means"].shape[0]),
                        "elapsed": round(time.time() - t0, 1)})
            print(f"  iter {step:5d}/{args.iters}  loss={loss.item():.4f}  PSNR={p:6.2f}  "
                  f"高斯={params['means'].shape[0]:>7d}  {time.time() - t0:7.1f}s", flush=True)

        if args.time_budget and (time.time() - t0) > args.time_budget:
            print(f"[info] 达到时间预算 {args.time_budget:.0f}s，提前结束")
            break

    # ---- 导出
    with torch.no_grad():
        means = params["means"].detach().cpu().numpy()
        scales = params["scales"].detach().cpu().numpy()
        quats = torch.nn.functional.normalize(params["quats"], dim=-1).detach().cpu().numpy()
        opac = params["opacities"].detach().cpu().numpy().reshape(-1)
        sh = torch.cat([params["sh0"], params["shN"]], 1).detach().cpu().numpy()
    ply = out / "point_cloud.ply"
    write_ply(ply, means, scales, quats, opac, sh)
    print(f"[save] {ply}（{means.shape[0]} 个高斯）")

    # ---- 质量评价 + 环绕预览
    psnrs, ssims = [], []
    with torch.no_grad():
        for i in range(0, len(scene.views), max(1, len(scene.views) // 8)):
            w, h = sizes[i]
            means_t = params["means"]
            rgb, _, _ = rasterize(means_t, torch.nn.functional.normalize(params["quats"], dim=-1),
                                  torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
                                  torch.cat([params["sh0"], params["shN"]], 1),
                                  viewmats[i][None], Ks[i][None], w, h, args.sh)
            img = rgb[0].clamp(0, 1)
            gt = gts[i]
            psnrs.append(float(10 * math.log10(1.0 / max(float(((img - gt) ** 2).mean()), 1e-12))))
            ssims.append(float(_ssim(img, gt)))
    previews = render_orbit(params, args, out, n=args.previews)

    metrics = {"backend": "gsplat", "num_gaussians": int(means.shape[0]),
               "iterations": args.iters, "sh_degree": args.sh,
               "downscale": args.downscale, "device": dev,
               "PSNR": round(float(np.mean(psnrs)), 3), "SSIM": round(float(np.mean(ssims)), 4),
               "train_seconds": round(time.time() - t0, 1),
               "views": len(scene.views), "previews": previews, "history": log[-40:]}
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(f"[done] PSNR={metrics['PSNR']}  SSIM={metrics['SSIM']}  "
          f"高斯={metrics['num_gaussians']}  耗时 {metrics['train_seconds']}s")
    return metrics


def _ssim(a, b, size: int = 11):
    import torch.nn.functional as F
    c = torch.arange(size, dtype=torch.float32, device=a.device) - size // 2
    g = torch.exp(-(c ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum())[:, None] * (g / g.sum())[None, :]
    k = g[None, None].expand(3, 1, size, size)
    x = a.permute(2, 0, 1)[None]
    y = b.permute(2, 0, 1)[None]
    pad = size // 2
    mx = F.conv2d(x, k, padding=pad, groups=3)
    my = F.conv2d(y, k, padding=pad, groups=3)
    vx = F.conv2d(x * x, k, padding=pad, groups=3) - mx ** 2
    vy = F.conv2d(y * y, k, padding=pad, groups=3) - my ** 2
    vxy = F.conv2d(x * y, k, padding=pad, groups=3) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + c1) * (2 * vxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))
    return s.mean()


def render_orbit(params, args, out: Path, n: int = 6):
    """绕场景转一圈渲染预览图（用 gsplat 自己的光栅化器）。"""
    import cv2
    with torch.no_grad():
        means = params["means"].detach()
        lo = means.min(0).values
        hi = means.max(0).values
        center = (lo + hi) / 2
        radius = float(torch.linalg.norm(hi - lo)) * 0.9 + 1e-3
        w, h = 640, 480
        fy = (h / 2) / math.tan(math.radians(45) / 2)
        K = torch.tensor([[fy, 0, w / 2], [0, fy, h / 2], [0, 0, 1]],
                         dtype=torch.float32, device=means.device)[None]
        paths = []
        for i in range(n):
            ang = 2 * math.pi * i / n
            e = math.radians(20)
            eye = center + torch.tensor(
                [radius * math.cos(ang) * math.cos(e), radius * math.sin(ang) * math.cos(e),
                 radius * math.sin(e)], dtype=torch.float32, device=means.device)
            z = (center - eye); z = z / z.norm()
            up = torch.tensor([0.0, 0, 1], device=means.device)
            x = torch.cross(up, z); x = x / x.norm()
            y = torch.cross(z, x)
            vm = torch.eye(4, device=means.device)
            vm[:3, 0], vm[:3, 1], vm[:3, 2] = x, y, z
            vm[:3, 3] = -vm[:3, :3] @ eye
            rgb, _, _ = rasterize(means, torch.nn.functional.normalize(params["quats"], dim=-1),
                                  torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
                                  torch.cat([params["sh0"], params["shN"]], 1),
                                  vm[None], K, w, h, args.sh)
            img = (rgb[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            p = out / f"orbit_{i:02d}.png"
            cv2.imwrite(str(p), img[:, :, ::-1])
            paths.append(str(p))
    print(f"[preview] 环绕渲染 {len(paths)} 张")
    return paths


def write_ply(path: Path, means, scales, quats, opac, sh):
    """标准 3DGS .ply（与官方格式互通）。"""
    n, C, _ = sh.shape
    data = {"x": means[:, 0], "y": means[:, 1], "z": means[:, 2],
            "nx": np.zeros(n), "ny": np.zeros(n), "nz": np.zeros(n),
            "f_dc_0": sh[:, 0, 0], "f_dc_1": sh[:, 0, 1], "f_dc_2": sh[:, 0, 2]}
    if C > 1:
        rest = sh[:, 1:, :].transpose(0, 2, 1).reshape(n, -1)
        for i in range(rest.shape[1]):
            data[f"f_rest_{i}"] = rest[:, i]
    data.update({"opacity": opac, "scale_0": scales[:, 0], "scale_1": scales[:, 1],
                 "scale_2": scales[:, 2], "rot_0": quats[:, 0], "rot_1": quats[:, 1],
                 "rot_2": quats[:, 2], "rot_3": quats[:, 3]})
    names = list(data.keys())
    dtype = np.dtype([(k, "<f4") for k in names])
    arr = np.empty(n, dtype=dtype)
    for k in names:
        arr[k] = np.asarray(data[k], np.float32).reshape(n)
    header = ["ply", "format binary_little_endian 1.0",
              "comment generated by aerosky + gsplat"]
    header.append(f"element vertex {n}")
    header += [f"property float {k}" for k in names]
    header.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(arr.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--sparse", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=7000)
    ap.add_argument("--sh", type=int, default=3)
    ap.add_argument("--downscale", type=int, default=1)
    ap.add_argument("--max-views", type=int, default=None)
    ap.add_argument("--max-gaussians", type=int, default=None)
    ap.add_argument("--time-budget", type=float, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--previews", type=int, default=6)
    a = ap.parse_args()
    print(f"[env] torch {torch.__version__}  CUDA={torch.cuda.is_available()}  "
          f"设备={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    import gsplat
    print(f"[env] gsplat {getattr(gsplat, '__version__', '?')}")
    train(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
