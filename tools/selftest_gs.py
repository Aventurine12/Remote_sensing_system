# -*- coding: utf-8 -*-
"""3DGS 核心自检：合成数据 -> 训练 -> 渲染 -> 导出 .ply -> 重新载入渲染。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from aerosky.gsplat.camera import OrbitCamera
from aerosky.gsplat.ply_io import ply_to_gaussians
from aerosky.gsplat.render_np import SplatRenderer
from aerosky.gsplat.synthetic import make_dataset
from aerosky.gsplat.trainer import GaussianTrainer

OUT = Path(r"D:\Code\dc\aerosky\outputs\selftest3d")
OUT.mkdir(parents=True, exist_ok=True)

t0 = time.time()
info = make_dataset(OUT, n_views=16, width=200, height=150, n_points=4000)
print(f"dataset  {time.time() - t0:.1f}s")

cfg = {"max_gaussians": 60000, "sh_degree": 1, "downscale": 1, "iterations": 400,
       "densify_from": 50, "densify_until": 350, "opacity_reset_interval": 150,
       "densify_grad_threshold": 0.0, "learning_rate": {}}
tr = GaussianTrainer(cfg, device="cpu")
t0 = time.time()
tr.prepare(info["images_dir"], info["sparse_dir"])
print(f"prepare  {time.time() - t0:.1f}s  gaussians={tr.model.num}")


def prog(i, n, rec):
    if i % 20 == 0 or i == n:
        print(f"  it{i:4d} loss={rec['loss']:.4f} psnr={rec['psnr']:.2f} "
              f"n={rec['n_gauss']} pairs={rec['pairs']}")


t0 = time.time()
res = tr.train(iterations=400, progress=prog)
print(f"train    {time.time() - t0:.1f}s  ->", {k: v for k, v in res.items() if k != "history"})
print("eval     ", {k: round(v, 3) for k, v in tr.evaluate(max_views=4).items()})

saved = tr.save(OUT / "out")
print("saved    ", saved["ply"])
for i, img in enumerate(tr.render_orbit(n=3, width=200, height=150)):
    cv2.imwrite(str(OUT / f"orbit{i}.png"), img[:, :, ::-1])

d = ply_to_gaussians(saved["ply"])
print("ply      ", {k: v.shape for k, v in d.items()})
r = SplatRenderer(d["means"], d["scales"], d["quats"], d["opacities"], d["shs"], 1)
c, s = r.center_scale()
cam = OrbitCamera(target=np.asarray(c), distance=s * 1.3).to_camera(200, 150)
t0 = time.time()
img, acc = r.render(cam)
print(f"np render {(time.time() - t0) * 1000:.0f} ms  {img.shape}  covered px={int((acc > 0).sum())}")
cv2.imwrite(str(OUT / "np_render.png"), img[:, :, ::-1])
print("OK")
