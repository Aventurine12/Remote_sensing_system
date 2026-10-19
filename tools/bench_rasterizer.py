# -*- coding: utf-8 -*-
"""3DGS 性能基准：分别测前向 / 反向耗时，定位瓶颈。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from aerosky.gsplat.rasterizer import camera_K, camera_viewmat, render_gaussians
from aerosky.gsplat.sh import num_coeffs
from aerosky.utils.logger import get_logger

get_logger("bench")
torch.set_num_threads(max(1, torch.get_num_threads()))
print("线程数:", torch.get_num_threads(), "| 逻辑核:", torch.get_num_threads())

W, H = 160, 120
dev = "cpu"
rng = np.random.default_rng(0)
means = torch.tensor(rng.uniform(-10, 10, (8000, 3)), dtype=torch.float32, device=dev,
                     requires_grad=True)
scales = torch.tensor(rng.uniform(0.05, 0.4, (8000, 3)), dtype=torch.float32, device=dev,
                      requires_grad=True)
quats = torch.tensor(rng.normal(0, 1, (8000, 4)), dtype=torch.float32, device=dev,
                     requires_grad=True)
opac = torch.tensor(rng.uniform(0.05, 0.6, (8000,)), dtype=torch.float32, device=dev,
                    requires_grad=True)
shs = torch.zeros(8000, num_coeffs(1), 3, device=dev, requires_grad=True)

viewmat = torch.eye(4, dtype=torch.float32)
viewmat[2, 3] = 26.0
K = camera_K(184.0, 184.0, W / 2, H / 2)

for tag, scale_mult in (("初始", 1.0), ("半径夹紧后", 1.0)):
    for it in (1, 2):
        t0 = time.time()
        img, acc, info = render_gaussians(means, scales * scale_mult, quats, opac, shs,
                                          viewmat, K, W, H, sh_degree=1)
        t1 = time.time()
        loss = img.mean()
        loss.backward()
        t2 = time.time()
        for p in (means, scales, quats, opac, shs):
            p.grad = None
        print(f"{tag} 第{it}次：前向 {(t1 - t0) * 1000:7.0f} ms | 反向 {(t2 - t1) * 1000:7.0f} ms "
              f"| 候选对 {info['num_pairs'] / 1e6:.2f}M | 可见 {info['visible']}")
