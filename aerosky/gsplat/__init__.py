# -*- coding: utf-8 -*-
"""3D 高斯泼溅（3D Gaussian Splatting）模块。

    camera.py      针孔相机与轨道相机
    sh.py          球谐函数（颜色编码）
    rasterizer.py  可微分光栅化（PyTorch，用于训练）
    render_np.py   纯 numpy 前向渲染（用于前端实时浏览）
    model.py       可训练的高斯模型与自适应密度控制
    trainer.py     训练流程
    ply_io.py      标准 3DGS .ply 读写
    colmap_io.py   COLMAP 稀疏重建读写
    synthetic.py   离线演示用合成多视角数据集
"""
from .camera import Camera, OrbitCamera  # noqa: F401
from .model import GaussianModel  # noqa: F401
from .ply_io import ply_to_gaussians, read_ply  # noqa: F401
from .render_np import SplatRenderer  # noqa: F401
from .trainer import GaussianTrainer, load_dataset  # noqa: F401

__all__ = ["Camera", "OrbitCamera", "GaussianModel", "GaussianTrainer",
           "SplatRenderer", "load_dataset", "read_ply", "ply_to_gaussians"]
