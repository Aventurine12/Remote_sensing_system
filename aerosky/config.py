# -*- coding: utf-8 -*-
"""全局配置：路径、模型权重、推理参数、轻量化策略。

配置文件为 YAML，默认读 configs/default.yaml；环境变量 AEROSKY_* 可覆盖常用项。
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("AEROSKY_CONFIG", ROOT / "configs" / "default.yaml"))

DEFAULTS = {
    "app": {
        "title": "空天协同的多光谱遥感影像智能解译与三维监测系统",
        "version": "1.0.0",
        "window": [1600, 950],
    },
    "paths": {
        "weights": str(ROOT / "weights"),
        "cache": str(ROOT / ".cache"),
        "output": str(ROOT / "outputs"),
        "logs": str(ROOT / "logs"),
    },
    "device": {"prefer": "cuda", "allow_cpu": True},
    "tiling": {
        # 大图分块推理："分块切割—并行推理—无缝拼接"
        "tile_size": 1024,
        "overlap": 128,
        "batch_size": 2,
        "blend": "cosine",          # none | cosine  (余弦权重无缝拼接)
        "max_workers": 2,
    },
    "models": {
        "segmentation": {
            "backend": "smp_unet",              # smp_unet | smp_deeplab | fallback
            "encoder": "resnet34",
            "encoder_weights": "imagenet",
            "checkpoint": "unet_building.pth",
            "classes": ["背景", "建筑"],
            "num_classes": 2,
            "input_size": [512, 512],
            "threshold": 0.5,
        },
        "detection": {
            "backend": "ultralytics_obb",       # ultralytics_obb | ultralytics_det | fallback
            "weights": "yolo11n-obb.pt",        # DOTAv1 预训练（航拍 15 类，含车辆、网球场等）
            "conf": 0.25,
            "iou": 0.45,
            "input_size": 1024,
            "fallback_weights": "yolov8n.pt",
        },
        "change_detection": {
            "backend": "siamunet_diff",         # siamunet_diff | fallback
            "checkpoint": "siamunet_diff_whucd.pth",
            "input_size": [256, 256],
            "base_channels": 16,
            "threshold": 0.5,
            "min_area_px": 64,                  # 小于该面积的图斑作为噪声剔除
        },
    },
    "gsplat": {
        # 轻量化策略：局部重建 + 参数精简
        "max_gaussians": 100000,
        "sh_degree": 1,                          # 球谐阶数由 3 阶降至 1 阶
        "iterations": 400,
        "learning_rate": {"position": 1.6e-4, "feature": 2.5e-3,
                          "opacity": 5e-2, "scaling": 5e-3, "rotation": 1e-3},
        "densify_from": 200,
        "densify_until": 500,
        "densify_grad_threshold": 0.0,   # 0 = 自适应阈值
        "opacity_reset_interval": 300,
        "downscale": 2,                          # 训练时影像降采样倍率
        "colmap": {"binary": False, "images_dir": "images", "sparse_dir": "sparse/0"},
    },
    "registration": {
        "crs": "EPSG:4326",
        "mode": "affine",                        # affine | similarity | manual
        "ransac_iters": 400,
        "max_rmse_m": 3.0,
    },
    "report": {
        "org": "哈尔滨工业大学 大学生创新训练计划",
        "template": "default",
        "formats": ["html", "pdf", "geojson"],
    },
    "uav_task": {
        "front_overlap": 0.8,
        "side_overlap": 0.7,
        "flight_height_m": 120,
        "gsd_cm": 3.0,
        "buffer_m": 10.0,
    },
}


def _deep_merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config(dict):
    """支持 cfg.a.b 与 cfg["a"]["b"] 两种访问方式的配置对象。"""

    def __getattr__(self, item):
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def get_path(self, dotted: str, default=None):
        cur = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def ensure_dirs(self):
        for key in ("weights", "cache", "output", "logs"):
            Path(self.get_path(f"paths.{key}")).mkdir(parents=True, exist_ok=True)
        return self


def load_config(path: str | os.PathLike | None = None, **overrides) -> Config:
    """读取 YAML 配置并与内置默认值深合并。"""
    cfg = copy.deepcopy(DEFAULTS)
    p = Path(path) if path else CONFIG_PATH
    if p.exists():
        with open(p, encoding="utf-8") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    out = Config(cfg)
    out.ensure_dirs()
    return out


def save_config(cfg: dict, path: str | os.PathLike) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(cfg), f, allow_unicode=True, sort_keys=False)
    return str(p)
