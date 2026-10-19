# -*- coding: utf-8 -*-
"""模型注册表与模型池：按任务名构建模型，并让变化检测复用同一个分割模型。"""
from __future__ import annotations

from typing import Dict, Optional

from ..utils.logger import get_logger
from .base import BaseModel
from .change_detection import ChangeDetectionModel
from .detection import DetectionModel
from .segmentation import SegmentationModel

LOG = get_logger("inference.registry")

BUILDERS = {
    "segmentation": SegmentationModel,
    "detection": DetectionModel,
    "change_detection": ChangeDetectionModel,
}


def available_models() -> Dict[str, str]:
    return {k: v.__name__ for k, v in BUILDERS.items()}


def build_model(task: str, cfg: dict | None = None, weights_dir: str = "weights",
                device: Optional[str] = None, segmenter: Optional[BaseModel] = None) -> BaseModel:
    if task not in BUILDERS:
        raise KeyError(f"未知任务：{task}，可选 {list(BUILDERS)}")
    if task == "change_detection":
        return ChangeDetectionModel(cfg, weights_dir, device, segmenter=segmenter)
    return BUILDERS[task](cfg, weights_dir, device)


class ModelHub:
    """按需构建并缓存三个模型；变化检测自动共享语义分割模型。"""

    def __init__(self, cfg, device: Optional[str] = None):
        self.cfg = cfg
        self.device = device
        self.weights_dir = str(cfg.get_path("paths.weights", "weights"))
        self._cache: Dict[str, BaseModel] = {}

    def get(self, task: str) -> BaseModel:
        if task in self._cache:
            return self._cache[task]
        sub = dict(self.cfg.get_path(f"models.{task}", {}) or {})
        seg = self.get("segmentation") if task == "change_detection" else None
        model = build_model(task, sub, self.weights_dir, self.device, segmenter=seg)
        model.load()
        LOG.info("模型就绪：%s", model.describe())
        self._cache[task] = model
        return model

    def preload_all(self) -> Dict[str, BaseModel]:
        for t in ("segmentation", "detection", "change_detection"):
            try:
                self.get(t)
            except Exception as e:  # noqa: BLE001
                LOG.warning("预载 %s 失败：%s", t, e)
        return self._cache

    def status(self) -> Dict[str, dict]:
        out = {}
        for t in ("segmentation", "detection", "change_detection"):
            m = self._cache.get(t)
            out[t] = {"loaded": m is not None,
                      "name": getattr(m, "name", ""),
                      "fallback": getattr(m, "fallback", None),
                      "device": getattr(m, "device", None)}
        return out
