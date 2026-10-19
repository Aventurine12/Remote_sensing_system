# -*- coding: utf-8 -*-
"""模型基类与结果数据结构。

三件套（语义分割 / 目标检测 / 变化检测）统一走 BaseModel 接口：
    - load()          载入权重（失败则降级到 fallback，绝不抛异常中断系统）
    - predict_tile()  对一批无重叠小图推理，返回与任务对应的中间结果
    - postprocess()   把窗口级结果拼成整图结果（掩膜/检测框/图斑）
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..utils.geo import RasterInfo


@dataclass
class Detection:
    """一个检测目标。box 为 xyxy（普通框）或 8 个数的旋转框四角坐标（OBB）。"""

    box: np.ndarray
    score: float = 1.0
    label: str = "目标"
    label_id: int = 0
    obb: bool = False
    tile_origin: Sequence[int] = (0, 0)

    def offset(self, dx: int, dy: int) -> "Detection":
        b = self.box.copy().astype(np.float32)
        b[0::2] += dx
        b[1::2] += dy
        return Detection(b, self.score, self.label, self.label_id, self.obb, self.tile_origin)

    def xyxy(self) -> np.ndarray:
        if not self.obb:
            return self.box[:4].astype(np.float32)
        pts = self.box[:8].reshape(4, 2)
        return np.array([pts[:, 0].min(), pts[:, 1].min(),
                         pts[:, 0].max(), pts[:, 1].max()], np.float32)

    def centroid(self) -> tuple:
        x1, y1, x2, y2 = self.xyxy()
        return float((x1 + x2) / 2), float((y1 + y2) / 2)

    def area_px(self) -> float:
        if not self.obb:
            x1, y1, x2, y2 = self.xyxy()
            return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
        pts = self.box[:8].reshape(4, 2)
        x, y = pts[:, 0], pts[:, 1]
        return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


@dataclass
class TaskResult:
    """一次任务（分割/检测/变化检测）的完整结果。"""

    task: str
    raster_info: Optional[RasterInfo] = None
    mask: Optional[np.ndarray] = None            # (H, W) 0/1 或类别 id
    prob: Optional[np.ndarray] = None            # (H, W) float 前景概率
    detections: List[Detection] = field(default_factory=list)
    features: List[dict] = field(default_factory=list)   # 世界坐标 GeoJSON Feature
    stats: Dict[str, Any] = field(default_factory=dict)
    elapsed: float = 0.0
    model_name: str = ""
    fallback: bool = False

    @property
    def num_detections(self) -> int:
        return len(self.detections)

    def summary(self) -> Dict[str, Any]:
        s = dict(self.stats)
        s.update(task=self.task, model=self.model_name, fallback=self.fallback,
                 elapsed=round(self.elapsed, 3))
        if self.mask is not None:
            s["foreground_px"] = int(np.count_nonzero(self.mask))
            s["foreground_ratio"] = round(float(np.count_nonzero(self.mask) /
                                                max(self.mask.size, 1)), 5)
        if self.detections:
            s["num_detections"] = len(self.detections)
        return s


class BaseModel(ABC):
    """所有解译模型的公共父类，负责设备选择、权重定位与降级。"""

    task = "base"

    def __init__(self, cfg: dict | None = None, weights_dir: str = "weights",
                 device: str | None = None):
        self.cfg = dict(cfg or {})
        self.weights_dir = weights_dir
        self._device = device
        self.model = None
        self.fallback = False
        self.name = self.__class__.__name__
        self.load_error = ""

    # ------------------------------------------------------------------ 设备
    @property
    def device(self) -> str:
        if self._device:
            return self._device
        try:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            self._device = "cpu"
        return self._device

    # ------------------------------------------------------------------ 权重
    def weight_path(self, filename: str | None = None) -> str:
        import os
        name = filename or self.cfg.get("checkpoint") or self.cfg.get("weights") or ""
        return os.path.join(self.weights_dir, name)

    def has_weight(self, filename: str | None = None) -> bool:
        import os
        p = self.weight_path(filename)
        return bool(p) and os.path.isfile(p)

    # ------------------------------------------------------------------ 生命周期
    @abstractmethod
    def load(self) -> "BaseModel":
        """载入模型；任何失败都应当降级（self.fallback=True）而不是抛出。"""

    def predict_tile(self, tiles: np.ndarray) -> List[Any]:
        """tiles: (N,H,W,C) uint8；返回长度 N 的列表，元素含义由子类定义。

        变化检测这类需要「成对输入」的任务不走逐窗接口，而是直接实现
        predict_pair()，因此这里给一个默认实现而不是抽象方法。
        """
        raise NotImplementedError(f"{self.__class__.__name__} 未实现 predict_tile")

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def timer():
        return time.time()

    def describe(self) -> str:
        state = "降级(传统算法)" if self.fallback else "已载入"
        extra = f" | {self.load_error}" if self.load_error else ""
        return f"{self.name}[{self.task}] {state} device={self.device}{extra}"
