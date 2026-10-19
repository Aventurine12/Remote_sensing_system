# -*- coding: utf-8 -*-
"""语义分割后端。

三种可选实现，按优先级自动降级：

1. ``yolo_building``  —— 卫星影像建筑实例分割（HuggingFace: keremberke/yolov8s-building-segmentation，
   在 keremberke/satellite-building-segmentation 上训练），实例掩膜合并成"建筑"语义类，
   再结合光谱指数（ExG 植被指数、NDWI 水体指数）得到 建筑/道路/植被/水体/裸地 五类图。
2. ``smp_unet``       —— segmentation-models-pytorch 的 U-Net（ResNet34 编码器），
   可用 ``tools/train_segmentation.py`` 在 WHU / Massachusetts 建筑数据集上微调后载入。
3. ``spectral``       —— 纯光谱+形态学的传统算法，无权重也能出图，保证系统永远可用。

输出统一为 (H, W, C) 概率或 (H, W) 类别图，由 Tiler 负责分块拼接。
"""
from __future__ import annotations

import os
from typing import List, Sequence

import numpy as np

from ..utils.logger import get_logger
from .base import BaseModel

LOG = get_logger("inference.seg")

CLASS_NAMES = ["背景", "建筑", "道路", "植被", "水体", "裸地"]

# 模型可能的落地位置（按顺序探测）
BUILDING_SEG_CANDIDATES = [
    "yolov8s-building-segmentation.pt",
    "yolov8n-building-segmentation.pt",
    "yolov8m-building-segmentation.pt",
]


class SegmentationModel(BaseModel):
    task = "segmentation"

    def __init__(self, cfg=None, weights_dir="weights", device=None):
        super().__init__(cfg, weights_dir, device)
        self.backend = ""
        self.num_classes = int(self.cfg.get("num_classes", 6))
        self.threshold = float(self.cfg.get("threshold", 0.5))
        self.class_names = list(self.cfg.get("classes") or CLASS_NAMES)

    # ------------------------------------------------------------------ 载入
    def load(self) -> "SegmentationModel":
        wanted = str(self.cfg.get("backend", "yolo_building"))
        order = [wanted] + [b for b in ("yolo_building", "smp_unet", "spectral") if b != wanted]
        for b in order:
            try:
                if b == "yolo_building" and self._load_yolo_building():
                    self.backend = "yolo_building"
                    LOG.info("语义分割后端：%s（建筑实例分割 + 光谱指数）", self.backend)
                    return self
                if b == "smp_unet" and self._load_smp():
                    self.backend = "smp_unet"
                    LOG.info("语义分割后端：%s（U-Net 微调权重）", self.backend)
                    return self
                if b == "spectral":
                    self.backend = "spectral"
                    self.fallback = True
                    LOG.info("语义分割后端：spectral（传统算法兜底）")
                    return self
            except Exception as e:  # noqa: BLE001
                self.load_error = f"{b}: {e}"
                LOG.warning("语义分割后端 %s 载入失败：%s", b, e)
        self.backend = "spectral"
        self.fallback = True
        return self

    def _find_building_weights(self):
        for name in BUILDING_SEG_CANDIDATES:
            p = self.weight_path(name)
            if os.path.isfile(p):
                return p
        return None

    def _load_yolo_building(self) -> bool:
        from ultralytics import YOLO
        p = self._find_building_weights()
        if p is None:
            # 允许从 HuggingFace 自动拉取（首次联网）
            from ..utils.weights import ensure_building_segmentation
            p = ensure_building_segmentation(self.weights_dir)
        if not p:
            return False
        self.model = YOLO(p)
        self.name = "YOLOv8-seg(建筑提取, 卫星影像)"
        return True

    def _load_smp(self) -> bool:
        import torch
        import segmentation_models_pytorch as smp
        ckpt = self.weight_path(self.cfg.get("checkpoint", "unet_building.pth"))
        if not os.path.isfile(ckpt):
            return False
        model = smp.Unet(encoder_name=self.cfg.get("encoder", "resnet34"),
                         encoder_weights=None, in_channels=3, classes=self.num_classes)
        state = torch.load(ckpt, map_location="cpu")
        state = state.get("state_dict", state.get("model", state))
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model.eval().to(self.device)
        self.model = model
        self.name = f"U-Net({self.cfg.get('encoder', 'resnet34')})"
        return True

    # ------------------------------------------------------------------ 推理
    def predict_tile(self, tiles: np.ndarray) -> List[np.ndarray]:
        if self.backend == "yolo_building":
            return [self._predict_yolo(t) for t in tiles]
        if self.backend == "smp_unet":
            return self._predict_smp(tiles)
        return [self._predict_spectral(t) for t in tiles]

    # ---- YOLO 建筑实例分割 -> 语义前景概率
    def _predict_yolo(self, tile: np.ndarray) -> np.ndarray:
        res = self.model.predict(tile, verbose=False, conf=0.20, iou=0.5,
                                 imgsz=int(self.cfg.get("input_size", [512, 512])[0]), device=self.device)
        h, w = tile.shape[:2]
        prob = np.zeros((h, w), np.float32)
        for r in res:
            if r.masks is None:
                continue
            masks = r.masks.data.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.ones(len(masks))
            for m, c in zip(masks, confs):
                if m.shape != (h, w):
                    import cv2
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                prob = np.maximum(prob, (m > 0.5).astype(np.float32) * float(c))
        return prob

    # ---- U-Net 多类分割
    def _predict_smp(self, tiles: np.ndarray) -> List[np.ndarray]:
        import torch
        size = tuple(self.cfg.get("input_size", [512, 512]))
        batch = np.stack([_resize(t, size) for t in tiles]).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        batch = (batch - mean) / std
        with torch.no_grad():
            x = torch.from_numpy(batch.transpose(0, 3, 1, 2)).to(self.device)
            logits = self.model(x)
            if logits.shape[1] == 1:
                p = torch.sigmoid(logits)[:, 0]
                outs = p.cpu().numpy()
            else:
                outs = torch.softmax(logits, 1)[:, 1].cpu().numpy()
        return [_resize(o, tiles[i].shape[:2], prob=True) for i, o in enumerate(outs)]

    # ---- 光谱 + 形态学兜底
    def _predict_spectral(self, tile: np.ndarray) -> np.ndarray:
        import cv2
        rgb = tile.astype(np.float32)
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        exg = 2 * g - r - b                     # 超绿指数 -> 植被
        brightness = rgb.mean(axis=2)
        # 建筑：亮且纹理规则（用局部方差衡量平滑屋顶），并做形态学清理
        smooth = cv2.blur(brightness, (5, 5))
        lap = np.abs(cv2.Laplacian(brightness, cv2.CV_32F))
        bld = ((smooth > np.percentile(smooth, 62)) & (lap < np.percentile(lap, 70))) \
            & (exg < np.percentile(exg, 55))
        bld = cv2.morphologyEx(bld.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        bld = cv2.morphologyEx(bld, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return bld.astype(np.float32)

    # ------------------------------------------------------------------ 多类合成
    def classify(self, rgb: np.ndarray, building_prob: np.ndarray) -> np.ndarray:
        """由建筑概率 + 光谱指数合成 5 类地物图（0 背景/1 建筑/2 道路/3 植被/4 水体/5 裸地）。"""
        a = rgb.astype(np.float32)
        r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
        exg = 2 * g - r - b
        ndwi = (g.astype(np.float32) - b.astype(np.float32)) / (g + b + 1e-6)
        gray = a.mean(axis=2)
        lab = np.zeros(rgb.shape[:2], np.uint8)
        water = ndwi > 0.08
        veg = (exg > np.percentile(exg, 78)) & ~water
        bld = (building_prob >= self.threshold) & ~water
        road = (~bld) & (~veg) & (~water) & (gray > np.percentile(gray, 30)) \
            & (gray < np.percentile(gray, 72))
        lab[bld] = 1
        lab[road] = 2
        lab[veg] = 3
        lab[water] = 4
        lab[(lab == 0) & (gray >= np.percentile(gray, 72)) & ~water] = 5
        return lab


def _resize(img: np.ndarray, size: Sequence[int], prob: bool = False) -> np.ndarray:
    import cv2
    h, w = int(size[0]), int(size[1])
    if img.shape[0] == h and img.shape[1] == w:
        return img
    interp = cv2.INTER_LINEAR if (prob or img.dtype != np.uint8) else cv2.INTER_AREA
    return cv2.resize(img, (w, h), interpolation=interp)
