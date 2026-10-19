# -*- coding: utf-8 -*-
"""目标检测后端。

默认用 DOTAv1 预训练的 YOLO-OBB（旋转框）：航拍 15 类，正好覆盖项目关心的
"车辆 / 临时棚屋（small vehicle、large vehicle、storage tank、bridge…）"等目标。
没有 OBB 权重时退回 COCO 通用权重，再退回传统连通域检测，保证系统可用。

DOTAv1 类别（与前端图例一致）：
    plane, ship, storage tank, baseball diamond, tennis court, swimming pool,
    ground track field, harbor, bridge, large vehicle, small vehicle,
    helicopter, roundabout, soccer ball field, basketball court
"""
from __future__ import annotations

import os
from typing import List

import numpy as np

from ..utils.logger import get_logger
from .base import BaseModel, Detection

LOG = get_logger("inference.det")

DOTA_NAMES = {
    0: "飞机", 1: "船舶", 2: "储罐", 3: "棒球场", 4: "网球场", 5: "游泳池",
    6: "田径场", 7: "港口", 8: "桥梁", 9: "大型车辆", 10: "小型车辆",
    11: "直升机", 12: "环岛", 13: "足球场", 14: "篮球场",
}


class DetectionModel(BaseModel):
    task = "detection"

    def __init__(self, cfg=None, weights_dir="weights", device=None):
        super().__init__(cfg, weights_dir, device)
        self.backend = ""
        self.class_names = DOTA_NAMES
        self.obb = False

    # ------------------------------------------------------------------ 载入
    def load(self) -> "DetectionModel":
        if self._load_ultralytics():
            return self
        self.backend = "blob"
        self.fallback = True
        LOG.info("目标检测后端：blob（传统连通域兜底）")
        return self

    def _load_ultralytics(self) -> bool:
        cfg_w = self.cfg.get("weights", "yolo11n-obb.pt")
        local = self.weight_path(cfg_w)
        try:
            from ultralytics import YOLO
        except Exception as e:  # noqa: BLE001
            self.load_error = f"ultralytics 不可用: {e}"
            return False
        path = local if os.path.isfile(local) else None
        if path is None:
            from ..utils.weights import ensure_dota_obb, ensure_yolo_coco
            path = ensure_dota_obb(self.weights_dir)
            if path is None:
                path = ensure_yolo_coco(self.weights_dir)
        if path is None:
            self.load_error = "无可用权重"
            return False
        try:
            self.model = YOLO(path)
            self.obb = "obb" in os.path.basename(path).lower()
            names = getattr(self.model, "names", None) or {}
            if self.obb and len(names) >= 15:
                self.class_names = {int(k): DOTA_NAMES.get(int(k), str(v)) for k, v in names.items()}
                self.name = f"YOLO-OBB(DOTAv1) {os.path.basename(path)}"
            else:
                self.class_names = {int(k): str(v) for k, v in names.items()}
                self.name = f"YOLO(COCO) {os.path.basename(path)}"
            LOG.info("目标检测后端：%s，共 %d 类", self.name, len(self.class_names))
            return True
        except Exception as e:  # noqa: BLE001
            self.load_error = str(e)
            return False

    # ------------------------------------------------------------------ 推理
    def predict_tile(self, tiles: np.ndarray) -> List[List[Detection]]:
        if self.backend == "blob" or self.model is None:
            return [self._predict_blob(t) for t in tiles]
        conf = float(self.cfg.get("conf", 0.25))
        iou = float(self.cfg.get("iou", 0.45))
        imgsz = int(self.cfg.get("input_size", 1024))
        res = self.model.predict(list(tiles), verbose=False, conf=conf, iou=iou,
                                 imgsz=imgsz, device=self.device)
        out: List[List[Detection]] = []
        for r in res:
            dets: List[Detection] = []
            boxes = r.obb if getattr(r, "obb", None) is not None else r.boxes
            if boxes is None or len(boxes) == 0:
                out.append(dets)
                continue
            xy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else boxes.xywhr.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            sco = boxes.conf.cpu().numpy()
            if self.obb and getattr(r, "obb", None) is not None:
                polys = r.obb.xyxyxyxy.cpu().numpy()      # (N,4,2)
                for p, c, s in zip(polys, cls, sco):
                    dets.append(Detection(p.reshape(-1).astype(np.float32), float(s),
                                          self.class_names.get(int(c), str(c)), int(c), True))
            else:
                for b, c, s in zip(xy, cls, sco):
                    dets.append(Detection(b.astype(np.float32), float(s),
                                          self.class_names.get(int(c), str(c)), int(c), False))
            out.append(dets)
        return out

    def _predict_blob(self, tile: np.ndarray) -> List[Detection]:
        """传统兜底：暗背景上找规则的亮色小块，当作"疑似目标"。"""
        import cv2
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 41, -12)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets: List[Detection] = []
        for c in cnts:
            a = cv2.contourArea(c)
            if not (40 < a < 4000):
                continue
            x, y, w, h = cv2.boundingRect(c)
            if not (0.4 < w / max(h, 1) < 2.5):
                continue
            dets.append(Detection(np.array([x, y, x + w, y + h], np.float32),
                                  0.5, "疑似目标", 0, False))
        return dets[:200]

    def class_legend(self) -> dict:
        return self.class_names
