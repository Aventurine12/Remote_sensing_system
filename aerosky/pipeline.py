# -*- coding: utf-8 -*-
"""端到端流程编排：把「影像 -> 模型 -> 地理坐标 -> 图斑/检测框 -> 报告」串起来。

这是 GUI 与命令行共用的核心。所有任务都返回统一的 TaskResult（含世界坐标 GeoJSON）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import load_config
from .inference.base import Detection, TaskResult
from .inference.registry import ModelHub
from .inference.tiler import Tiler
from .utils.geo import (Raster, feature_bounds_lonlat, feature_centroid,
                        geojson_of, mask_to_polygons, save_geojson)
from .utils.logger import get_logger
from .utils.vis import CLASS_COLORS, colorize_mask, iou_f1

LOG = get_logger("pipeline")


class Pipeline:
    """系统主流程。典型用法::

        pipe = Pipeline()
        r = Raster.open("before.tif")
        seg = pipe.segment(r)
        pipe.save_geojson(seg, "out/seg.geojson")
    """

    def __init__(self, cfg=None, device: Optional[str] = None, preload: bool = False):
        self.cfg = cfg or load_config()
        self.hub = ModelHub(self.cfg, device=device)
        t = self.cfg.get_path("tiling", {}) or {}
        self.tiler = Tiler(tile_size=t.get("tile_size", 1024), overlap=t.get("overlap", 128),
                           batch_size=t.get("batch_size", 2), blend=t.get("blend", "cosine"),
                           max_workers=t.get("max_workers", 2))
        self.output_dir = Path(self.cfg.get_path("paths.output", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if preload:
            self.hub.preload_all()

    # ================================================================== 语义分割
    def segment(self, raster: Raster, with_features: bool = True,
                progress=None) -> TaskResult:
        t0 = time.time()
        model = self.hub.get("segmentation")
        rgb = raster.rgb
        size = tuple(model.cfg.get("input_size", [512, 512]))

        def predict(tiles):
            outs = []
            for t in tiles:
                p = model.predict_tile([t])[0]
                outs.append(_resize(p, t.shape[:2]))
            return np.stack(outs)[:, None]

        if model.backend == "spectral":
            prob = self.tiler.run_mask_threaded(rgb, predict, progress=progress)
        else:
            prob = self.tiler.run_mask(rgb, predict, progress=progress)
        prob = prob[:, :, 0]
        labels = model.classify(rgb, prob)

        feats: List[dict] = []
        if with_features:
            for cls in (1, 2, 3, 4):
                m = (labels == cls).astype(np.uint8)
                if m.sum() == 0:
                    continue
                for f in mask_to_polygons(m, raster.info.transform,
                                          min_area_px=64 if cls == 1 else 256):
                    f["properties"]["class"] = cls
                    f["properties"]["class_name"] = model.class_names[cls]
                    feats.append(f)
        res = TaskResult(
            task="segmentation", raster_info=raster.info, mask=labels, prob=prob,
            features=feats, model_name=model.name, fallback=model.fallback,
            elapsed=time.time() - t0,
        )
        res.stats["class_pixels"] = {model.class_names[i]: int((labels == i).sum())
                                     for i in range(len(model.class_names))
                                     if (labels == i).sum() > 0}
        res.stats["num_polygons"] = len(feats)
        return res

    # ================================================================== 目标检测
    def detect(self, raster: Raster, progress=None) -> TaskResult:
        t0 = time.time()
        model = self.hub.get("detection")
        rgb = raster.rgb
        dets = self.tiler.run_detection(rgb, model.predict_tile, progress=progress)

        feats = []
        for d in dets:
            feats.append(_detection_feature(d, raster, model.class_names))
        res = TaskResult(task="detection", raster_info=raster.info, detections=dets,
                         features=feats, model_name=model.name, fallback=model.fallback,
                         elapsed=time.time() - t0)
        by_class: Dict[str, int] = {}
        for d in dets:
            by_class[d.label] = by_class.get(d.label, 0) + 1
        res.stats["by_class"] = by_class
        res.stats["num_detections"] = len(dets)
        return res

    # ================================================================== 变化检测
    def change(self, raster_a: Raster, raster_b: Raster, progress=None) -> TaskResult:
        t0 = time.time()
        model = self.hub.get("change_detection")
        rgb_a = raster_a.rgb
        rgb_b = _align_to(raster_b.rgb, rgb_a.shape[:2])

        if model.backend == "mask_diff":
            # 前后时相各跑一次分块推理，再做掩膜差分（最稳的方案）
            seg = self.hub.get("segmentation")
            pa = self._building_prob(rgb_a, seg, progress)
            pb = self._building_prob(rgb_b, seg, progress)
            prob = model._clean(np.maximum(np.clip(pb - pa, 0, 1),
                                           np.clip(pa - pb, 0, 1) * 0.8))
            new_b = np.clip(pb - pa, 0, 1)
            del_b = np.clip(pa - pb, 0, 1)
        else:
            prob = model.predict_pair(rgb_a, rgb_b)
            new_b = prob
            del_b = np.zeros_like(prob)

        mask = (prob >= model.threshold).astype(np.uint8)
        feats = []
        for m, kind in ((new_b >= model.threshold, "新增"), (del_b >= model.threshold, "拆除")):
            if not np.any(m):
                continue
            for f in mask_to_polygons(m.astype(np.uint8), raster_a.info.transform,
                                      min_area_px=model.min_area_px):
                f["properties"]["change_type"] = kind
                feats.append(f)
        res = TaskResult(task="change_detection", raster_info=raster_a.info, mask=mask,
                         prob=prob, features=feats, model_name=model.name,
                         fallback=model.fallback, elapsed=time.time() - t0)
        mx, my = raster_a.pixel_size_m()
        res.stats["num_changes"] = len(feats)
        res.stats["new_count"] = sum(1 for f in feats if f["properties"].get("change_type") == "新增")
        res.stats["del_count"] = sum(1 for f in feats if f["properties"].get("change_type") == "拆除")
        res.stats["changed_area_m2"] = round(float(mask.sum()) * mx * my, 2)
        res.stats["pixel_size_m"] = [round(mx, 3), round(my, 3)]
        return res

    def _building_prob(self, rgb: np.ndarray, seg, progress=None) -> np.ndarray:
        def predict(tiles):
            return np.stack([seg.predict_tile([t])[0] for t in tiles])[:, None]
        return self.tiler.run_mask(rgb, predict, progress=progress)[:, :, 0]

    # ================================================================== 输出
    def to_geojson(self, res: TaskResult, to_wgs84: bool = True) -> dict:
        crs = res.raster_info.crs if res.raster_info else None
        return geojson_of(res.features, to_wgs84=to_wgs84, src_crs=crs)

    def save_geojson(self, res: TaskResult, path: str | Path | None = None) -> str:
        path = Path(path) if path else self.output_dir / f"{res.task}.geojson"
        return save_geojson(path, self.to_geojson(res))

    def save_mask(self, raster: Raster, res: TaskResult,
                  path: str | Path | None = None, overlay: bool = False) -> str:
        path = Path(path) if path else self.output_dir / f"{res.task}_mask.tif"
        if res.mask is None:
            raise ValueError("该任务没有掩膜结果")
        arr = colorize_mask(res.mask) if res.mask.max() > 1 else (res.mask * 255).astype(np.uint8)
        if overlay:
            from .utils.geo import overlay_mask
            arr = overlay_mask(raster.rgb, res.mask, color=(255, 60, 60), alpha=0.5)
        return raster.save(path, arr, dtype="uint8")

    def summary_json(self, results: List[TaskResult],
                     path: str | Path | None = None) -> str:
        path = Path(path) if path else self.output_dir / "summary.json"
        payload = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tasks": [r.summary() for r in results],
            "models": self.hub.status(),
            "config": {"tiling": self.cfg.get_path("tiling"),
                       "gsplat": {k: v for k, v in (self.cfg.get_path("gsplat") or {}).items()
                                  if k != "learning_rate"}},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return str(path)


# --------------------------------------------------------------------------- 工具
def _resize(img: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    import cv2
    h, w = int(hw[0]), int(hw[1])
    if img.shape[:2] == (h, w):
        return img.astype(np.float32)
    return cv2.resize(img.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def _align_to(img: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    return _resize(img, hw)


def _detection_feature(d: Detection, raster: Raster, names: Dict[int, str]) -> dict:
    """把像素框/旋转框换算成经纬度多边形。"""
    if d.obb:
        pts = d.box[:8].reshape(4, 2).astype(np.float64)
    else:
        x1, y1, x2, y2 = d.box[:4]
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float64)
    lon, lat = raster.xy_to_lonlat(pts[:, 0], pts[:, 1])
    ring = [[float(a), float(b)] for a, b in zip(np.ravel(lon), np.ravel(lat))]
    ring.append(ring[0])
    cx, cy = d.centroid()
    clon, clat = raster.xy_to_lonlat([cx], [cy])
    mx, my = raster.pixel_size_m()
    return {
        "type": "Feature",
        "properties": {
            "class_id": int(d.label_id),
            "class_name": d.label,
            "score": round(float(d.score), 4),
            "area_m2": round(float(d.area_px()) * mx * my, 2),
            "center_lon": float(clon[0]),
            "center_lat": float(clat[0]),
            "obb": bool(d.obb),
        },
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


def center_of(feat: dict) -> Tuple[float, float]:
    """优先用属性里的中心，其次用几何中心。"""
    p = feat.get("properties", {})
    if "center_lon" in p and "center_lat" in p:
        return float(p["center_lon"]), float(p["center_lat"])
    return feature_centroid(feat)


__all__ = ["Pipeline", "center_of", "iou_f1", "CLASS_COLORS", "feature_bounds_lonlat",
           "colorize_mask"]
