# -*- coding: utf-8 -*-
"""可视化与评价指标：调色板、检测框/图斑绘制、IoU/F1 等指标。"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

# 地物调色板（与前端图例一致）
PALETTE: Dict[str, Tuple[int, int, int]] = {
    "建筑": (214, 132, 122),
    "道路": (170, 170, 170),
    "植被": (86, 156, 84),
    "水体": (78, 128, 196),
    "裸地": (176, 160, 132),
    "变化": (255, 60, 60),
}

CLASS_COLORS = [
    (0, 0, 0), (214, 132, 122), (170, 170, 170), (86, 156, 84),
    (78, 128, 196), (176, 160, 132), (255, 60, 60),
]


def colorize_mask(mask: np.ndarray, colors: Sequence[Sequence[int]] = CLASS_COLORS) -> np.ndarray:
    """类别掩膜 -> 彩色图。"""
    m = np.asarray(mask).astype(np.int32)
    out = np.zeros((*m.shape[:2], 3), np.uint8)
    for idx, c in enumerate(colors):
        out[m == idx] = c
    return out


def draw_boxes(img: np.ndarray, boxes: np.ndarray, labels: Sequence[str] | None = None,
               scores: Sequence[float] | None = None, color=(0, 255, 0),
               thickness: int = 2, obb: bool = False) -> np.ndarray:
    """在影像上画检测框；obb=True 时 boxes 为 (N,8) 旋转框四角坐标。"""
    import cv2
    out = img.copy()
    boxes = np.asarray(boxes)
    if boxes.size == 0:
        return out
    for i, b in enumerate(boxes):
        if obb and b.size >= 8:
            pts = b[:8].reshape(4, 2).astype(np.int32)
            cv2.polylines(out, [pts], True, color, thickness)
            anchor = pts.min(axis=0)
        else:
            x1, y1, x2, y2 = b[:4].astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            anchor = np.array([x1, y1])
        text = ""
        if labels is not None and i < len(labels):
            text = str(labels[i])
        if scores is not None and i < len(scores):
            text += f" {float(scores[i]):.2f}"
        if text:
            cv2.putText(out, text, (int(anchor[0]), max(12, int(anchor[1]) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------- 评价指标
def confusion(pred: np.ndarray, gt: np.ndarray, num_classes: int = 2) -> np.ndarray:
    p = np.asarray(pred).astype(np.int64).ravel()
    g = np.asarray(gt).astype(np.int64).ravel()
    valid = (g >= 0) & (g < num_classes) & (p >= 0) & (p < num_classes)
    k = num_classes
    idx = g[valid] * k + p[valid]
    return np.bincount(idx, minlength=k * k).reshape(k, k)


def iou_f1(pred: np.ndarray, gt: np.ndarray, num_classes: int = 2,
           foreground: int = 1) -> Dict[str, float]:
    cm = confusion(pred, gt, num_classes)
    tp = cm[foreground, foreground]
    fp = cm[:, foreground].sum() - tp
    fn = cm[foreground, :].sum() - tp
    tn = cm.sum() - tp - fp - fn
    iou = tp / max(tp + fp + fn, 1e-9)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
    precision = tp / max(tp + fp, 1e-9)
    recall = tp / max(tp + fn, 1e-9)
    oa = (tp + tn) / max(cm.sum(), 1e-9)
    return {"IoU": float(iou), "F1": float(f1), "Precision": float(precision),
            "Recall": float(recall), "OA": float(oa), "TP": int(tp), "FP": int(fp), "FN": int(fn)}


def recall_precision_at_mask(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """变化检测常用的漏检率 / 误检率。"""
    m = iou_f1(pred, gt, 2, 1)
    return {"漏检率": 1.0 - m["Recall"], "误检率": 1.0 - m["Precision"],
            "F1": m["F1"], "IoU": m["IoU"]}


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(255.0 ** 2 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
        a = np.asarray(a)
        b = np.asarray(b)
        return float(structural_similarity(a, b, channel_axis=-1 if a.ndim == 3 else None,
                                           data_range=255))
    except Exception:
        a = np.asarray(a, np.float32).mean(axis=-1) if np.asarray(a).ndim == 3 else np.asarray(a, np.float32)
        b = np.asarray(b, np.float32).mean(axis=-1) if np.asarray(b).ndim == 3 else np.asarray(b, np.float32)
        mu_a, mu_b = a.mean(), b.mean()
        va, vb = a.var(), b.var()
        cov = ((a - mu_a) * (b - mu_b)).mean()
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                     ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))
