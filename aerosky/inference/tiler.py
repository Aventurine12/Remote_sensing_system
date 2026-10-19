# -*- coding: utf-8 -*-
"""大图分块推理引擎：「分块切割 — 并行推理 — 无缝拼接」。

整幅遥感影像动辄上万像素，直接整图推理会爆显存。这里把影像按固定窗口 + 重叠切分，
分批送入模型推理，再用余弦权重把各窗口结果无缝拼回原图：

    掩膜类任务：  P(x) = Σ w_i(x) * p_i(x) / Σ w_i(x)
    检测类任务：  各窗口检测框平移到全局坐标后做类别内 NMS 去重

窗口内边缘的预测往往不如下采样中心可靠，因此余弦窗让中心权重为 1、边缘趋近 0，
重叠区由多个窗口的结果平滑过渡，拼接处不会出现明显接缝。
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .base import Detection


def blend_window(h: int, w: int, kind: str = "cosine", eps: float = 1e-3) -> np.ndarray:
    """生成单窗口的融合权重图（中心高、边缘低）。"""
    if kind == "none":
        return np.ones((h, w), np.float32)
    wy = np.hanning(max(h, 4))[:h] if h >= 4 else np.ones(h)
    wx = np.hanning(max(w, 4))[:w] if w >= 4 else np.ones(w)
    win = np.outer(wy, wx).astype(np.float32)
    win = np.clip(win, eps, None)
    return win


def _windows_1d(length: int, tile: int, overlap: int) -> List[Tuple[int, int]]:
    if length <= tile:
        return [(0, length)]
    step = max(1, tile - overlap)
    starts = list(range(0, max(1, length - tile) + 1, step))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return [(s, s + tile) for s in starts]


class Tiler:
    """滑动窗口切图 + 结果拼接。"""

    def __init__(self, tile_size: int = 1024, overlap: int = 128,
                 batch_size: int = 2, blend: str = "cosine", max_workers: int = 2):
        self.tile_size = int(tile_size)
        self.overlap = int(overlap)
        self.batch_size = max(1, int(batch_size))
        self.blend = blend
        self.max_workers = max(1, int(max_workers))

    # ------------------------------------------------------------------ 窗口
    def windows(self, h: int, w: int) -> List[Tuple[int, int, int, int]]:
        ys = _windows_1d(h, min(self.tile_size, h), self.overlap)
        xs = _windows_1d(w, min(self.tile_size, w), self.overlap)
        return [(x0, y0, x1, y1) for y0, y1 in ys for x0, x1 in xs]

    # ------------------------------------------------------------------ 掩膜
    def run_mask(self, image: np.ndarray, predict: Callable[[np.ndarray], np.ndarray],
                 out_channels: int = 1, pad_to: Optional[Tuple[int, int]] = None,
                 progress: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
        """逐窗推理并加权拼接，返回 (H, W, out_channels) 的概率图。

        predict: 接受 (N,h,w,C) uint8，返回 (N,out_channels,h,w) float 的数组。
        """
        h, w = image.shape[:2]
        wins = self.windows(h, w)
        acc = np.zeros((h, w, out_channels), np.float32)
        wsum = np.zeros((h, w, 1), np.float32)
        done = 0
        for i in range(0, len(wins), self.batch_size):
            batch_wins = wins[i:i + self.batch_size]
            tiles = np.stack([_crop(image, win) for win in batch_wins])
            preds = np.asarray(predict(tiles), dtype=np.float32)
            if preds.ndim == 3:
                preds = preds[:, None]
            for j, (x0, y0, x1, y1) in enumerate(batch_wins):
                tw = blend_window(y1 - y0, x1 - x0, self.blend)[:, :, None]
                acc[y0:y1, x0:x1] += preds[j].transpose(1, 2, 0) * tw
                wsum[y0:y1, x0:x1] += tw
            done += len(batch_wins)
            if progress:
                progress(min(done, len(wins)), len(wins))
        np.divide(acc, np.maximum(wsum, 1e-6), out=acc)
        return acc

    # ------------------------------------------------------------------ 检测
    def run_detection(self, image: np.ndarray,
                      predict: Callable[[np.ndarray], List[List[Detection]]],
                      nms_iou: float = 0.5,
                      progress: Optional[Callable[[int, int], None]] = None) -> List[Detection]:
        """逐窗检测，平移到全局坐标后做类别内 NMS。"""
        h, w = image.shape[:2]
        wins = self.windows(h, w)
        all_dets: List[Detection] = []
        done = 0
        for i in range(0, len(wins), self.batch_size):
            batch_wins = wins[i:i + self.batch_size]
            tiles = np.stack([_crop(image, win) for win in batch_wins])
            per_tile = predict(tiles)
            for j, (x0, y0, _, _) in enumerate(batch_wins):
                for d in per_tile[j]:
                    all_dets.append(d.offset(x0, y0))
            done += len(batch_wins)
            if progress:
                progress(min(done, len(wins)), len(wins))
        return nms(all_dets, iou_thr=nms_iou)

    # ------------------------------------------------------------------ 并行（CPU 场景）
    def run_mask_threaded(self, image, predict, out_channels=1, workers=None):
        """把窗口分给多个线程（适合 CPU 推理/传统算法）。"""
        workers = workers or self.max_workers
        h, w = image.shape[:2]
        wins = self.windows(h, w)
        chunk = max(1, math.ceil(len(wins) / workers))

        def job(sub_wins):
            acc = np.zeros((h, w, out_channels), np.float32)
            wsum = np.zeros((h, w, 1), np.float32)
            for i in range(0, len(sub_wins), self.batch_size):
                bw = sub_wins[i:i + self.batch_size]
                tiles = np.stack([_crop(image, win) for win in bw])
                preds = np.asarray(predict(tiles), np.float32)
                if preds.ndim == 3:
                    preds = preds[:, None]
                for j, (x0, y0, x1, y1) in enumerate(bw):
                    tw = blend_window(y1 - y0, x1 - x0, self.blend)[:, :, None]
                    acc[y0:y1, x0:x1] += preds[j].transpose(1, 2, 0) * tw
                    wsum[y0:y1, x0:x1] += tw
            return acc, wsum

        with ThreadPoolExecutor(max_workers=workers) as ex:
            parts = list(ex.map(job, [wins[i:i + chunk] for i in range(0, len(wins), chunk)]))
        acc = sum(p[0] for p in parts)
        wsum = sum(p[1] for p in parts)
        return acc / np.maximum(wsum, 1e-6)


# --------------------------------------------------------------------------- 工具
def _crop(image: np.ndarray, win: Sequence[int]) -> np.ndarray:
    x0, y0, x1, y1 = win
    return image[y0:y1, x0:x1]


def nms(dets: List[Detection], iou_thr: float = 0.5) -> List[Detection]:
    """类别内贪心 NMS（OBB 用外接矩形近似），并把重合框做加权融合。"""
    if not dets:
        return []
    keep: List[Detection] = []
    for cls in sorted({d.label_id for d in dets}):
        group = [d for d in dets if d.label_id == cls]
        group.sort(key=lambda d: -d.score)
        used = [False] * len(group)
        for i, a in enumerate(group):
            if used[i]:
                continue
            cluster = [a]
            used[i] = True
            ai = a.xyxy()
            for j in range(i + 1, len(group)):
                if used[j]:
                    continue
                if _iou(ai, group[j].xyxy()) > iou_thr:
                    used[j] = True
                    cluster.append(group[j])
            keep.append(_fuse(cluster))
    keep.sort(key=lambda d: -d.score)
    return keep


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def _fuse(cluster: List[Detection]) -> Detection:
    if len(cluster) == 1:
        return cluster[0]
    w = np.array([max(d.score, 1e-6) for d in cluster], np.float32)
    w = w / w.sum()
    boxes = np.stack([d.box.astype(np.float32) for d in cluster])
    box = (boxes * w[:, None]).sum(axis=0) if boxes.ndim == 2 and boxes.shape[1] == 4 else cluster[0].box
    if cluster[0].obb and boxes.shape[1] == 8:
        box = (boxes * w[:, None]).sum(axis=0)
    best = cluster[int(np.argmax([d.score for d in cluster]))]
    return Detection(np.asarray(box), float(min(1.0, best.score + 0.05 * (len(cluster) - 1))),
                     best.label, best.label_id, best.obb, best.tile_origin)
