# -*- coding: utf-8 -*-
"""二维地图视图：GeoTIFF 底图 + 解译结果图层叠加 + 缩放平移 + 图斑点击联动。

图层用 numpy RGBA 合成一次缓存成 QImage，重绘时只做裁剪缩放，因此大图也能流畅拖拽。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import QWidget

from ..utils.vis import CLASS_COLORS


class MapView(QWidget):
    """中间的地图视图。"""

    clicked = pyqtSignal(float, float)            # 图像像素坐标 (x, y)
    status = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.base: Optional[np.ndarray] = None
        self.layers: Dict[str, dict] = {}          # name -> {rgba, visible, alpha, order}
        self.boxes: List[dict] = []                # 矢量框 [{pts, label, color}]
        self.polygons: List[dict] = []             # 矢量图斑 [{ring(lon/lat->px), label, color}]
        self.selected: Optional[dict] = None
        self.scale = 1.0
        self.offset = QPointF(0, 0)
        self._drag_from: Optional[QPoint] = None
        self._cache: Optional[QImage] = None
        self._dirty = True
        self.pixel_size_m = (1.0, 1.0)

    # ------------------------------------------------------------------ 数据
    def set_base(self, rgb: np.ndarray, pixel_size_m=(1.0, 1.0)) -> None:
        self.base = np.ascontiguousarray(rgb)
        self.pixel_size_m = pixel_size_m
        self.scale = 1.0
        self.offset = QPointF(0, 0)
        self._fit_to_view()
        self._dirty = True
        self.update()

    def set_layer(self, name: str, rgba: Optional[np.ndarray], visible: bool = True,
                  alpha: float = 1.0, order: int = 0) -> None:
        if rgba is None:
            self.layers.pop(name, None)
        else:
            self.layers[name] = {"rgba": np.ascontiguousarray(rgba), "visible": visible,
                                 "alpha": alpha, "order": order}
        self._dirty = True
        self.update()

    def toggle_layer(self, name: str, visible: bool) -> None:
        if name in self.layers:
            self.layers[name]["visible"] = visible
            self._dirty = True
            self.update()

    def set_layer_alpha(self, name: str, alpha: float) -> None:
        if name in self.layers:
            self.layers[name]["alpha"] = float(alpha)
            self._dirty = True
            self.update()

    def clear_layers(self) -> None:
        self.layers.clear()
        self.boxes.clear()
        self.polygons.clear()
        self.selected = None
        self._dirty = True
        self.update()

    def set_boxes(self, boxes: List[dict]) -> None:
        self.boxes = list(boxes or [])
        self.update()

    def set_polygons(self, polys: List[dict]) -> None:
        self.polygons = list(polys or [])
        self.update()

    def select(self, item: Optional[dict]) -> None:
        self.selected = item
        self.update()

    # ------------------------------------------------------------------ 合成
    def _composite(self) -> QImage:
        h, w = self.base.shape[:2]
        buf = self.base.astype(np.float32).copy()
        for name in sorted(self.layers, key=lambda k: self.layers[k]["order"]):
            lay = self.layers[name]
            if not lay["visible"]:
                continue
            rgba = lay["rgba"]
            if rgba.shape[:2] != (h, w):
                continue
            a = (rgba[:, :, 3:4].astype(np.float32) / 255.0) * float(lay["alpha"])
            buf = buf * (1 - a) + rgba[:, :, :3].astype(np.float32) * a
        img = np.clip(buf, 0, 255).astype(np.uint8)
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
        self._cache = qimg.copy()
        self._dirty = False
        return self._cache

    # ------------------------------------------------------------------ 视图
    def _fit_to_view(self) -> None:
        if self.base is None:
            return
        h, w = self.base.shape[:2]
        s = min(self.width() / w, self.height() / h) * 0.98
        self.scale = max(s, 1e-6)
        self.offset = QPointF((self.width() - w * self.scale) / 2,
                              (self.height() - h * self.scale) / 2)

    def _to_view(self, x: float, y: float) -> QPointF:
        return QPointF(self.offset.x() + x * self.scale, self.offset.y() + y * self.scale)

    def _to_image(self, pos) -> Tuple[float, float]:
        return ((pos.x() - self.offset.x()) / self.scale,
                (pos.y() - self.offset.y()) / self.scale)

    def zoom_by(self, factor: float, center: Optional[QPointF] = None) -> None:
        if self.base is None:
            return
        c = center or QPointF(self.width() / 2, self.height() / 2)
        ix, iy = self._to_image(c)
        self.scale = float(np.clip(self.scale * factor, 0.02, 40.0))
        self.offset = QPointF(c.x() - ix * self.scale, c.y() - iy * self.scale)
        self.update()

    def fit(self) -> None:
        self._fit_to_view()
        self.update()

    def zoom_to_pixel(self, x: float, y: float, scale: float = 4.0) -> None:
        self.scale = float(np.clip(scale, 0.02, 40.0))
        self.offset = QPointF(self.width() / 2 - x * self.scale,
                              self.height() / 2 - y * self.scale)
        self.update()

    doubleClicked = pyqtSignal(float, float)

    # ------------------------------------------------------------------ 事件
    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(28, 31, 36))
        if self.base is None:
            p.setPen(QColor(150, 160, 172))
            p.setFont(QFont("Microsoft YaHei", 11))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "尚未加载影像\n\n文件 → 打开多光谱遥感影像（GeoTIFF）\n或 文件 → 生成演示数据集")
            return
        if self._dirty or self._cache is None:
            self._composite()
        src, dst = self._visible_rects()
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.drawImage(dst, self._cache, src)

        # 矢量叠加：检测框与图斑
        p.setRenderHint(QPainter.Antialiasing, True)
        for poly in self.polygons:
            pts = [self._to_view(x, y) for x, y in poly["pts"]]
            if len(pts) < 3:
                continue
            color = QColor(*poly.get("color", (255, 60, 60)))
            sel = self.selected is not None and self.selected.get("id") == poly.get("id")
            pen = QPen(color, 2.6 if sel else 1.6)
            p.setPen(pen)
            fill = QColor(color)
            fill.setAlpha(90 if sel else 40)
            p.setBrush(fill)
            p.drawPolygon(QPolygonF(pts))
        p.setBrush(Qt.NoBrush)
        for bx in self.boxes:
            pts = [self._to_view(x, y) for x, y in bx["pts"]]
            color = QColor(*bx.get("color", (0, 255, 0)))
            p.setPen(QPen(color, 1.6))
            p.drawPolygon(QPolygonF(pts))
            if bx.get("label") and self.scale > 0.08:
                p.setPen(QPen(color))
                p.setFont(QFont("Consolas", 8))
                p.drawText(pts[0] + QPointF(0, -3), bx["label"])
        p.setPen(QColor(120, 132, 148))
        p.setFont(QFont("Consolas", 9))
        p.drawText(8, self.height() - 8,
                   f"缩放 {self.scale * 100:.0f}%   |   地面分辨率 "
                   f"{self.pixel_size_m[0] * 100:.1f} cm/px   |   滚轮缩放 / 拖动平移 / 双击放大")

    def _visible_rects(self):
        h, w = self.base.shape[:2]
        x0 = max(0.0, -self.offset.x() / self.scale)
        y0 = max(0.0, -self.offset.y() / self.scale)
        x1 = min(float(w), (self.width() - self.offset.x()) / self.scale)
        y1 = min(float(h), (self.height() - self.offset.y()) / self.scale)
        src = QRectF(x0, y0, max(x1 - x0, 1), max(y1 - y0, 1))
        dst = QRectF(self.offset.x() + x0 * self.scale, self.offset.y() + y0 * self.scale,
                     src.width() * self.scale, src.height() * self.scale)
        return src, dst

    def wheelEvent(self, ev):
        if self.base is None:
            return
        self.zoom_by(1.18 ** (ev.angleDelta().y() / 120.0), QPointF(ev.pos()))

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._drag_from = ev.pos()
            self._press_pos = ev.pos()

    def mouseMoveEvent(self, ev):
        if self.base is None:
            return
        if self._drag_from is not None and (ev.buttons() & Qt.LeftButton):
            d = ev.pos() - self._drag_from
            self.offset += QPointF(d)
            self._drag_from = ev.pos()
            self.update()
        else:
            x, y = self._to_image(ev.pos())
            h, w = self.base.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                mx, my = self.pixel_size_m
                self.status.emit(f"像素 ({int(x)}, {int(y)})　≈　"
                                 f"{x * mx:.1f} m , {y * my:.1f} m")

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton and self._drag_from is not None:
            moved = (ev.pos() - getattr(self, "_press_pos", self._drag_from)).manhattanLength()
            self._drag_from = None
            if moved < 4:
                x, y = self._to_image(ev.pos())
                self.clicked.emit(x, y)

    def mouseDoubleClickEvent(self, ev):
        if self.base is None:
            return
        x, y = self._to_image(ev.pos())
        self.zoom_by(1.8, QPointF(ev.pos()))
        self.doubleClicked.emit(x, y)

    def resizeEvent(self, ev):
        if self.base is not None and self.scale == 1.0 and self.offset == QPointF(0, 0):
            self._fit_to_view()

    # ------------------------------------------------------------------ 图层色板
    @staticmethod
    def mask_to_rgba(mask: np.ndarray, alpha: int = 140) -> np.ndarray:
        """类别掩膜 -> RGBA（用于叠加显示）。"""
        m = np.asarray(mask)
        h, w = m.shape[:2]
        out = np.zeros((h, w, 4), np.uint8)
        if m.max() <= 1:
            out[m > 0] = (255, 60, 60, alpha)
        else:
            for i, c in enumerate(CLASS_COLORS):
                out[m == i] = (c[0], c[1], c[2], 0 if i == 0 else alpha)
        return out

    @staticmethod
    def prob_to_rgba(prob: np.ndarray, color=(255, 255, 0), alpha: int = 120) -> np.ndarray:
        p = np.clip(np.asarray(prob, np.float32), 0, 1)
        h, w = p.shape[:2]
        out = np.zeros((h, w, 4), np.uint8)
        out[:, :, 0], out[:, :, 1], out[:, :, 2] = color
        out[:, :, 3] = (p * alpha).astype(np.uint8)
        return out
