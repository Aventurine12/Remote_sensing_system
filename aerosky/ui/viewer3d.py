# -*- coding: utf-8 -*-
"""三维高斯模型浏览视图。

不是把高斯当点云画，而是用 numpy 前向光栅化把三维高斯真正"泼溅"渲染成图像，
再交给 Qt 显示 —— 因此界面上看到的旋转/缩放效果与 3DGS 的实时渲染一致。
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
from PyQt5.QtCore import QPoint, QPointF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter
from PyQt5.QtWidgets import QWidget

from ..gsplat.camera import OrbitCamera
from ..gsplat.render_np import SplatRenderer


class SplatView(QWidget):
    """右侧的三维视图：拖拽旋转、滚轮缩放、右键平移。"""

    status = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 300)
        self.setMouseTracking(True)
        self.renderer: Optional[SplatRenderer] = None
        self.cam = OrbitCamera()
        self.render_scale = 0.55           # 渲染分辨率相对控件尺寸的比例
        self.bg = (16, 18, 22)
        self._last_pos: Optional[QPoint] = None
        self._image: Optional[QImage] = None
        self._dirty = True
        self._last_ms = 0.0
        self._highlight: Optional[np.ndarray] = None      # 高亮用的世界坐标点

    # ------------------------------------------------------------------ 数据
    def set_renderer(self, renderer: Optional[SplatRenderer], reset: bool = True) -> None:
        self.renderer = renderer
        if renderer is not None and reset:
            self.reset_view()
        self._dirty = True
        self.update()

    def reset_view(self) -> None:
        if self.renderer is None:
            return
        center, diag = self.renderer.center_scale()
        self.cam = OrbitCamera(target=np.asarray(center, np.float64),
                               distance=max(diag * 1.15, 1e-3), yaw=42.0, pitch=26.0,
                               fov_y_deg=45.0)
        self._dirty = True

    def set_highlight(self, points: Optional[np.ndarray]) -> None:
        self._highlight = None if points is None else np.asarray(points, np.float64).reshape(-1, 3)
        self.update()

    # ------------------------------------------------------------------ 渲染
    def _render(self, w: int, h: int) -> QImage:
        w = max(int(w * self.render_scale), 64)
        h = max(int(h * self.render_scale), 64)
        cam = self.cam.to_camera(w, h)
        t0 = time.time()
        rgb, _ = self.renderer.render(cam)
        self._last_ms = (time.time() - t0) * 1000.0
        rgb = np.ascontiguousarray(rgb)
        return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(*self.bg))
        if self.renderer is None:
            p.setPen(QColor(150, 160, 172))
            p.setFont(QFont("Microsoft YaHei", 10))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "尚未重建三维模型\n\n三维 → 生成演示数据并重建（离线可跑）\n"
                       "三维 → 用 COLMAP 结果重建（无人机影像）")
            return
        if self._dirty or self._image is None:
            self._image = self._render(self.width(), self.height())
            self._dirty = False
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.drawImage(self.rect(), self._image)

        # 高亮标记（例如被点击的异常区域在模型中的位置）
        if self._highlight is not None and self._highlight.size:
            cam = self.cam.to_camera(int(self.width() * self.render_scale),
                                     int(self.height() * self.render_scale))
            uv, z = cam.project(self._highlight)
            p.setRenderHint(QPainter.Antialiasing, True)
            for (u, v), zz in zip(uv, z):
                if zz <= 0:
                    continue
                x = u / self.render_scale
                y = v / self.render_scale
                p.setPen(QColor(255, 90, 60))
                p.drawEllipse(QPointF(x, y), 6, 6)

        p.setPen(QColor(130, 142, 158))
        p.setFont(QFont("Consolas", 9))
        n = self.renderer.n
        p.drawText(8, self.height() - 8,
                   f"高斯 {n:,}　|　{self._last_ms:.0f} ms/帧　|　"
                   f"距离 {self.cam.distance:.1f}　|　拖拽旋转 / 滚轮缩放 / 右键平移")

    # ------------------------------------------------------------------ 事件
    def mousePressEvent(self, ev):
        self._last_pos = ev.pos()

    def mouseMoveEvent(self, ev):
        if self._last_pos is None or self.renderer is None:
            return
        d = ev.pos() - self._last_pos
        self._last_pos = ev.pos()
        if ev.buttons() & Qt.LeftButton:
            self.cam.rotate(d.x(), d.y())
            self._dirty = True
            self.update()
        elif ev.buttons() & Qt.RightButton:
            self.cam.pan(d.x(), d.y())
            self._dirty = True
            self.update()

    def mouseReleaseEvent(self, ev):
        self._last_pos = None

    def wheelEvent(self, ev):
        if self.renderer is None:
            return
        self.cam.zoom(ev.angleDelta().y() / 120.0)
        self._dirty = True
        self.update()

    def resizeEvent(self, ev):
        self._dirty = True
