# -*- coding: utf-8 -*-
"""针孔相机模型与轨道相机（供 3DGS 重建、渲染与前端交互使用）。

约定：OpenCV 相机坐标系 —— x 向右、y 向下、z 向前（朝向场景）。
世界 -> 相机：p_cam = R @ p_world + t
投影：       u = fx * x / z + cx ,  v = fy * y / z + cy
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class Camera:
    """一帧针孔相机。R/t 为世界->相机的外参。"""

    R: np.ndarray                      # (3,3)
    t: np.ndarray                      # (3,)
    width: int
    height: int
    fx: float
    fy: float
    cx: Optional[float] = None
    cy: Optional[float] = None
    znear: float = 0.01
    zfar: float = 1e4
    image_path: str = ""

    def __post_init__(self):
        self.R = np.asarray(self.R, np.float64).reshape(3, 3)
        self.t = np.asarray(self.t, np.float64).reshape(3)
        if self.cx is None:
            self.cx = self.width / 2.0
        if self.cy is None:
            self.cy = self.height / 2.0

    # ---------------------------------------------------------------- 属性
    @property
    def center(self) -> np.ndarray:
        """相机光心在世界坐标中的位置。"""
        return -self.R.T @ self.t

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], np.float64)

    def fov_x(self) -> float:
        return 2 * math.atan(self.width / (2 * self.fx))

    def fov_y(self) -> float:
        return 2 * math.atan(self.height / (2 * self.fy))

    def world_to_cam(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, np.float64).reshape(-1, 3)
        return pts @ self.R.T + self.t

    def project(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """世界点 -> 像素坐标 + 深度。"""
        pc = self.world_to_cam(pts)
        z = np.clip(pc[:, 2], 1e-6, None)
        u = self.fx * pc[:, 0] / z + self.cx
        v = self.fy * pc[:, 1] / z + self.cy
        return np.stack([u, v], axis=1), pc[:, 2]

    # ---------------------------------------------------------------- 构造
    @classmethod
    def look_at(cls, eye, target, up=(0, -1, 0), width=640, height=480,
                fov_y_deg: float = 45.0, image_path: str = "") -> "Camera":
        """由视点与目标点构造相机（y 轴向下）。"""
        eye = np.asarray(eye, np.float64)
        target = np.asarray(target, np.float64)
        up = np.asarray(up, np.float64)
        z = target - eye
        z = z / (np.linalg.norm(z) + 1e-12)
        x = np.cross(up, z)
        if np.linalg.norm(x) < 1e-8:
            x = np.array([1.0, 0, 0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z], axis=0)            # world->cam 的旋转
        t = -R @ eye
        fy = (height / 2.0) / math.tan(math.radians(fov_y_deg) / 2.0)
        return cls(R, t, width, height, fy, fy, image_path=image_path)


@dataclass
class OrbitCamera:
    """给前端用的轨道相机：拖拽旋转、滚轮缩放、右键平移。"""

    target: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float64))
    distance: float = 6.0
    yaw: float = 45.0          # 度
    pitch: float = 25.0        # 度
    fov_y_deg: float = 45.0
    up_sign: float = -1.0      # 影像坐标系 y 向下

    def eye(self) -> np.ndarray:
        cy, sy = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        cp, sp = math.cos(math.radians(self.pitch)), math.sin(math.radians(self.pitch))
        d = np.array([cp * cy, cp * sy, sp], np.float64)
        return np.asarray(self.target, np.float64) + d * self.distance

    def to_camera(self, width: int, height: int) -> Camera:
        return Camera.look_at(self.eye(), self.target, up=(0, 0, 1) if self.up_sign > 0 else (-0, 0, -1),
                              width=width, height=height, fov_y_deg=self.fov_y_deg)

    # ---------------------------------------------------------------- 交互
    def rotate(self, dx: float, dy: float, sensitivity: float = 0.4) -> None:
        self.yaw = (self.yaw - dx * sensitivity) % 360.0
        self.pitch = float(np.clip(self.pitch + dy * sensitivity, -89.0, 89.0))

    def zoom(self, steps: float) -> None:
        self.distance = float(np.clip(self.distance * (0.9 ** steps), 1e-3, 1e6))

    def pan(self, dx: float, dy: float, scale: float = 1.0) -> None:
        cam = self.to_camera(1, 1)
        right = cam.R.T @ np.array([1.0, 0, 0])
        down = cam.R.T @ np.array([0, 1.0, 0])
        k = self.distance * 0.0015 * scale
        self.target = np.asarray(self.target, np.float64) - right * dx * k - down * dy * k


def intrinsics_from_exif(width: int, height: int, focal_mm: float = 24.0,
                         sensor_mm: float = 36.0) -> Tuple[float, float]:
    """没有内参时用等效焦距估算。"""
    f = focal_mm / sensor_mm * max(width, height)
    return f, f
