# -*- coding: utf-8 -*-
"""合成多视角数据集：没有无人机照片 / 没跑 COLMAP 时，也能完整跑通 3D 高斯泼溅。

用光线投射渲染一个程序化生成的小城市场景（地面 + 若干建筑体块），
输出 N 个环绕视角的影像 + 相机位姿 + 稀疏点云（COLMAP text 格式），
因此 3DGS 训练、渲染、地理配准、前端三维浏览全流程都能离线演示。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ..utils.logger import get_logger
from .camera import Camera
from .colmap_io import (ColmapCamera, ColmapImage, ColmapReconstruction,
                        write_reconstruction)

LOG = get_logger("gsplat.synthetic")


@dataclass
class Box:
    lo: np.ndarray
    hi: np.ndarray
    color: np.ndarray
    windows: bool = True

    @property
    def center(self) -> np.ndarray:
        return (self.lo + self.hi) / 2


@dataclass
class Scene:
    boxes: List[Box] = field(default_factory=list)
    ground_color: np.ndarray = field(default_factory=lambda: np.array([0.42, 0.45, 0.40]))
    bounds: Tuple[float, float, float, float] = (-14.0, -14.0, 14.0, 14.0)


def make_city_scene(seed: int = 7, n_buildings: int = 11) -> Scene:
    rng = np.random.default_rng(seed)
    scene = Scene()
    palette = [np.array([0.78, 0.74, 0.68]), np.array([0.70, 0.52, 0.44]),
               np.array([0.62, 0.66, 0.72]), np.array([0.80, 0.78, 0.74])]
    placed: List[Tuple[float, float, float, float]] = []
    tries = 0
    while len(scene.boxes) < n_buildings and tries < 400:
        tries += 1
        w = float(rng.uniform(2.0, 4.0))
        d = float(rng.uniform(2.0, 4.0))
        h = float(rng.uniform(1.8, 6.5))
        cx = float(rng.uniform(-10, 10))
        cy = float(rng.uniform(-10, 10))
        if any(abs(cx - px) < (w + pw) / 2 + 1.2 and abs(cy - py) < (d + pd) / 2 + 1.2
               for px, py, pw, pd in placed):
            continue
        placed.append((cx, cy, w, d))
        scene.boxes.append(Box(np.array([cx - w / 2, cy - d / 2, 0.0]),
                               np.array([cx + w / 2, cy + d / 2, h]),
                               palette[len(scene.boxes) % len(palette)]))
    return scene


# --------------------------------------------------------------------------- 渲染
def _ray_box(o: np.ndarray, d: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    inv = 1.0 / np.where(np.abs(d) < 1e-9, 1e-9, d)
    t1 = (lo[None, :] - o) * inv
    t2 = (hi[None, :] - o) * inv
    tmin = np.max(np.minimum(t1, t2), axis=1)
    tmax = np.min(np.maximum(t1, t2), axis=1)
    hit = (tmax >= np.maximum(tmin, 0.0)) & (tmax > 0)
    t = np.where(tmin > 1e-6, tmin, tmax)
    return hit, np.where(hit, t, np.inf)


def render_scene(scene: Scene, cam: Camera, light_dir=(0.4, 0.5, 0.75)) -> np.ndarray:
    """用光线投射渲染一帧（含简单漫反射与程序化窗户纹理）。"""
    H, W = int(cam.height), int(cam.width)
    xs = (np.arange(W, dtype=np.float64) + 0.5 - cam.cx) / cam.fx
    ys = (np.arange(H, dtype=np.float64) + 0.5 - cam.cy) / cam.fy
    gx, gy = np.meshgrid(xs, ys)
    dirs_cam = np.stack([gx.ravel(), gy.ravel(), np.ones(H * W)], axis=1)
    dirs_world = dirs_cam @ cam.R            # R^T @ v  ==  v @ R
    dirs_world /= np.linalg.norm(dirs_world, axis=1, keepdims=True)
    origin = np.tile(cam.center, (H * W, 1))

    best_t = np.full(H * W, np.inf)
    best_n = np.zeros((H * W, 3))
    best_c = np.tile(scene.ground_color, (H * W, 1))
    world_pts = np.zeros((H * W, 3))

    # 地面 z = 0
    dz = dirs_world[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ground = -origin[:, 2] / np.where(np.abs(dz) < 1e-9, 1e-9, dz)
    ok = (t_ground > 1e-6) & (dz < 0)
    if ok.any():
        p = origin[ok] + dirs_world[ok] * t_ground[ok, None]
        inside = ((p[:, 0] > scene.bounds[0]) & (p[:, 0] < scene.bounds[2]) &
                  (p[:, 1] > scene.bounds[1]) & (p[:, 1] < scene.bounds[3]))
        idx = np.nonzero(ok)[0][inside]
        best_t[idx] = t_ground[idx]
        best_n[idx] = np.array([0, 0, 1.0])
        # 网格花纹，给地面一点纹理
        gx_i = np.floor(p[inside, 0] / 2.0)
        gy_i = np.floor(p[inside, 1] / 2.0)
        checker = ((gx_i + gy_i) % 2 == 0).astype(np.float64)
        base = scene.ground_color[None, :] * (0.86 + 0.22 * checker[:, None])
        best_c[idx] = base

    # 建筑体块
    for box in scene.boxes:
        hit, t = _ray_box(origin, dirs_world, box.lo, box.hi)
        upd = hit & (t < best_t)
        if not upd.any():
            continue
        idx = np.nonzero(upd)[0]
        best_t[idx] = t[idx]
        p = origin[idx] + dirs_world[idx] * t[idx, None]
        # 判断命中的是哪个面 -> 法向 + 窗户纹理
        rel = (p - box.lo[None, :]) / np.maximum(box.hi - box.lo, 1e-6)[None, :]
        face = np.argmin(np.stack([rel[:, 0], rel[:, 1], rel[:, 2],
                                   1 - rel[:, 0], 1 - rel[:, 1], 1 - rel[:, 2]], 1), axis=1)
        normal = np.zeros((idx.size, 3))
        normal[face == 0] = (-1, 0, 0)
        normal[face == 1] = (0, -1, 0)
        normal[face == 2] = (0, 0, -1)
        normal[face == 3] = (1, 0, 0)
        normal[face == 4] = (0, 1, 0)
        normal[face == 5] = (0, 0, 1)
        best_n[idx] = normal
        col = np.tile(box.color, (idx.size, 1))
        wall = face < 5
        if box.windows and wall.any():
            wid = idx[wall]
            pw = p[wall]
            u = np.where(face[wall] < 2, pw[:, 1], pw[:, 0])
            v = pw[:, 2]
            win = ((np.floor(u / 0.9) % 2 == 0) & (np.floor(v / 1.1) % 2 == 0) & (v > 0.6))
            col[wall][win] = np.array([0.22, 0.28, 0.36])
        best_c[idx] = col

    # 着色：环境光 + 漫反射
    ld = np.array(light_dir, np.float64)
    ld /= np.linalg.norm(ld)
    lam = np.clip(best_n @ ld, 0, 1)
    shade = (0.42 + 0.62 * lam)[:, None]
    rgb = np.clip(best_c * shade, 0, 1).reshape(H, W, 3)
    # 远处加一点雾，模拟航拍大气影响
    fog = np.clip((best_t.reshape(H, W) - 25.0) / 40.0, 0, 0.35)
    rgb = rgb * (1 - fog[:, :, None]) + np.array([0.72, 0.78, 0.86]) * fog[:, :, None]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


# --------------------------------------------------------------------------- 数据集
def make_dataset(out_dir: str | Path, n_views: int = 24, width: int = 320, height: int = 240,
                 n_points: int = 6000, seed: int = 7, scene: Optional[Scene] = None,
                 radius: float = 26.0, elevation_deg: float = 24.0,
                 origin_lonlat=(126.6320, 45.7520), gsd_m: float = 0.08) -> dict:
    """生成一套完整的演示数据：images/ + sparse/0/ + 真值位姿。

    返回 dict：{root, images_dir, sparse_dir, n_views, center, radius, origin_lonlat}
    """
    import cv2
    root = Path(out_dir)
    images_dir = root / "images"
    sparse_dir = root / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    scene = scene or make_city_scene(seed=seed)
    rng = np.random.default_rng(seed)

    centers = [b.center for b in scene.boxes] or [np.zeros(3)]
    target = np.mean(centers, axis=0) * np.array([1, 1, 0.35])
    rec = ColmapReconstruction()
    cam_id = 1
    rec.cameras[cam_id] = ColmapCamera(cam_id, "PINHOLE", width, height,
                                       np.array([width * 1.15, width * 1.15,
                                                 width / 2, height / 2]))

    for i in range(n_views):
        ang = 2 * math.pi * i / n_views
        r = radius * float(rng.uniform(0.94, 1.06))
        elev = math.radians(elevation_deg + float(rng.uniform(-4, 4)))
        eye = np.array([r * math.cos(ang) * math.cos(elev),
                        r * math.sin(ang) * math.cos(elev),
                        r * math.sin(elev) + 3.5])
        cam = Camera.look_at(eye, target + rng.uniform(-1.2, 1.2, 3),
                             up=(0, 0, -1), width=width, height=height, fov_y_deg=42.0)
        img = render_scene(scene, cam)
        name = f"view_{i:03d}.png"
        cv2.imwrite(str(images_dir / name), img[:, :, ::-1])
        from .model import _rotmat_to_quat
        q = _rotmat_to_quat(cam.R)
        rec.images[i + 1] = ColmapImage(i + 1, q, cam.t, cam_id, name)

    # 稀疏点云：在可见表面上采样
    pts, cols = [], []
    for _ in range(n_points):
        if scene.boxes and rng.random() < 0.72:
            b = scene.boxes[int(rng.integers(0, len(scene.boxes)))]
            face = int(rng.integers(0, 6))
            p = rng.uniform(b.lo, b.hi)
            axis = face // 2
            p[axis] = b.hi[axis] if face % 2 else b.lo[axis]
            c = b.color * 255
        else:
            p = np.array([rng.uniform(scene.bounds[0], scene.bounds[2]),
                          rng.uniform(scene.bounds[1], scene.bounds[3]), 0.0])
            c = scene.ground_color * 255
        pts.append(p)
        cols.append(c + rng.normal(0, 8, 3))
    rec.points3D = np.array(pts, np.float64)
    rec.colors = np.clip(np.array(cols), 0, 255)

    write_reconstruction(sparse_dir, rec)
    LOG.info("合成演示数据：%d 视角 %dx%d -> %s", n_views, width, height, root)

    # 模型中心在世界系中的"地理原点"（供地理配准用）
    return {"root": str(root), "images_dir": str(images_dir), "sparse_dir": str(sparse_dir),
            "n_views": n_views, "width": width, "height": height,
            "center": np.asarray(target).tolist(), "radius": float(radius),
            "origin_lonlat": list(origin_lonlat), "gsd_m": float(gsd_m)}


def scene_surface_area(scene: Scene) -> float:
    a = (scene.bounds[2] - scene.bounds[0]) * (scene.bounds[3] - scene.bounds[1])
    for b in scene.boxes:
        s = b.hi - b.lo
        a += 2 * (s[0] * s[1] + s[1] * s[2] + s[0] * s[2])
    return float(a)
