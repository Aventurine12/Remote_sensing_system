# -*- coding: utf-8 -*-
"""COLMAP 稀疏重建结果读写（text / binary 两种格式都支持）。

真实作业流程是：无人机多视角影像 -> COLMAP（运动恢复结构 SfM）解算相机位姿与稀疏点云
-> 本模块读进来 -> 交给 3D 高斯泼溅做重建与优化。
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.logger import get_logger

LOG = get_logger("gsplat.colmap")

# COLMAP 相机模型 -> 参数含义
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", ["f", "cx", "cy"]),
    1: ("PINHOLE", ["fx", "fy", "cx", "cy"]),
    2: ("SIMPLE_RADIAL", ["f", "cx", "cy", "k"]),
    3: ("RADIAL", ["f", "cx", "cy", "k1", "k2"]),
    4: ("OPENCV", ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"]),
    5: ("OPENCV_FISHEYE", ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"]),
}


@dataclass
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    @property
    def fx(self) -> float:
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            return float(self.params[0])
        return float(self.params[0])

    @property
    def fy(self) -> float:
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            return float(self.params[0])
        return float(self.params[1])

    @property
    def cx(self) -> float:
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            return float(self.params[1])
        return float(self.params[2])

    @property
    def cy(self) -> float:
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            return float(self.params[2])
        return float(self.params[3])


@dataclass
class ColmapImage:
    image_id: int
    qvec: np.ndarray          # (4,) [w,x,y,z] 世界->相机的旋转
    tvec: np.ndarray          # (3,)
    camera_id: int
    name: str
    xys: Optional[np.ndarray] = None      # (N,2) 观测到的特征点像素坐标
    point3D_ids: Optional[np.ndarray] = None

    def R(self) -> np.ndarray:
        return qvec2rotmat(self.qvec)

    def center(self) -> np.ndarray:
        return -self.R().T @ self.tvec


@dataclass
class ColmapReconstruction:
    cameras: Dict[int, ColmapCamera] = field(default_factory=dict)
    images: Dict[int, ColmapImage] = field(default_factory=dict)
    points3D: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    def named(self) -> List[Tuple[ColmapImage, ColmapCamera]]:
        out = []
        for img in sorted(self.images.values(), key=lambda i: i.name):
            cam = self.cameras.get(img.camera_id)
            if cam is not None:
                out.append((img, cam))
        return out

    def summary(self) -> dict:
        centers = np.stack([i.center() for i in self.images.values()]) if self.images else np.zeros((0, 3))
        return {
            "num_cameras": len(self.cameras), "num_images": len(self.images),
            "num_points3D": int(self.points3D.shape[0]),
            "bbox_min": centers.min(0).round(3).tolist() if len(centers) else None,
            "bbox_max": centers.max(0).round(3).tolist() if len(centers) else None,
        }


def qvec2rotmat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64).reshape(4)
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ], np.float64)


# --------------------------------------------------------------------------- 读取
def read_reconstruction(sparse_dir: str | Path) -> ColmapReconstruction:
    """自动识别 text / binary 格式并读取一个稀疏重建目录。"""
    d = Path(sparse_dir)
    if not d.exists():
        raise FileNotFoundError(f"稀疏重建目录不存在：{d}")
    rec = ColmapReconstruction()
    if (d / "cameras.bin").exists():
        _read_cameras_bin(d / "cameras.bin", rec)
        _read_images_bin(d / "images.bin", rec)
        p = d / "points3D.bin"
        if p.exists():
            _read_points_bin(p, rec)
    elif (d / "cameras.txt").exists():
        _read_cameras_txt(d / "cameras.txt", rec)
        _read_images_txt(d / "images.txt", rec)
        p = d / "points3D.txt"
        if p.exists():
            _read_points_txt(p, rec)
    else:
        raise FileNotFoundError(f"{d} 下没有 cameras.bin / cameras.txt")
    LOG.info("COLMAP 稀疏模型：%s", rec.summary())
    return rec


def _read_cameras_txt(path, rec):
    """COLMAP 文本格式：CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]"""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        cid = int(f[0])
        model = f[1]                       # 字符串模型名（二进制里才是 int id）
        w, h = int(f[2]), int(f[3])
        rec.cameras[cid] = ColmapCamera(cid, model, w, h,
                                        np.array([float(x) for x in f[4:]], np.float64))


def _read_images_txt(path, rec):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        f = line.split()
        iid = int(f[0])
        q = np.array([float(x) for x in f[1:5]], np.float64)
        t = np.array([float(x) for x in f[5:8]], np.float64)
        cid = int(f[8])
        name = " ".join(f[9:])
        xys = pids = None
        if i + 1 < len(lines):
            pts = lines[i + 1].split()
            if pts:
                arr = np.array([float(x) for x in pts], np.float64).reshape(-1, 3)
                xys, pids = arr[:, :2], arr[:, 2].astype(np.int64)
            i += 1
        rec.images[iid] = ColmapImage(iid, q, t, cid, name, xys, pids)
        i += 1


def _read_points_txt(path, rec):
    pts, cols = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        pts.append([float(f[1]), float(f[2]), float(f[3])])
        cols.append([float(f[4]), float(f[5]), float(f[6])])
    rec.points3D = np.array(pts, np.float64).reshape(-1, 3)
    rec.colors = np.array(cols, np.float64).reshape(-1, 3)


def _read_cameras_bin(path, rec):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cid, mid = struct.unpack("<ii", f.read(8))
            w, h = struct.unpack("<QQ", f.read(16))
            model, names = CAMERA_MODELS.get(mid, ("PINHOLE", []))
            params = np.array(struct.unpack("<" + "d" * len(names), f.read(8 * len(names))), np.float64)
            rec.cameras[cid] = ColmapCamera(cid, model, int(w), int(h), params)


def _read_images_bin(path, rec):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            iid = struct.unpack("<i", f.read(4))[0]
            q = np.array(struct.unpack("<4d", f.read(32)), np.float64)
            t = np.array(struct.unpack("<3d", f.read(24)), np.float64)
            cid = struct.unpack("<i", f.read(4))[0]
            name_chars = bytearray()
            while True:
                c = f.read(1)
                if c == b"\x00" or not c:
                    break
                name_chars += c
            name = name_chars.decode("utf-8", "replace")
            npts = struct.unpack("<Q", f.read(8))[0]
            if npts:
                raw = f.read(24 * npts)
                arr = np.frombuffer(raw, dtype=np.float64).reshape(-1, 3)
                xys, pids = arr[:, :2].copy(), arr[:, 2].astype(np.int64)
            else:
                xys = pids = None
            rec.images[iid] = ColmapImage(iid, q, t, cid, name, xys, pids)


def _read_points_bin(path, rec):
    pts, cols = [], []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(8)                                   # point3D_id
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            f.read(8)                                   # error
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * track_len)
            pts.append(xyz)
            cols.append(rgb)
    rec.points3D = np.array(pts, np.float64).reshape(-1, 3)
    rec.colors = np.array(cols, np.float64).reshape(-1, 3)


# --------------------------------------------------------------------------- 写出
def write_reconstruction(out_dir: str | Path, rec: ColmapReconstruction) -> str:
    """把重建结果写成 COLMAP text 格式（相机 + 影像位姿 + 稀疏点云）。"""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "cameras.txt", "w", encoding="utf-8") as f:
        f.write("# Camera list\n")
        for cid, cam in sorted(rec.cameras.items()):
            mid = next((k for k, v in CAMERA_MODELS.items() if v[0] == cam.model), 1)
            params = " ".join(f"{p:.10g}" for p in cam.params)
            f.write(f"{cid} {mid} {cam.width} {cam.height} {params}\n")
    with open(d / "images.txt", "w", encoding="utf-8") as f:
        f.write("# Image list\n")
        for iid, img in sorted(rec.images.items(), key=lambda kv: kv[1].name):
            q = " ".join(f"{v:.10g}" for v in img.qvec)
            t = " ".join(f"{v:.10g}" for v in img.tvec)
            f.write(f"{iid} {q} {t} {img.camera_id} {img.name}\n")
            f.write("\n")
    with open(d / "points3D.txt", "w", encoding="utf-8") as f:
        f.write("# 3D point list\n")
        for i in range(rec.points3D.shape[0]):
            x, y, z = rec.points3D[i]
            r, g, b = (rec.colors[i] if rec.colors.shape[0] > i else (128, 128, 128))
            f.write(f"{i + 1} {x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} 0.0\n")
    LOG.info("已写出 COLMAP 模型 -> %s（%d 影像 / %d 点）",
             d, len(rec.images), rec.points3D.shape[0])
    return str(d)
