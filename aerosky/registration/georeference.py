# -*- coding: utf-8 -*-
"""地理配准：把 3D 高斯模型所在的局部坐标系锚定到 WGS84 经纬度。

局部坐标系（模型自己随便定的单位与朝向）与真实地理坐标之间差一个**相似变换**：

    [E]       [ x·cosθ + y·sinθ ]
    [N] = s ·  [-x·sinθ + y·cosθ ]

其中 s 是"每个局部单位对应多少米"，θ 是局部 +x 轴相对正东方向的逆时针转角
（局部 +y 朝南，与俯视图一致）。再按等距圆柱近似把 east/north 米换算成经纬度，
在城市尺度（几公里）内误差 < 0.1 m。

配准结果就是三元组 (origin_lonlat, meters_per_unit, heading)，有了它：
  · 卫星图上的变化图斑（经纬度）-> 直接落到三维模型的局部坐标 -> "点击二维图斑跳转三维"；
  · 三维模型 -> 可以叠加回卫星底图核验站位。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON_EQ = 111320.0


@dataclass
class GeoReference:
    origin_lon: float = 126.6320
    origin_lat: float = 45.7520
    meters_per_unit: float = 1.0
    heading_deg: float = 0.0
    source: str = "manual"          # manual | exif | auto_registration
    rmse_m: float | None = None
    confidence: float | None = None

    # ------------------------------------------------------------------ 基础
    @property
    def origin(self) -> Tuple[float, float]:
        return self.origin_lon, self.origin_lat

    def _rot(self) -> Tuple[float, float]:
        th = math.radians(self.heading_deg)
        return math.cos(th), math.sin(th)

    # ------------------------------------------------------------------ 正反变换
    def local_to_lonlat(self, x, y):
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        c, s = self._rot()
        east = self.meters_per_unit * (x * c + y * s)
        north = self.meters_per_unit * (-x * s + y * c)
        lat = self.origin_lat + north / M_PER_DEG_LAT
        lon = self.origin_lon + east / (M_PER_DEG_LON_EQ * math.cos(math.radians(self.origin_lat)))
        return lon, lat

    def lonlat_to_local(self, lon, lat):
        lon = np.asarray(lon, np.float64)
        lat = np.asarray(lat, np.float64)
        east = (lon - self.origin_lon) * M_PER_DEG_LON_EQ * math.cos(math.radians(self.origin_lat))
        north = (lat - self.origin_lat) * M_PER_DEG_LAT
        c, s = self._rot()
        x = (east * c - north * s) / max(self.meters_per_unit, 1e-9)
        y = (east * s + north * c) / max(self.meters_per_unit, 1e-9)
        return x, y

    def local_to_local_m(self, x, y):
        """局部单位 -> 以原点为基准的东/北米坐标。"""
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        c, s = self._rot()
        return (self.meters_per_unit * (x * c + y * s),
                self.meters_per_unit * (-x * s + y * c))

    # ------------------------------------------------------------------ 顶点变换
    def transform_points(self, pts: np.ndarray) -> np.ndarray:
        """把模型的点云（N,3）平移到"以原点为局部原点的米制 ENU"坐标系（z 保留为高程米）。"""
        pts = np.asarray(pts, np.float64).reshape(-1, 3)
        e, n = self.local_to_local_m(pts[:, 0], pts[:, 1])
        return np.stack([e, n, pts[:, 2] * self.meters_per_unit], axis=1)

    def anchor_of(self, lon: float, lat: float) -> Tuple[float, float]:
        """返回局部坐标系下代表该经纬度的点（供"点击图斑跳转模型"用）。"""
        x, y = self.lonlat_to_local(lon, lat)
        return float(np.ravel(x)[0]), float(np.ravel(y)[0])

    # ------------------------------------------------------------------ 存取
    def to_dict(self) -> dict:
        return {"origin_lon": self.origin_lon, "origin_lat": self.origin_lat,
                "meters_per_unit": self.meters_per_unit, "heading_deg": self.heading_deg,
                "source": self.source, "rmse_m": self.rmse_m, "confidence": self.confidence}

    @classmethod
    def from_dict(cls, d: dict) -> "GeoReference":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str | Path) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return str(p)

    @classmethod
    def load(cls, path: str | Path) -> "GeoReference":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ------------------------------------------------------------------ 与卫星图互检
    def footprint_corners_local(self, xmin, xmax, ymin, ymax) -> List[Tuple[float, float]]:
        pts = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        lon, lat = self.local_to_lonlat([p[0] for p in pts], [p[1] for p in pts])
        return list(zip(np.ravel(lon).tolist(), np.ravel(lat).tolist()))

    def describe(self) -> str:
        return (f"原点 ({self.origin_lon:.6f}E, {self.origin_lat:.6f}N) | "
                f"比例 {self.meters_per_unit:.4g} m/unit | 朝向 {self.heading_deg:.1f}° | "
                f"RMSE {self.rmse_m if self.rmse_m is None else round(self.rmse_m, 3)} m | "
                f"来源 {self.source}")
