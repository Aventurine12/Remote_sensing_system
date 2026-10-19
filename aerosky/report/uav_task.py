# -*- coding: utf-8 -*-
"""空天协同任务联动：由卫星发现的变化图斑自动生成无人机航摄任务建议。

核心是标准航空摄影测量公式（保持与项目中期方案一致的"卫星发现异常 → 无人机定向详查"闭环）：

    GSD = H · p / f                ->  航高  H = GSD · f / p
    航带间距  = GSD · 像幅宽(px) · (1 − 旁向重叠)
    曝光间距  = GSD · 像幅高(px) · (1 − 航向重叠)

据此算出航带数、每条航带像片数、总像片数、航程与预计作业时间，并输出蛇形航线的航点。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.geo import haversine_m

M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON_EQ = 111320.0


# 常见消费级无人机的相机参数（可从配置文件覆盖）
DEFAULT_SENSOR = {
    "name": "1 英寸 CMOS 20MP（等效 24mm）",
    "image_width_px": 5472,
    "image_height_px": 3648,
    "sensor_width_mm": 13.2,
    "sensor_height_mm": 8.8,
    "focal_mm": 8.8,
    "speed_m_s": 8.0,
    "battery_min": 30.0,
}


@dataclass
class FlightTask:
    polygon: List[List[float]]                  # 目标区多边形（经纬度）
    center: Tuple[float, float] = (0.0, 0.0)
    area_m2: float = 0.0
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    gsd_cm: float = 3.0
    height_m: float = 0.0
    front_overlap: float = 0.8
    side_overlap: float = 0.7
    line_spacing_m: float = 0.0
    trigger_spacing_m: float = 0.0
    n_strips: int = 0
    photos_per_strip: int = 0
    total_photos: int = 0
    flight_distance_m: float = 0.0
    flight_time_min: float = 0.0
    waypoints: List[Tuple[float, float]] = field(default_factory=list)
    sensor: Dict = field(default_factory=lambda: dict(DEFAULT_SENSOR))
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["polygon"] = [[round(x, 7) for x in p] for p in self.polygon]
        d["center"] = [round(self.center[0], 7), round(self.center[1], 7)]
        d["waypoints"] = [[round(x, 7), round(y, 7)] for x, y in self.waypoints]
        for k in ("area_m2", "line_spacing_m", "trigger_spacing_m", "flight_distance_m"):
            d[k] = round(float(d[k]), 2)
        d["height_m"] = round(float(self.height_m), 1)
        d["flight_time_min"] = round(float(self.flight_time_min), 1)
        return d

    def summary_lines(self) -> List[str]:
        return [
            f"目标区中心：{self.center[0]:.6f}°E, {self.center[1]:.6f}°N",
            f"目标区面积：{self.area_m2:,.0f} m²（约 {self.area_m2 / 10000:.3f} 公顷）",
            f"设计地面分辨率 GSD：{self.gsd_cm:.1f} cm/px",
            f"推荐航高：{self.height_m:.0f} m（相对起飞点）",
            f"重叠度：航向 {self.front_overlap * 100:.0f}% / 旁向 {self.side_overlap * 100:.0f}%",
            f"航带间距 {self.line_spacing_m:.0f} m，曝光间距 {self.trigger_spacing_m:.0f} m",
            f"航带数 {self.n_strips} 条，每条 {self.photos_per_strip} 张，共 {self.total_photos} 张像片",
            f"预计航程 {self.flight_distance_m / 1000:.2f} km，作业时间约 {self.flight_time_min:.0f} min",
        ]

    def geojson(self) -> dict:
        feats = [{
            "type": "Feature",
            "properties": {"kind": "target_area", "area_m2": round(self.area_m2, 1)},
            "geometry": {"type": "Polygon", "coordinates": [[[*p] for p in self.polygon] +
                                                            [[*self.polygon[0]]]]},
        }]
        if self.waypoints:
            feats.append({
                "type": "Feature",
                "properties": {"kind": "flight_path", "n_strips": self.n_strips,
                               "height_m": round(self.height_m, 1),
                               "photos": self.total_photos},
                "geometry": {"type": "LineString",
                             "coordinates": [[x, y] for x, y in self.waypoints]},
            })
            feats.append({
                "type": "Feature",
                "properties": {"kind": "takeoff", "height_m": round(self.height_m, 1)},
                "geometry": {"type": "Point", "coordinates": list(self.waypoints[0])},
            })
        return {"type": "FeatureCollection", "features": feats}


def polygon_area_m2(polygon: Sequence[Sequence[float]]) -> float:
    """球面多边形面积（经纬度），小区域内精度足够。"""
    pts = np.asarray(polygon, np.float64)
    if pts.shape[0] < 3:
        return 0.0
    lat0 = pts[:, 1].mean()
    x = pts[:, 0] * M_PER_DEG_LON_EQ * math.cos(math.radians(lat0))
    y = pts[:, 1] * M_PER_DEG_LAT
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def plan_flight(polygon: Sequence[Sequence[float]], cfg: Optional[dict] = None,
                sensor: Optional[dict] = None, gsd_cm: Optional[float] = None) -> FlightTask:
    """由目标区多边形生成完整航摄任务。"""
    cfg = dict(cfg or {})
    sen = dict(DEFAULT_SENSOR)
    sen.update(sensor or {})
    pts = np.asarray(polygon, np.float64)
    if pts.shape[0] < 3:
        raise ValueError("至少需要 3 个顶点的多边形")

    front = float(cfg.get("front_overlap", 0.8))
    side = float(cfg.get("side_overlap", 0.7))
    gsd = float(gsd_cm or cfg.get("gsd_cm", 3.0)) / 100.0        # m/px
    buffer_m = float(cfg.get("buffer_m", 10.0))

    # GSD <-> 航高
    pitch_mm = sen["sensor_width_mm"] / sen["image_width_px"]
    height = gsd * sen["focal_mm"] / pitch_mm
    if cfg.get("flight_height_m"):
        height = float(cfg["flight_height_m"])
        gsd = height * pitch_mm / sen["focal_mm"]

    foot_w = gsd * sen["image_width_px"]
    foot_h = gsd * sen["image_height_px"]
    line_spacing = max(foot_w * (1 - side), 1.0)
    trigger_spacing = max(foot_h * (1 - front), 1.0)

    center = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
    lat0 = center[1]
    mx = M_PER_DEG_LON_EQ * math.cos(math.radians(lat0))
    my = M_PER_DEG_LAT
    xs = (pts[:, 0] - center[0]) * mx
    ys = (pts[:, 1] - center[1]) * my
    x0, x1 = xs.min() - buffer_m, xs.max() + buffer_m
    y0, y1 = ys.min() - buffer_m, ys.max() + buffer_m

    n_strips = max(1, int(math.ceil((x1 - x0) / line_spacing)))
    if n_strips == 1:
        strip_xs = [(x0 + x1) / 2.0]
    else:
        strip_xs = [x0 + i * (x1 - x0) / (n_strips - 1) for i in range(n_strips)]
    photos = max(1, int(math.ceil((y1 - y0) / trigger_spacing)) + 1)

    waypoints = []
    for i, sx in enumerate(strip_xs):
        if i % 2 == 0:
            seg = [(sy, sx) for sy in np.linspace(y0, y1, max(photos, 2))]
        else:
            seg = [(sy, sx) for sy in np.linspace(y1, y0, max(photos, 2))]
        for sy, sxx in seg:
            waypoints.append((center[0] + sxx / mx, center[1] + sy / my))

    dist = abs(y1 - y0) * n_strips + line_spacing * (n_strips - 1)
    speed = float(sen.get("speed_m_s", 8.0))
    time_min = dist / max(speed, 0.5) / 60.0 + 0.25 * n_strips   # 含转弯

    task = FlightTask(
        polygon=[[float(p[0]), float(p[1])] for p in pts], center=center,
        area_m2=polygon_area_m2(pts),
        bbox=(float(pts[:, 0].min()), float(pts[:, 1].min()),
              float(pts[:, 0].max()), float(pts[:, 1].max())),
        gsd_cm=gsd * 100.0, height_m=height, front_overlap=front, side_overlap=side,
        line_spacing_m=line_spacing, trigger_spacing_m=trigger_spacing,
        n_strips=n_strips, photos_per_strip=photos, total_photos=n_strips * photos,
        flight_distance_m=dist, flight_time_min=time_min, waypoints=waypoints, sensor=sen,
    )
    task.notes = [
        f"相机：{sen['name']}，像幅 {sen['image_width_px']}×{sen['image_height_px']} px",
        "建议在目标区外扩 50 m 设置起降点，先试拍 3 张检查重叠与曝光。",
        f"目标区最大跨度约 {max(x1 - x0, y1 - y0):.0f} m，"
        f"{'单架次可完成' if time_min <= float(sen.get('battery_min', 30)) else '需分多架次'}。",
    ]
    return task


def nearest_anomaly(features: Sequence[dict], lon: float, lat: float) -> Optional[dict]:
    """找出离给定经纬度最近的异常图斑（用于点击联动）。"""
    best, best_d = None, float("inf")
    for f in features:
        from ..utils.geo import feature_centroid
        try:
            cx, cy = feature_centroid(f)
        except Exception:  # noqa: BLE001
            continue
        d = haversine_m(lon, lat, cx, cy)
        if d < best_d:
            best, best_d = f, d
    return best
