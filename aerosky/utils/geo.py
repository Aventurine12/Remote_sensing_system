# -*- coding: utf-8 -*-
"""地理空间基础：GeoTIFF 读写、仿射变换、像素坐标 <-> WGS84 经纬度、矢量化与 GeoJSON 输出。

系统所有"真实世界坐标"都从这里出去：模型输出的是像素，报告与无人机任务需要的是经纬度。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

try:  # rasterio 提供 GDAL 能力（含仿射变换参数与投影信息解析）
    import rasterio
    from rasterio import features as rio_features
    from rasterio.crs import CRS
    from rasterio.transform import Affine, from_bounds
    from rasterio.warp import transform as rio_transform
    from rasterio.windows import Window

    HAS_RASTERIO = True
except Exception:  # pragma: no cover - 仅在依赖缺失时走到
    rasterio = None
    rio_features = None
    CRS = None
    Affine = None
    rio_transform = None
    Window = None
    HAS_RASTERIO = False

WGS84 = "EPSG:4326"


# --------------------------------------------------------------------------- 数据结构
@dataclass
class RasterInfo:
    """一份栅格影像的元信息（与数组解耦，便于只读元数据时不必载入像素）。"""

    path: str = ""
    width: int = 0
    height: int = 0
    count: int = 1
    dtype: str = "uint8"
    crs: str | None = None
    transform: Sequence[float] | None = None      # GDAL 六参数 (a,b,c,d,e,f)
    bounds: Tuple[float, float, float, float] | None = None   # (left, bottom, right, top)
    nodata: float | None = None
    band_names: List[str] = field(default_factory=list)
    georeferenced: bool = False

    @property
    def res(self) -> Tuple[float, float]:
        if not self.transform:
            return (1.0, 1.0)
        return (abs(self.transform[0]), abs(self.transform[4]))

    @property
    def crs_is_wgs84(self) -> bool:
        return bool(self.crs) and str(self.crs).upper().endswith("4326")

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        if self.bounds is not None:
            d["bounds"] = list(self.bounds)
        return d


class Raster:
    """栅格影像（读取时把像元载入内存；大图请用 read_window / Tiler 分块）。"""

    def __init__(self, array: np.ndarray, info: RasterInfo):
        self.array = array
        self.info = info

    # ---------------------------------------------------------------- 构造
    @classmethod
    def open(cls, path: str | Path, bands: Sequence[int] | None = None) -> "Raster":
        path = str(path)
        if HAS_RASTERIO and _looks_like_tiff(path):
            with rasterio.open(path) as ds:
                idxs = list(bands) if bands else list(range(1, ds.count + 1))
                arr = ds.read(idxs)
                arr = np.transpose(arr, (1, 2, 0)) if len(idxs) > 1 else arr[0]
                info = RasterInfo(
                    path=path, width=ds.width, height=ds.height, count=len(idxs),
                    dtype=str(ds.dtypes[0]),
                    crs=str(ds.crs) if ds.crs else None,
                    transform=tuple(ds.transform)[:6] if ds.transform else None,
                    bounds=tuple(ds.bounds) if ds.bounds else None,
                    nodata=ds.nodata,
                    band_names=[ds.descriptions[i - 1] or f"band{i}" for i in idxs],
                    georeferenced=bool(ds.crs and ds.transform),
                )
                return cls(arr, info)
        return cls._open_plain(path)

    @classmethod
    def _open_plain(cls, path: str) -> "Raster":
        """非 GeoTIFF（png/jpg 或未带地理信息的 tif）也能进系统：给一个近似"经纬度"占位变换。"""
        try:
            import cv2
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(path)
            if img.ndim == 2:
                arr = img
                count = 1
            elif img.shape[2] == 1:
                arr = img[:, :, 0]
                count = 1
            else:
                arr = img[:, :, ::-1]
                count = img.shape[2]
        except Exception:
            from PIL import Image
            im = Image.open(path)
            arr = np.array(im)
            count = 1 if arr.ndim == 2 else arr.shape[2]
        h, w = arr.shape[:2]
        # 无地理参考：以影像中心为 126.63E / 45.75N（哈尔滨），按 0.5 m/px 折算经纬度步长
        m_per_deg = 111320.0
        px = 0.5 / m_per_deg
        transform = (px, 0.0, 126.63 - w * px / 2, 0.0, -px, 45.75 + h * px / 2)
        info = RasterInfo(path=path, width=w, height=h, count=count, dtype=str(arr.dtype),
                          crs=None, transform=transform,
                          bounds=(transform[2], transform[5] + h * transform[4],
                                  transform[2] + w * transform[0], transform[5]),
                          georeferenced=False, band_names=[f"band{i + 1}" for i in range(count)])
        return cls(arr, info)

    @classmethod
    def from_array(cls, array: np.ndarray, transform: Sequence[float],
                   crs: str = WGS84, nodata: float | None = None) -> "Raster":
        h, w = array.shape[:2]
        count = 1 if array.ndim == 2 else array.shape[2]
        info = RasterInfo(path="", width=w, height=h, count=count, dtype=str(array.dtype),
                          crs=crs, transform=tuple(transform),
                          bounds=(transform[2], transform[5] + h * transform[4],
                                  transform[2] + w * transform[0], transform[5]),
                          nodata=nodata, georeferenced=True,
                          band_names=[f"band{i + 1}" for i in range(count)])
        return cls(array, info)

    # ---------------------------------------------------------------- 属性
    @property
    def width(self) -> int:
        return self.info.width

    @property
    def height(self) -> int:
        return self.info.height

    @property
    def rgb(self) -> np.ndarray:
        """用于显示的三通道 uint8 影像（多光谱取前 3 波段并拉伸）。"""
        a = self.array
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=-1)
        elif a.shape[2] >= 3:
            a = a[:, :, :3]
        else:
            a = np.repeat(a[:, :, :1], 3, axis=2)
        return to_uint8(a)

    @property
    def profile(self) -> dict:
        return {
            "driver": "GTiff", "width": self.width, "height": self.height,
            "count": self.info.count, "dtype": self.info.dtype,
            "crs": self.info.crs, "transform": Affine(*self.info.transform) if self.info.transform else None,
            "nodata": self.info.nodata,
        }

    # ---------------------------------------------------------------- 坐标换算
    def xy_to_lonlat(self, xs: float | np.ndarray, ys: float | np.ndarray,
                     src_crs: str | None = None):
        """影像坐标（左上角为原点，向右 x 增大、向下 y 增大）-> WGS84 经纬度。"""
        gx, gy = self.pixel_to_world(xs, ys)
        return self.world_to_lonlat(gx, gy, src_crs=src_crs or self.info.crs)

    def pixel_to_world(self, xs, ys):
        a, b, c, d, e, f = self.info.transform
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        return a * xs + b * ys + c, d * xs + e * ys + f

    def world_to_pixel(self, gx, gy):
        a, b, c, d, e, f = self.info.transform
        det = a * e - b * d
        gx = np.asarray(gx, dtype=np.float64) - c
        gy = np.asarray(gy, dtype=np.float64) - f
        xs = (e * gx - b * gy) / det
        ys = (-d * gx + a * gy) / det
        return xs, ys

    def world_to_lonlat(self, gx, gy, src_crs: str | None = None):
        crs = src_crs or self.info.crs
        if not crs or not HAS_RASTERIO:
            return np.asarray(gx), np.asarray(gy)
        if str(crs).upper().endswith("4326"):
            return np.asarray(gx), np.asarray(gy)
        lon, lat = rio_transform(crs, WGS84, np.asarray(gx).ravel().tolist(),
                                 np.asarray(gy).ravel().tolist())
        return np.asarray(lon), np.asarray(lat)

    def lonlat_to_pixel(self, lon, lat):
        if not self.info.crs or not HAS_RASTERIO or str(self.info.crs).upper().endswith("4326"):
            gx, gy = np.asarray(lon), np.asarray(lat)
        else:
            gx, gy = rio_transform(WGS84, self.info.crs,
                                   np.asarray(lon).ravel().tolist(),
                                   np.asarray(lat).ravel().tolist())
            gx, gy = np.asarray(gx), np.asarray(gy)
        return self.world_to_pixel(gx, gy)

    def bounds_lonlat(self) -> Tuple[float, float, float, float]:
        """(min_lon, min_lat, max_lon, max_lat)"""
        xs = [0, self.width, 0, self.width]
        ys = [0, 0, self.height, self.height]
        lon, lat = self.xy_to_lonlat(xs, ys)
        return float(np.min(lon)), float(np.min(lat)), float(np.max(lon)), float(np.max(lat))

    def pixel_size_m(self) -> Tuple[float, float]:
        """单像元的地面尺寸（米），纬度方向按影像中心纬度做余弦校正。"""
        lon0, lat0 = self.xy_to_lonlat([self.width / 2], [self.height / 2])
        lon1, lat1 = self.xy_to_lonlat([self.width / 2 + 1], [self.height / 2 + 1])
        mx = abs(float(lon1[0] - lon0[0])) * 111320.0 * math.cos(math.radians(float(lat0[0])))
        my = abs(float(lat1[0] - lat0[0])) * 110540.0
        return mx or 1.0, my or 1.0

    # ---------------------------------------------------------------- 写出
    def save(self, path: str | Path, array: np.ndarray | None = None,
             dtype: str | None = None, nodata: float | None = None) -> str:
        path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        arr = self.array if array is None else array
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        count = 1 if arr.ndim == 2 else arr.shape[2]
        if HAS_RASTERIO:
            prof = dict(driver="GTiff", width=self.width, height=self.height,
                        count=count, dtype=dtype or str(arr.dtype),
                        crs=self.info.crs if self.info.georeferenced else None,
                        transform=Affine(*self.info.transform),
                        nodata=nodata if nodata is not None else self.info.nodata,
                        compress="deflate")
            with rasterio.open(path, "w", **prof) as ds:
                if count == 1:
                    ds.write(arr.astype(prof["dtype"]), 1)
                else:
                    ds.write(np.transpose(arr, (2, 0, 1)).astype(prof["dtype"]))
            return path
        from PIL import Image
        Image.fromarray(to_uint8(arr)).save(path)
        return path


def _looks_like_tiff(path: str) -> bool:
    return Path(path).suffix.lower() in (".tif", ".tiff", ".gtiff", ".vrt", ".img", ".jp2")


def to_uint8(arr: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """按 2%~98% 分位拉伸到 uint8，用于显示。"""
    a = np.asarray(arr, dtype=np.float32)
    if a.dtype != np.uint8:
        if a.ndim == 3:
            out = np.zeros(a.shape, np.uint8)
            for i in range(a.shape[2]):
                out[:, :, i] = to_uint8(a[:, :, i], lo, hi)
            return out
        lo_v, hi_v = np.percentile(a, [lo, hi])
        if hi_v - lo_v < 1e-6:
            hi_v = lo_v + 1.0
        return np.clip((a - lo_v) / (hi_v - lo_v) * 255.0, 0, 255).astype(np.uint8)
    return a


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(255, 60, 60), alpha=0.45) -> np.ndarray:
    """把二值/多类掩膜以半透明色叠加到 RGB 底图上。"""
    out = rgb.astype(np.float32).copy()
    m = np.asarray(mask).astype(bool)
    c = np.array(color, dtype=np.float32)
    out[m] = out[m] * (1 - alpha) + c * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def mask_to_polygons(mask: np.ndarray, transform: Sequence[float],
                     min_area_px: float = 16.0, simplify_px: float = 1.5) -> List[dict]:
    """把二值掩膜矢量化为 GeoJSON Feature（世界坐标，由 transform 决定）。"""
    if not HAS_RASTERIO:
        return _mask_to_polygons_shapely(mask, transform, min_area_px)
    from shapely.geometry import shape as shp_shape
    m = (np.asarray(mask) > 0).astype(np.uint8)
    tf = Affine(*transform)
    feats = []
    # 注意：rasterio.features.shapes 产出的是 GeoJSON 风格的 dict，不是 shapely 几何体
    for geom, val in rio_features.shapes(m, mask=m.astype(bool), transform=tf):
        if int(val) != 1:
            continue
        poly = shp_shape(geom)
        area_px = abs(poly.area) / abs(tf.a * tf.e)
        if area_px < min_area_px:
            continue
        if simplify_px > 0:
            poly = poly.simplify(abs(tf.a) * simplify_px, preserve_topology=True)
            if poly.is_empty:
                continue
        feats.append({"type": "Feature", "properties": {"area_px": round(area_px, 1)},
                      "geometry": poly.__geo_interface__})
    return feats


def _mask_to_polygons_shapely(mask, transform, min_area_px):
    import cv2
    from shapely.geometry import Polygon
    cnts, _ = cv2.findContours((np.asarray(mask) > 0).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    a, b, c, d, e, f = transform
    feats = []
    for cnt in cnts:
        if cv2.contourArea(cnt) < min_area_px:
            continue
        pts = cnt[:, 0, :].astype(np.float64)
        gx = a * pts[:, 0] + b * pts[:, 1] + c
        gy = d * pts[:, 0] + e * pts[:, 1] + f
        poly = Polygon(np.stack([gx, gy], axis=1))
        if not poly.is_valid:
            poly = poly.buffer(0)
        feats.append({"type": "Feature", "properties": {}, "geometry": poly.__geo_interface__})
    return feats


def geojson_of(features: Iterable[dict], crs: str = WGS84,
               to_wgs84: bool = False, src_crs: str | None = None) -> dict:
    """打包成 GeoJSON FeatureCollection；to_wgs84=True 时把坐标从 src_crs 投到经纬度。"""
    feats = list(features)
    if to_wgs84 and src_crs and not str(src_crs).upper().endswith("4326") and HAS_RASTERIO:
        feats = [_reproject_feature(f, src_crs) for f in feats]
    return {"type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": f"urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": feats}


def _reproject_feature(feat: dict, src_crs: str) -> dict:
    geom = json.loads(json.dumps(feat["geometry"]))
    coords = geom["coordinates"]

    def walk(node):
        if isinstance(node, (list, tuple)) and node and isinstance(node[0], (int, float)):
            lon, lat = rio_transform(src_crs, WGS84, [node[0]], [node[1]])
            return [lon[0], lat[0]]
        return [walk(x) for x in node]

    geom["coordinates"] = walk(coords)
    out = dict(feat)
    out["geometry"] = geom
    return out


def save_geojson(path: str | Path, fc: dict) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)
    return str(p)


def feature_centroid(feat: dict) -> Tuple[float, float]:
    """GeoJSON Feature 的点中心（经纬度）。"""
    from shapely.geometry import shape
    c = shape(feat["geometry"]).centroid
    return float(c.x), float(c.y)


def feature_bounds_lonlat(feat: dict) -> Tuple[float, float, float, float]:
    from shapely.geometry import shape
    return tuple(shape(feat["geometry"]).bounds)  # type: ignore[return-value]


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def make_demo_geotiff(path: str | Path, size: Tuple[int, int] = (2048, 1536),
                      seed: int = 2026, with_change: bool = False) -> str:
    """生成一张带真实地理参考的演示用"城市航拍" GeoTIFF（离线可用，不依赖任何外部数据）。

    用程序化方式铺设街区、道路、建筑屋顶、植被与水体，并加上运动场、停车场、船只等
    "典型目标"，使语义分割 / 目标检测 / 变化检测三条链路都能在离线演示数据上出结果。

    Args:
        size: (宽, 高)，单位像素。
    """
    rng = np.random.default_rng(seed)
    w, h = size
    img = np.zeros((h, w, 3), np.float32)
    img[:, :] = np.array([92, 104, 88])                      # 裸地/背景

    # 植被斑块
    for _ in range(28):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        ry, rx = rng.integers(60, 240), rng.integers(60, 240)
        yy, xx = np.ogrid[:h, :w]
        m = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
        img[m] = np.array([58, 96, 52]) + rng.normal(0, 6, 3)

    # 水系
    band = np.zeros((h, w), bool)
    y0 = rng.integers(int(h * 0.55), int(h * 0.8))
    for x in range(w):
        y = int(y0 + 60 * math.sin(x / 260.0))
        band[max(0, y - 45):min(h, y + 45), x] = True
    img[band] = np.array([46, 78, 122])

    # 道路网格
    road = np.zeros((h, w), bool)
    for x in range(0, w, 256):
        road[:, max(0, x - 6):x + 6] = True
    for y in range(0, h, 224):
        road[max(0, y - 6):y + 6, :] = True
    img[road] = np.array([120, 120, 118])

    # 建筑（屋顶：亮灰/砖红，带少量阴影）
    buildings = np.zeros((h, w), bool)
    for by in range(24, h - 80, 224):
        for bx in range(24, w - 80, 256):
            for _ in range(rng.integers(1, 3)):
                bw = int(rng.integers(46, 92))
                bh = int(rng.integers(40, 84))
                ox = bx + int(rng.integers(8, 60))
                oy = by + int(rng.integers(8, 60))
                if oy + bh + 12 >= h or ox + bw + 12 >= w:
                    continue
                color = np.array([196, 192, 186]) if rng.random() < 0.7 else np.array([168, 96, 76])
                buildings[oy:oy + bh, ox:ox + bw] = True
                img[oy:oy + bh, ox:ox + bw] = color + rng.normal(0, 5, 3)
                img[oy + bh:oy + bh + 10, ox:ox + bw + 8] = np.array([70, 74, 70])  # 阴影
    img = np.clip(img, 0, 255)

    # ---- 典型目标：运动场地 / 停车场 / 船只（让 DOTA 目标检测在演示数据上也有检出）
    import cv2
    img = np.clip(img, 0, 255).astype(np.uint8)      # OpenCV 绘图在 uint8 上做

    def _field(cy: int, cx: int, hh: int, ww: int, base, line=(240, 240, 240)):
        y0, y1 = max(0, cy - hh // 2), min(h, cy + hh // 2)
        x0, x1 = max(0, cx - ww // 2), min(w, cx + ww // 2)
        img[y0:y1, x0:x1] = base
        cv2.rectangle(img, (x0 + 4, y0 + 4), (x1 - 4, y1 - 4), line, 2)
        cv2.line(img, ((x0 + x1) // 2, y0 + 4), ((x0 + x1) // 2, y1 - 4), line, 2)
        cv2.ellipse(img, ((x0 + x1) // 2, (y0 + y1) // 2), (max(6, ww // 8), max(6, hh // 8)),
                    0, 0, 360, line, 2)

    # 田径场（红色跑道环 + 绿色内场）
    ty, tx = int(h * 0.20), int(w * 0.78)
    cv2.ellipse(img, (tx, ty), (120, 66), 0, 0, 360, (196, 88, 66), -1)
    cv2.ellipse(img, (tx, ty), (92, 42), 0, 0, 360, (72, 138, 70), -1)
    cv2.ellipse(img, (tx, ty), (120, 66), 0, 0, 360, (240, 240, 240), 2)

    # 网球场 ×2
    _field(int(h * 0.42), int(w * 0.80), 96, 58, np.array([62, 136, 122]))
    _field(int(h * 0.42), int(w * 0.80) + 74, 96, 58, np.array([62, 136, 122]))

    # 篮球场
    _field(int(h * 0.70), int(w * 0.80), 88, 56, np.array([150, 112, 74]))

    # 停车场（灰色场地 + 白色车位线 + 若干车辆）
    py0, px0 = int(h * 0.62), int(w * 0.20)
    img[py0:py0 + 130, px0:px0 + 220] = np.array([128, 128, 126])
    for i in range(7):
        x = px0 + 14 + i * 30
        cv2.line(img, (x, py0 + 10), (x, py0 + 120), (235, 235, 235), 1)
        if rng.random() < 0.75:
            col = np.array([rng.integers(40, 230) for _ in range(3)], np.float32)
            img[py0 + 24:py0 + 70, x + 4:x + 22] = col

    # 水面上的船只
    if band.any():
        yb = int(np.argmax(band.sum(axis=1)))
        for i in range(6):
            sx = int(rng.integers(60, max(61, w - 120)))
            sy = yb + int(rng.integers(-25, 25))
            img[max(0, sy - 5):sy + 5, sx:sx + 34] = np.array([232, 232, 228])
            img[max(0, sy - 5):sy + 5, sx + 12:sx + 22] = np.array([150, 150, 150])

    img = np.clip(img, 0, 255)

    if with_change:
        # 在后时相里新增一片建筑并清掉一角植被，用于演示变化检测
        for _ in range(6):
            oy = int(rng.integers(100, h - 160))
            ox = int(rng.integers(100, w - 160))
            bh, bw = int(rng.integers(44, 78)), int(rng.integers(50, 90))
            img[oy:oy + bh, ox:ox + bw] = np.array([210, 206, 200], np.uint8)
            buildings[oy:oy + bh, ox:ox + bw] = True

    img = img.astype(np.float32)

    # 地理参考：哈尔滨市中心附近，0.5 m/px
    lon0, lat0 = 126.6320, 45.7520
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat0))
    px = 0.5
    transform = (px / m_per_deg_lon, 0.0, lon0,
                 0.0, -px / 110540.0, lat0)
    r = Raster.from_array(img.astype(np.uint8), transform, crs=WGS84)
    r.save(path, dtype="uint8")
    return str(path)
