# -*- coding: utf-8 -*-
"""三维模型 <-> 卫星底图 的自动地理配准。

思路（全流程真实可跑，不依赖人工点选）：

1. 取三维模型里"抬高的表面"（建筑屋顶）的俯视占据图，作为模型侧的建筑轮廓；
2. 取卫星底图的建筑掩膜（可由本系统的语义分割给出）作为参考侧的建筑轮廓；
3. 在 (尺度 k, 朝向 θ, 平移 t) 上做搜索：对每个 (k, θ) 用 FFT 互相关求最优平移，
   并用积分图算局部 IoU，取 IoU 最大的候选；再做一轮细化；
4. 由最优相似变换解出 GeoReference（原点经纬度、米/局部单位、朝向），
   并用距离变换统计模型轮廓到参考轮廓的残差，作为几何偏差 RMSE 输出。

这样"点击二维变化图斑 -> 跳转到三维模型对应位置"就有了精确的坐标依据。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.geo import Raster
from ..utils.logger import get_logger
from .georeference import GeoReference, M_PER_DEG_LAT, M_PER_DEG_LON_EQ

LOG = get_logger("registration")

try:
    from scipy.signal import fftconvolve
    HAS_SCIPY = True
except Exception:  # pragma: no cover
    HAS_SCIPY = False


@dataclass
class RegistrationResult:
    georef: GeoReference
    iou: float = 0.0
    rmse_m: float = 0.0
    offset_px: Tuple[float, float] = (0.0, 0.0)
    scale_m_per_unit: float = 1.0
    rotation_deg: float = 0.0
    overlay: Optional[str] = None
    candidates: List[dict] = field(default_factory=list)

    def summary(self) -> dict:
        return {"iou": round(self.iou, 4), "rmse_m": round(self.rmse_m, 4),
                "origin_lon": round(self.georef.origin_lon, 7),
                "origin_lat": round(self.georef.origin_lat, 7),
                "meters_per_unit": round(self.scale_m_per_unit, 6),
                "heading_deg": round(self.rotation_deg, 3),
                "offset_px": [round(v, 2) for v in self.offset_px]}


# --------------------------------------------------------------------------- 主入口
def register_to_raster(means: np.ndarray, raster: Raster,
                       ref_mask: Optional[np.ndarray] = None,
                       heading_hint: Optional[float] = None,
                       gsd_hint: Optional[float] = None,
                       max_dim: int = 320,
                       elev_ratio: float = 0.75,
                       coarse_heading_step: float = 10.0,
                       coarse_scale_span: float = 3.0,
                       coarse_scale_steps: int = 13,
                       refine: int = 2,
                       overlay_path: Optional[str] = None,
                       progress: Optional[Callable[[str], None]] = None) -> RegistrationResult:
    """把模型局部坐标配准到卫星影像的 WGS84 坐标。"""
    def tick(msg):
        LOG.info(msg)
        if progress:
            progress(msg)

    pts = np.asarray(means, np.float64).reshape(-1, 3)
    if pts.shape[0] < 50:
        raise ValueError("高斯点太少，无法配准")

    # ---- 1) 模型侧建筑轮廓（抬高表面）
    z = pts[:, 2]
    thr = np.percentile(z, 55.0)
    hi = pts[z >= thr][:, :2]
    if hi.shape[0] < max(30, 0.02 * pts.shape[0]):
        hi = pts[:, :2]
    lo_xy = hi.min(0)
    size_local = np.maximum(hi.max(0) - lo_xy, 1e-6)
    model_pts = (hi - lo_xy) / max(size_local.max(), 1e-9)      # 归一到 [0,1]

    # ---- 2) 参考侧建筑掩膜
    S = _reference_mask(raster, ref_mask)
    gsd_m = gsd_hint or (sum(raster.pixel_size_m()) / 2.0)
    # 参考掩膜下采样，控制搜索规模
    f = max(1, int(math.ceil(max(S.shape) / max_dim)))
    S_small = S[::f, ::f]
    H, W = S_small.shape
    gsd_small = gsd_m * f
    tick(f"参考掩膜 {S.shape} -> {S_small.shape}，地面分辨率 {gsd_small:.3f} m/px")

    beta = math.sqrt(max((S_small > 0).sum(), 1)) / 4.0        # 模型轮廓目标像素尺度

    best = None
    cands: List[dict] = []
    span = np.pi if heading_hint is None else math.radians(60.0)
    center_h = 0.0 if heading_hint is None else math.radians(heading_hint)
    thetas = ([center_h + t for t in np.linspace(-span, span, int(360 / coarse_heading_step) + 1)]
              if heading_hint is not None
              else list(np.linspace(-math.pi, math.pi, int(360 / coarse_heading_step) + 1))[:-1])
    scales = np.geomspace(1.0 / coarse_scale_span, coarse_scale_span, coarse_scale_steps)

    tick(f"粗搜索：{len(thetas)} 个朝向 × {len(scales)} 个尺度")
    for it, th in enumerate(thetas):
        R = _rotmat(th)
        for k in scales:
            q = (model_pts @ R.T) * (k * beta)
            r = _eval_candidate(q, S_small, H, W)
            if r is None:
                continue
            score, dy, dx = r
            cands.append({"theta": th, "k": k, "score": score, "dy": dy, "dx": dx})
            if best is None or score > best["score"]:
                best = cands[-1]
        if progress and it % 8 == 0:
            progress(f"  朝向 {math.degrees(th):.0f}° 完成，当前最优 IoU {best['score']:.3f}" if best else "…")

    if best is None:
        raise RuntimeError("配准搜索失败：没有任何候选")
    tick(f"粗搜索最优：IoU={best['score']:.3f} θ={math.degrees(best['theta']):.1f}° k={best['k']:.3f}")

    # ---- 3) 细化
    th0, k0, dy0, dx0, s0 = best["theta"], best["k"], best["dy"], best["dx"], best["score"]
    for _ in range(refine):
        ths = th0 + np.linspace(-math.radians(coarse_heading_step), math.radians(coarse_heading_step), 9)
        ks = k0 * np.geomspace(0.85, 1.18, 9)
        for th in ths:
            R = _rotmat(th)
            for k in ks:
                q = (model_pts @ R.T) * (k * beta)
                r = _eval_candidate(q, S_small, H, W, local=True)
                if r is None:
                    continue
                score, dy, dx = r
                if score > s0:
                    th0, k0, dy0, dx0, s0 = th, k, dy, dx, score
    tick(f"细化后：IoU={s0:.3f} θ={math.degrees(th0):.2f}° k={k0:.4f}")

    # ---- 4) 换算成地理坐标
    Rf = _rotmat(th0)
    scale_unit = (k0 * beta) * gsd_small / max(size_local.max(), 1e-9)   # m / 局部单位

    # 小图 -> ENU 米：列向为东、行向为北（ENU 图 row 0 = 最北）
    # 取模型归一化坐标中心 (0.5, 0.5) 的落点作为参考点
    center_local = lo_xy + size_local * 0.5
    c_pts = np.array([[0.5, 0.5]])
    qc = (c_pts @ Rf.T) * (k0 * beta)
    col_c = qc[0, 0] + dx0
    row_c = qc[0, 1] + dy0
    east_c = col_c * gsd_small
    north_c = row_c * gsd_small

    # ENU 图 (0,0) 对应卫星影像的 (col=0, row=H_img-1*f)
    H_full = S.shape[0]
    lon0, lat0 = raster.xy_to_lonlat([0.0], [float(H_full)])
    lon0, lat0 = float(np.ravel(lon0)[0]), float(np.ravel(lat0)[0])
    lat_c = lat0 + north_c / M_PER_DEG_LAT
    lon_c = lon0 + east_c / (M_PER_DEG_LON_EQ * math.cos(math.radians(lat0)))

    # 模型局部原点 (0,0) 在参考系里位于 (-center_local)，据此解出 GeoReference 原点。
    # 先构造以模型中心为原点、朝向 -θ 的临时参考系，再由它反推局部原点对应的经纬度。
    ref = GeoReference(origin_lon=lon_c, origin_lat=lat_c, meters_per_unit=scale_unit,
                       heading_deg=-math.degrees(th0), source="auto_registration")
    lx, ly = ref.local_to_lonlat([-center_local[0]], [-center_local[1]])
    georef = GeoReference(origin_lon=float(np.ravel(lx)[0]), origin_lat=float(np.ravel(ly)[0]),
                          meters_per_unit=scale_unit, heading_deg=-math.degrees(th0),
                          source="auto_registration")

    # ---- 5) 残差统计（模型轮廓 -> 参考轮廓的距离变换）
    rmse_m = _residual_rmse(model_pts, Rf, k0 * beta, dx0, dy0, S_small, gsd_small)
    georef.rmse_m = rmse_m
    georef.confidence = float(s0)

    overlay = None
    if overlay_path:
        overlay = _save_overlay(raster, model_pts, Rf, k0 * beta, dx0, dy0, S, f, overlay_path)

    res = RegistrationResult(georef=georef, iou=float(s0), rmse_m=float(rmse_m),
                             offset_px=(float(dx0), float(dy0)),
                             scale_m_per_unit=float(scale_unit),
                             rotation_deg=-math.degrees(th0), overlay=overlay,
                             candidates=sorted(cands, key=lambda c: -c["score"])[:5])
    tick(f"配准完成：{georef.describe()}")
    return res


# --------------------------------------------------------------------------- 内部
def _rotmat(th: float) -> np.ndarray:
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]], np.float64)


def _reference_mask(raster: Raster, ref_mask: Optional[np.ndarray]) -> np.ndarray:
    """参考建筑掩膜：优先用传入的分割结果，否则用影像的边缘能量自动提取。"""
    if ref_mask is not None:
        m = (np.asarray(ref_mask) > 0).astype(np.uint8)
        if m.shape[:2] == (raster.height, raster.width):
            return m[::1, ::1]
        import cv2
        return (cv2.resize(m, (raster.width, raster.height), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
    import cv2
    rgb = raster.rgb
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.GaussianBlur(mag, (5, 5), 0)
    thr = np.percentile(mag, 88)
    m = (mag > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    return m


def _eval_candidate(q: np.ndarray, S: np.ndarray, H: int, W: int, local: bool = False):
    """给定模型点在小图坐标系里的位置，用 FFT 互相关求最优平移与 IoU。"""
    qmin = q.min(0)
    qq = q - qmin
    ph, pw = int(np.ceil(qq[:, 1].max())) + 2, int(np.ceil(qq[:, 0].max())) + 2
    if ph < 2 or pw < 2 or ph > 4 * H or pw > 4 * W:
        return None
    Himg = np.zeros((ph, pw), np.float32)
    xi = np.clip(qq[:, 0].astype(int), 0, pw - 1)
    yi = np.clip(qq[:, 1].astype(int), 0, ph - 1)
    np.add.at(Himg, (yi, xi), 1.0)
    Himg = np.minimum(Himg, 1.0)

    corr = _correlate(S.astype(np.float32), Himg)
    if corr.size == 0:
        return None
    nH = float(Himg.sum())
    Ssum = _box_sum(S.astype(np.float32), Himg.shape)
    denom = np.maximum(nH + Ssum - corr, 1e-6)
    iou = corr / denom
    idx = int(np.argmax(iou))
    dy, dx = np.unravel_index(idx, iou.shape)
    return float(iou[dy, dx]), int(dy), int(dx)


def _correlate(S: np.ndarray, T: np.ndarray) -> np.ndarray:
    """S 在 T 上滑动时二者重叠像素数，返回 (Hs-ht+1, Ws-wt+1)。"""
    if not HAS_SCIPY:
        import cv2
        res = cv2.matchTemplate(S, T, cv2.TM_CCORR)
        return res
    return fftconvolve(S, T[::-1, ::-1], mode="valid")


def _box_sum(S: np.ndarray, shape) -> np.ndarray:
    """每个滑动窗口内的 S 之和（用积分图一次算完）。"""
    th, tw = shape
    ii = np.zeros((S.shape[0] + 1, S.shape[1] + 1), np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(S, 0), 1)
    Hs, Ws = S.shape
    oh, ow = Hs - th + 1, Ws - tw + 1
    if oh <= 0 or ow <= 0:
        return np.zeros((max(oh, 1), max(ow, 1)), np.float64)
    total = (ii[th:th + oh, tw:tw + ow] - ii[:oh, tw:tw + ow]
             - ii[th:th + oh, :ow] + ii[:oh, :ow])
    return total


def _map_points(model_pts, R, k, dx, dy):
    q = (model_pts @ R.T) * k
    return q[:, 0] + dx, q[:, 1] + dy


def _residual_rmse(model_pts, R, k, dx, dy, S_small, gsd_small) -> float:
    """模型轮廓点到参考轮廓边界的距离的均方根，单位米。"""
    import cv2
    xs, ys = _map_points(model_pts, R, k, dx, dy)
    H, W = S_small.shape
    ref = S_small.astype(np.uint8)
    if ref.sum() == 0:
        return float("nan")
    # 参考掩膜边界
    er = cv2.erode(ref, np.ones((3, 3), np.uint8))
    boundary = (ref - er).astype(np.uint8)
    dist = cv2.distanceTransform((1 - boundary).astype(np.uint8), cv2.DIST_L2, 3)
    xi = np.clip(xs.astype(int), 0, W - 1)
    yi = np.clip(ys.astype(int), 0, H - 1)
    inside = ref[yi, xi] > 0
    if inside.sum() < 10:
        return float("nan")
    d = dist[yi[inside], xi[inside]]
    return float(np.sqrt(np.mean(d ** 2)) * gsd_small)


def _save_overlay(raster, model_pts, R, k, dx, dy, S, f, path) -> str:
    import cv2
    xs, ys = _map_points(model_pts, R, k, dx, dy)
    xs = xs * f
    ys = ys * f
    canvas = raster.rgb.copy()
    m = np.zeros(S.shape, np.uint8)
    xi = np.clip(xs.astype(int), 0, S.shape[1] - 1)
    yi = np.clip(ys.astype(int), 0, S.shape[0] - 1)
    m[yi, xi] = 255
    m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=2)
    canvas[m > 0] = (0.4 * canvas[m > 0] + 0.6 * np.array([255, 40, 40])).astype(np.uint8)
    ref = (S > 0).astype(np.uint8)
    edges = cv2.morphologyEx(ref, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    canvas[edges > 0] = np.array([40, 220, 40])
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), canvas[:, :, ::-1])
    return str(p)
