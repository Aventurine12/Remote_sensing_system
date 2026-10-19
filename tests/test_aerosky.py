# -*- coding: utf-8 -*-
"""核心链路的自动化测试（不需要任何外部权重，全程可离线运行）。

    python -m pytest tests -q
    # 或（不装 pytest 时）
    python tests/test_aerosky.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from aerosky.gsplat.camera import Camera, OrbitCamera
from aerosky.gsplat.colmap_io import read_reconstruction
from aerosky.gsplat.ply_io import ply_to_gaussians
from aerosky.gsplat.render_np import SplatRenderer
from aerosky.gsplat.synthetic import make_dataset
from aerosky.inference.tiler import Tiler, blend_window, nms
from aerosky.registration.georeference import GeoReference
from aerosky.report.uav_task import plan_flight, polygon_area_m2
from aerosky.utils.geo import (Raster, mask_to_polygons, make_demo_geotiff,
                               overlay_mask, to_uint8)
from aerosky.utils.vis import colorize_mask, iou_f1, recall_precision_at_mask


def test_geo_raster_roundtrip(tmp: Path):
    p = make_demo_geotiff(tmp / "a.tif", size=(768, 512), seed=1)
    r = Raster.open(p)
    assert r.width == 768 and r.height == 512, (r.width, r.height)
    assert r.info.georeferenced and r.info.crs.endswith("4326")
    # 像素 -> 经纬度 -> 像素 往返一致
    lon, lat = r.xy_to_lonlat([100, 400], [50, 300])
    x, y = r.lonlat_to_pixel(lon, lat)
    assert np.allclose(x, [100, 400], atol=1e-3), x
    assert np.allclose(y, [50, 300], atol=1e-3), y
    b = r.bounds_lonlat()
    assert b[0] < b[2] and b[1] < b[3]
    mx, my = r.pixel_size_m()
    assert 0.4 < mx < 0.6 and 0.4 < my < 0.6, (mx, my)


def test_mask_to_polygons_world_coords(tmp: Path):
    r = Raster.open(make_demo_geotiff(tmp / "b.tif", size=(512, 512), seed=2))
    m = np.zeros((512, 512), np.uint8)
    m[100:200, 150:300] = 1
    feats = mask_to_polygons(m, r.info.transform, min_area_px=10)
    assert len(feats) >= 1
    xy = feats[0]["geometry"]["coordinates"][0]
    xs = [c[0] for c in xy]
    ys = [c[1] for c in xy]
    # 世界坐标应落在影像范围内
    assert min(xs) > r.info.bounds[0] and max(xs) < r.info.bounds[2]
    assert min(ys) > r.info.bounds[1] and max(ys) < r.info.bounds[3]


def test_tiler_stitching():
    t = Tiler(tile_size=64, overlap=16, batch_size=2)
    img = np.zeros((100, 140, 3), np.uint8)
    wins = t.windows(100, 140)
    assert len(wins) >= 4
    # 覆盖性：每个像素至少被一个窗口覆盖
    cov = np.zeros((100, 140), np.int32)
    for x0, y0, x1, y1 in wins:
        cov[y0:y1, x0:x1] += 1
    assert cov.min() >= 1

    def predict(tiles):
        # 恒等映射：拼接结果应当等于原图
        return np.stack([tile.astype(np.float32).mean(axis=-1) / 255.0 for tile in tiles])[:, None]

    out = t.run_mask(img, predict)[:, :, 0]
    assert out.shape == (100, 140)
    assert abs(float(out.mean()) - 0.0) < 1e-6
    w = blend_window(32, 32, "cosine")
    assert abs(float(w[16, 16]) - 1.0) < 0.2 and w[0, 0] < 0.5


def test_nms_merges_duplicates():
    from aerosky.inference.base import Detection
    ds = [Detection(np.array([0, 0, 10, 10], np.float32), 0.9, "车", 0),
          Detection(np.array([1, 1, 11, 11], np.float32), 0.8, "车", 0),
          Detection(np.array([50, 50, 60, 60], np.float32), 0.7, "车", 0)]
    out = nms(ds, iou_thr=0.5)
    assert len(out) == 2, [d.box.tolist() for d in out]


def test_metrics():
    gt = np.zeros((20, 20), np.uint8); gt[5:15, 5:15] = 1
    pred = gt.copy()
    m = iou_f1(pred, gt)
    assert abs(m["IoU"] - 1.0) < 1e-6 and abs(m["F1"] - 1.0) < 1e-6
    pred2 = np.zeros_like(gt)
    r = recall_precision_at_mask(pred2, gt)
    assert abs(r["漏检率"] - 1.0) < 1e-6
    cm = colorize_mask(gt)
    assert cm.shape == (20, 20, 3)


def test_georeference_roundtrip():
    g = GeoReference(origin_lon=126.63, origin_lat=45.75, meters_per_unit=0.05,
                     heading_deg=37.0)
    lon, lat = g.local_to_lonlat([100.0, -250.0], [80.0, 400.0])
    x, y = g.lonlat_to_local(lon, lat)
    assert np.allclose(x, [100.0, -250.0], atol=1e-3), x
    assert np.allclose(y, [80.0, 400.0], atol=1e-3), y
    # 1 个局部单位 = 0.05 m
    d = np.hypot(*(np.array(g.local_to_local_m([1.0], [0.0])).ravel()))
    assert abs(d - 0.05) < 1e-9


def test_uav_task():
    poly = [[126.632, 45.752], [126.634, 45.752], [126.634, 45.750], [126.632, 45.750]]
    area = polygon_area_m2(poly)
    # 约 155 m（东西）× 221 m（南北）≈ 3.4 万平方米
    assert 30000 < area < 40000, area
    task = plan_flight(poly, {"gsd_cm": 3.0})
    assert task.height_m > 60 and task.n_strips >= 1 and task.total_photos >= 1
    assert len(task.waypoints) >= 2
    assert len(task.summary_lines()) == 8
    gj = task.geojson()
    assert gj["type"] == "FeatureCollection" and len(gj["features"]) >= 2


def test_synthetic_dataset_and_colmap(tmp: Path):
    info = make_dataset(tmp / "scene", n_views=8, width=96, height=72, n_points=500)
    rec = read_reconstruction(info["sparse_dir"])
    assert len(rec.images) == 8 and rec.points3D.shape[0] == 500
    assert len(rec.named()) == 8
    from aerosky.gsplat.trainer import load_dataset
    views, pts, cols = load_dataset(info["images_dir"], info["sparse_dir"])
    assert len(views) == 8 and pts.shape[0] == 500
    v = views[0]
    assert v.image.shape[:2] == (72, 96)
    # 相机位姿自洽：模型中心应落在相机前方
    center = pts.mean(0)
    pc = v.cam.world_to_cam(center[None, :])[0]
    assert pc[2] > 0, pc


def test_rasterizer_gradients_and_ply(tmp: Path):
    import torch
    from aerosky.gsplat.model import GaussianModel
    from aerosky.gsplat.rasterizer import camera_K, camera_viewmat
    torch.manual_seed(0)
    model = GaussianModel(sh_degree=1, device="cpu")
    pts = np.random.RandomState(0).uniform(-1, 1, (200, 3))
    model.init_from_points(pts, np.random.RandomState(1).uniform(0, 1, (200, 3)))
    cam = Camera.look_at([0, -4, 1.5], [0, 0, 0], up=(0, 0, -1), width=64, height=48)
    vm = camera_viewmat(cam.R, cam.t)
    K = camera_K(cam.fx, cam.fy, cam.cx, cam.cy)
    img, alpha, info = model(vm, K, 64, 48)
    assert img.shape == (48, 64, 3) and info["num_pairs"] > 0
    loss = img.mean()
    loss.backward()
    assert model._means.grad is not None and torch.isfinite(model._means.grad).all()
    assert float(model._means.grad.abs().sum()) > 0

    # 导出 -> 重新载入 -> 参数一致
    p = tmp / "pc.ply"
    model.save_ply(p)
    d = ply_to_gaussians(p)
    assert np.allclose(d["means"], model._means.detach().numpy(), atol=1e-4)
    assert np.allclose(d["opacities"], model.get_opacity().detach().numpy(), atol=1e-3)
    r = SplatRenderer(d["means"], d["scales"], d["quats"], d["opacities"], d["shs"], 1)
    c, s = r.center_scale()
    rgb, acc = r.render(OrbitCamera(target=np.asarray(c), distance=s * 1.5).to_camera(64, 48))
    assert rgb.shape == (48, 64, 3) and rgb.max() > 0


def test_rasterizer_trains(tmp: Path):
    """跑 80 次迭代（关闭致密化以排除扰动），验证损失确实在下降。"""
    from aerosky.gsplat.trainer import GaussianTrainer
    info = make_dataset(tmp / "s2", n_views=8, width=112, height=84, n_points=800)
    tr = GaussianTrainer({"max_gaussians": 20000, "sh_degree": 1, "downscale": 1,
                          "iterations": 80, "densify_from": 10000, "densify_until": 10001,
                          "opacity_reset_interval": 0, "densify_grad_threshold": 0.0,
                          "learning_rate": {}}, device="cpu")
    tr.prepare(info["images_dir"], info["sparse_dir"])
    res = tr.train(iterations=80)
    hist = res["history"]
    assert len(hist) >= 3, hist
    head = sum(h["loss"] for h in hist[:2]) / 2
    tail = sum(h["loss"] for h in hist[-2:]) / 2
    assert tail < head, (head, tail)
    assert res["num_gaussians"] > 0


def test_report_and_pipeline_offline(tmp: Path):
    from aerosky.pipeline import Pipeline
    from aerosky.report.builder import ReportContext, build_html
    p = make_demo_geotiff(tmp / "r.tif", size=(384, 256), seed=3)
    pipe = Pipeline(device="cpu")
    res = pipe.segment(Raster.open(p))
    assert res.mask is not None and res.mask.shape == (256, 384)
    assert isinstance(res.summary(), dict)
    gj = pipe.save_geojson(res, tmp / "seg.geojson")
    data = json.loads(Path(gj).read_text(encoding="utf-8"))
    assert data["type"] == "FeatureCollection"
    html = build_html(ReportContext(tasks=[dict(res.summary(), model=res.model_name)],
                                    changes=[], conclusion=["测试"]), tmp / "rep")
    assert Path(html).exists() and "分析报告" in Path(html).read_text(encoding="utf-8")


def test_utils_vis():
    a = np.zeros((10, 10, 3), np.uint8)
    b = overlay_mask(a, np.ones((10, 10), bool), (255, 0, 0), 0.5)
    assert b[0, 0, 0] > 100 and b[0, 0, 1] < 50
    assert to_uint8(np.linspace(0, 255, 100).reshape(10, 10)).dtype == np.uint8


# --------------------------------------------------------------------------- 运行器
def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    tmp_root = Path(tempfile.mkdtemp(prefix="aerosky_test_"))
    failed = []
    for fn in tests:
        d = tmp_root / fn.__name__
        d.mkdir(parents=True, exist_ok=True)
        try:
            import inspect
            if "tmp" in inspect.signature(fn).parameters:
                fn(d)
            else:
                fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAIL  {fn.__name__}: {e}")
            traceback.print_exc()
            failed.append(fn.__name__)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
