# -*- coding: utf-8 -*-
"""无人值守的端到端演示：一条命令跑完"输入影像 → 智能解译 → 三维重建 → 配准 → 报告"。

    python run_app.py --demo

输出（默认落在 outputs/demo_run/）：
    satellite_before.tif / satellite_after.tif   前/后时相演示影像
    fig_segmentation.png / fig_detection.png / fig_change.png
    3dgs/point_cloud.ply + orbit_*.png           三维高斯模型与环绕视图
    registration_overlay.png                     配准核验图
    uav_task.geojson / change_detection.geojson
    analysis_report.html / analysis_report.pdf   结构化分析报告
    run_summary.json                             全流程指标
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

from .pipeline import Pipeline
from .reconstruction import load_renderer, reconstruct
from .registration import register_to_raster
from .report.builder import ReportContext, export_report
from .report.uav_task import plan_flight
from .utils.geo import Raster, make_demo_geotiff, overlay_mask, save_geojson
from .utils.logger import get_logger
from .utils.vis import colorize_mask, draw_boxes

LOG = get_logger("demo")


def run_demo(cfg, out_root: Optional[str] = None, device: Optional[str] = None,
             reuse_3d: bool = False) -> int:
    t_all = time.time()
    out = Path(out_root or (Path(cfg.get_path("paths.output")) / "demo_run"))
    out.mkdir(parents=True, exist_ok=True)
    summary = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "output": str(out)}
    print("=" * 78)
    print("  aerosky 端到端演示")
    print("=" * 78)

    # ---------------------------------------------------------------- 1 数据
    print("\n[1/7] 准备演示数据（带真实地理参考的多光谱影像）…")
    a_path = out / "satellite_before.tif"
    b_path = out / "satellite_after.tif"
    data_source = "程序化合成"
    try:                                   # 优先用公开数据集里的真实卫星影像
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fetch_real_sample", Path(__file__).resolve().parent.parent / "tools" / "fetch_real_sample.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        info = mod.build(out / "real", n=9)
        a_path, b_path = Path(info["before"]), Path(info["after"])
        data_source = f"真实卫星影像（{info['source']}，{info['n']} 景拼接）"
    except Exception as e:  # noqa: BLE001
        print(f"      真实影像获取失败（{str(e).splitlines()[0]}），回退到程序化合成数据")
        make_demo_geotiff(a_path, size=(2048, 1536), seed=2026, with_change=False)
        make_demo_geotiff(b_path, size=(2048, 1536), seed=2026, with_change=True)
    raster_a = Raster.open(a_path)
    raster_b = Raster.open(b_path)
    mx, my = raster_a.pixel_size_m()
    print(f"      数据来源：{data_source}")
    print(f"      影像 {raster_a.width}×{raster_a.height}，地面分辨率 {mx:.3f} m/px，"
          f"范围 {raster_a.bounds_lonlat()}")
    summary["raster"] = {"size": [raster_a.width, raster_a.height],
                         "gsd_m": round(mx, 4), "source": data_source,
                         "bounds_lonlat": [round(v, 6) for v in raster_a.bounds_lonlat()]}

    # ---------------------------------------------------------------- 2 解译
    import cv2
    pipe = Pipeline(cfg, device=device)
    results = {}

    print("\n[2/7] 语义分割 …")

    def seg_prog(cur, total):
        print(f"\r      分块推理 {cur}/{total}", end="")

    try:
        seg = pipe.segment(raster_a, progress=seg_prog)
        print(f"\r      完成：模型 {seg.model_name}{'（降级）' if seg.fallback else ''}，"
              f"耗时 {seg.elapsed:.1f}s，图斑 {seg.stats.get('num_polygons')} 个")
        results["segmentation"] = seg
        cv2.imwrite(str(out / "fig_segmentation.png"), colorize_mask(seg.mask)[:, :, ::-1])
    except Exception:  # noqa: BLE001
        print("      失败：" + traceback.format_exc(limit=2))

    print("\n[3/7] 目标检测 …")

    def det_prog(cur, total):
        print(f"\r      分块推理 {cur}/{total}", end="")

    try:
        det = pipe.detect(raster_a, progress=det_prog)
        print(f"\r      完成：模型 {det.model_name}{'（降级）' if det.fallback else ''}，"
              f"耗时 {det.elapsed:.1f}s，目标 {det.num_detections} 个")
        results["detection"] = det
        if det.detections:
            img = draw_boxes(raster_a.rgb, [d.box for d in det.detections],
                             [d.label for d in det.detections],
                             [d.score for d in det.detections],
                             obb=any(d.obb for d in det.detections))
            cv2.imwrite(str(out / "fig_detection.png"), img[:, :, ::-1])
        print("      类别分布：" + str(det.stats.get("by_class", {})))
    except Exception:  # noqa: BLE001
        print("      失败：" + traceback.format_exc(limit=2))

    print("\n[4/7] 变化检测 …")

    def chg_prog(cur, total):
        print(f"\r      双时相推理 {cur}/{total}", end="")

    try:
        chg = pipe.change(raster_a, raster_b, progress=chg_prog)
        print(f"\r      完成：模型 {chg.model_name}{'（降级）' if chg.fallback else ''}，"
              f"耗时 {chg.elapsed:.1f}s")
        print(f"      变化图斑 {chg.stats.get('num_changes')} 个（新增 "
              f"{chg.stats.get('new_count')}，拆除 {chg.stats.get('del_count')}），"
              f"变化面积 {chg.stats.get('changed_area_m2')} m²")
        results["change_detection"] = chg
        cv2.imwrite(str(out / "fig_change.png"),
                    overlay_mask(raster_a.rgb, chg.mask)[:, :, ::-1])
        pipe.save_geojson(chg, out / "change_detection.geojson")
    except Exception:  # noqa: BLE001
        print("      失败：" + traceback.format_exc(limit=2))

    # ---------------------------------------------------------------- 5 三维
    print("\n[5/7] 3D 高斯泼溅重建 …")
    gs = dict(cfg.get_path("gsplat", {}) or {})
    recon = None
    try:
        device_used = device or _pick_device()
        prog_state = {"t": 0}

        def recon_prog(line):
            """reconstruct() 的进度回调现在按行（字符串）回调，不是旧的 (i, n, rec)。"""
            s = str(line).strip()
            if s:
                print(f"\r      {s[:110]}", end="", flush=True)

        recon = reconstruct(out / "3dgs", gs, device=device_used, progress=recon_prog,
                            reuse=reuse_3d)
        ps = f"{recon.psnr:.2f}" if recon.psnr is not None else "-"
        ss = f"{recon.ssim:.4f}" if recon.ssim is not None else "-"
        print(f"\r      完成：后端 {recon.backend}，{recon.num_gaussians:,} 个高斯，"
              f"PSNR {ps} dB，SSIM {ss}，耗时 {recon.seconds:.1f}s（{device_used}）")
        print(f"      模型：{recon.ply}")
        summary["reconstruction"] = recon.summary()
    except Exception:  # noqa: BLE001
        print("      失败：" + traceback.format_exc(limit=3))

    # ---------------------------------------------------------------- 6 配准
    print("\n[6/7] 地理配准 …")
    reg = None
    if recon is not None:
        try:
            from .gsplat.ply_io import read_ply
            d = read_ply(recon.ply)
            means = np.stack([d["x"], d["y"], d["z"]], axis=1)
            ref = (results["segmentation"].mask == 1).astype(np.uint8) \
                if "segmentation" in results else None
            reg = register_to_raster(means, raster_a, ref_mask=ref,
                                     overlay_path=str(out / "registration_overlay.png"),
                                     progress=lambda m: print("      " + m))
            print(f"      完成：IoU={reg.iou:.4f}，RMSE={reg.rmse_m:.3f} m，"
                  f"比例 {reg.scale_m_per_unit:.4f} m/unit，朝向 {reg.rotation_deg:.1f}°")
            reg.georef.save(out / "georeference.json")
            summary["registration"] = reg.summary()
        except Exception:  # noqa: BLE001
            print("      失败：" + traceback.format_exc(limit=3))
    else:
        print("      跳过（无三维模型）")

    # ---------------------------------------------------------------- 7 任务与报告
    print("\n[7/7] 无人机任务与结构化报告 …")
    uav_tasks = []
    change_items = []
    if "change_detection" in results and raster_a is not None:
        chg = results["change_detection"]
        for i, f in enumerate(chg.features[:12]):
            coords = f["geometry"]["coordinates"][0]
            lon = np.array([c[0] for c in coords])
            lat = np.array([c[1] for c in coords])
            mx_, my_ = raster_a.pixel_size_m()
            change_items.append({
                "type": f["properties"].get("change_type", "变化"),
                "area_m2": float(f["properties"].get("area_px", 0)) * mx_ * my_,
                "center_lon": float(lon.mean()), "center_lat": float(lat.mean()),
                "advice": "现场核查并比对规划许可",
            })
        if chg.features:
            # 取面积最大的变化图斑作为航摄目标（最小的图斑只有几平方米，不构成一个作业区）
            import shapely.geometry as _sg
            def _area(f):
                try:
                    return _sg.shape(f["geometry"]).area
                except Exception:  # noqa: BLE001
                    return 0.0
            target_feat = max(chg.features, key=_area)
            try:
                task = plan_flight(target_feat["geometry"]["coordinates"][0],
                                   dict(cfg.get_path("uav_task", {}) or {}))
                save_geojson(out / "uav_task.geojson", task.geojson())
                uav_tasks.append({
                    "rows": [(l.split("：")[0], "：".join(l.split("：")[1:]))
                             for l in task.summary_lines()],
                    "notes": task.notes})
                print("      无人机任务：")
                for l in task.summary_lines():
                    print("        " + l)
            except Exception:  # noqa: BLE001
                print("      任务生成失败：" + traceback.format_exc(limit=2))

    images = []
    for name, cap in (("fig_segmentation.png", "图　语义分割结果（地物分类图）"),
                      ("fig_detection.png", "图　目标检测结果（DOTA 航拍目标）"),
                      ("fig_change.png", "图　变化检测结果（红色为检出的变化区域）")):
        if (out / name).exists():
            images.append((name, cap))
    if reg is not None and reg.overlay:
        images.append((Path(reg.overlay).name, "图　三维模型与卫星底图配准核验"))

    ctx = ReportContext(
        area_name="哈尔滨市典型城市更新场景（演示区）",
        backend=("CUDA" if (recon and recon.device == "cuda") else "CPU"),
        rasters={"前时相": str(a_path), "后时相": str(b_path)},
        raster_meta=[{"name": "前时相", "size": f"{raster_a.width}×{raster_a.height}",
                      "count": raster_a.info.count, "res": f"{mx:.2f}",
                      "crs": raster_a.info.crs or "无(占位)",
                      "bounds": "%.5f~%.5fE, %.5f~%.5fN" % raster_a.bounds_lonlat()}],
        tasks=[dict(r.summary(), model=r.model_name) for r in results.values()],
        changes=change_items, uav_tasks=uav_tasks,
        reconstruction=(recon.summary() if recon else None),
        registration=(reg.summary() if reg else None),
        models=pipe.hub.status(), images=images,
        conclusion=[
            "系统完成了「多光谱影像输入 → 语义分割/目标检测/变化检测 → 三维高斯泼溅重建 → "
            "地理配准 → 结构化报告输出」的空天协同端到端流程。",
            f"本次共检出变化图斑 {len(change_items)} 个"
            + (f"，合计变化面积 {sum(c['area_m2'] for c in change_items):,.0f} m²"
               if change_items else "") + "。",
            "全部解译结果均已由 GDAL 仿射变换映射至 WGS84 经纬度，可直接作为无人机航摄"
            "与现场核查的定位依据。",
            "建议对新增建筑类图斑优先核查规划许可；对拆除/损毁类图斑核实修复重建需求。",
        ],
    )
    paths = export_report(ctx, out)
    for k, v in paths.items():
        print(f"      {k.upper()}: {v}")

    summary["tasks"] = {k: v.summary() for k, v in results.items()}
    summary["models"] = pipe.hub.status()
    summary["changes"] = change_items
    summary["elapsed_total_s"] = round(time.time() - t_all, 1)
    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(out / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print(f"  全部完成，总耗时 {summary['elapsed_total_s']} s")
    print(f"  成果目录：{out}")
    print("=" * 78)
    return 0


def _pick_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"
