# -*- coding: utf-8 -*-
"""结构化分析报告生成：HTML（可直接打印）+ PDF（由 Qt 渲染输出）。

报告内容对应"城市治理业务闭环"需要的东西：
    地理位置 · 变化类型 · 量化指标（面积/高度差）· 趋势研判 · 处置建议
"""
from __future__ import annotations

import base64
import html
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..utils.geo import feature_centroid
from ..utils.logger import get_logger

LOG = get_logger("report")

CSS = """
body{font-family:"Microsoft YaHei","SimSun",Arial,sans-serif;margin:0;padding:28px 36px;
     color:#1c1f24;line-height:1.65;font-size:13px;}
h1{font-size:22px;text-align:center;margin:6px 0 2px;letter-spacing:1px;}
.sub{text-align:center;color:#606a78;font-size:12px;margin-bottom:18px;}
h2{font-size:15px;margin:22px 0 8px;padding-left:9px;border-left:4px solid #2f6fb5;color:#173a5e;}
h3{font-size:13px;margin:14px 0 6px;color:#24486e;}
table{border-collapse:collapse;width:100%;margin:6px 0 12px;font-size:12px;}
th,td{border:1px solid #b9c3cf;padding:5px 7px;text-align:center;}
th{background:#eaf1f8;font-weight:600;}
td.l{text-align:left;}
.kv td:first-child{background:#f4f7fa;width:22%;text-align:left;}
.fig{margin:8px 0 4px;text-align:center;}
.fig img{max-width:100%;border:1px solid #ccd4dd;}
.cap{text-align:center;color:#5a6473;font-size:11px;margin-bottom:12px;}
.tag{display:inline-block;padding:1px 8px;border-radius:9px;background:#e8f2e8;color:#2c6b34;
     font-size:11px;margin-right:6px;}
.tag.warn{background:#fdecec;color:#a03030;}
.note{background:#f7f9fc;border-left:3px solid #9db8d6;padding:8px 12px;font-size:12px;color:#41505f;}
.foot{margin-top:26px;border-top:1px solid #d5dde5;padding-top:8px;color:#7b8794;font-size:11px;}
ul{margin:6px 0 10px 18px;padding:0;} li{margin:3px 0;}
"""


@dataclass
class ReportContext:
    """报告需要的全部素材。"""

    title: str = "空天协同的多光谱遥感影像智能解译与三维监测系统"
    subtitle: str = "城市更新场景智能分析报告"
    area_name: str = "哈尔滨市典型城市更新场景"
    analyst: str = "空天协同项目组"
    backend: str = ""
    rasters: Dict[str, str] = field(default_factory=dict)      # {角色: 路径}
    raster_meta: List[dict] = field(default_factory=list)
    tasks: List[dict] = field(default_factory=list)            # TaskResult.summary()
    changes: List[dict] = field(default_factory=list)          # 变化图斑清单
    detections: List[dict] = field(default_factory=list)
    uav_tasks: List[dict] = field(default_factory=list)
    reconstruction: Optional[dict] = None
    registration: Optional[dict] = None
    models: Dict[str, dict] = field(default_factory=dict)
    images: List[tuple] = field(default_factory=list)          # (相对路径, 题注)
    conclusion: List[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- HTML
def build_html(ctx: ReportContext, out_dir: str | Path) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    now = time.strftime("%Y-%m-%d %H:%M")

    p = [f"<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>",
         f"<title>{html.escape(ctx.title)}</title><style>{CSS}</style></head><body>"]
    p.append(f"<h1>{html.escape(ctx.title)}</h1>")
    p.append(f"<div class='sub'>{html.escape(ctx.subtitle)}　|　生成时间 {now}　|　"
             f"引擎 aerosky v1.0</div>")

    # ---- 1 概览
    p.append("<h2>一、项目与数据概览</h2>")
    rows = [("项目名称", ctx.title), ("分析区域", ctx.area_name), ("分析人", ctx.analyst),
            ("生成时间", now), ("推理后端", ctx.backend or "本地推理")]
    for role, path in (ctx.rasters or {}).items():
        rows.append((f"{role}影像", str(path)))
    p.append(_kv_table(rows))

    if ctx.raster_meta:
        p.append("<h3>1.1 影像地理信息</h3>")
        p.append(_table(["影像", "尺寸(px)", "波段", "地面分辨率(m)", "坐标系", "范围(经纬度)"],
                        [[m.get("name", "-"), m.get("size", "-"), m.get("count", "-"),
                          m.get("res", "-"), m.get("crs", "-"), m.get("bounds", "-")]
                         for m in ctx.raster_meta]))

    # ---- 2 智能解译结果
    p.append("<h2>二、遥感智能解译结果</h2>")
    if ctx.tasks:
        rows = []
        for t in ctx.tasks:
            extra = []
            if t.get("num_changes") is not None:
                extra.append(f"变化图斑 {t['num_changes']} 个")
            if t.get("changed_area_m2") is not None:
                extra.append(f"变化面积 {t['changed_area_m2']:,.0f} m²")
            if t.get("num_detections") is not None:
                extra.append(f"目标 {t['num_detections']} 个")
            if t.get("num_polygons") is not None:
                extra.append(f"图斑 {t['num_polygons']} 个")
            rows.append([_task_cn(t.get("task")), t.get("model", "-"),
                         "是" if t.get("fallback") else "否",
                         f"{t.get('elapsed', 0):.1f} s", "；".join(extra) or "-"])
        p.append(_table(["解译任务", "使用模型", "是否降级", "耗时", "主要结果"], rows))
    else:
        p.append("<div class='note'>本次未执行解译任务。</div>")

    for rel, cap in ctx.images:
        p.append(f"<div class='fig'><img src='{rel}'/></div><div class='cap'>{html.escape(cap)}</div>")

    # ---- 3 变化图斑
    if ctx.changes:
        p.append("<h2>三、变化图斑清单与处置建议</h2>")
        rows = []
        for i, c in enumerate(ctx.changes[:200], 1):
            rows.append([i, c.get("type", "变化"), f"{c.get('area_m2', 0):,.1f}",
                         f"{c.get('center_lon', 0):.6f}", f"{c.get('center_lat', 0):.6f}",
                         c.get("advice", "现场核查")])
        p.append(_table(["序号", "变化类型", "面积(m²)", "中心经度", "中心纬度", "建议处置"],
                        rows))
        p.append("<div class='note'>说明：面积由变化掩膜像元数乘以地面分辨率平方得到；"
                 "中心经纬度由像元坐标经 GDAL 仿射变换反算至 WGS84，可直接作为无人机航摄"
                 "与现场核查的定位依据。</div>")

    # ---- 4 无人机任务
    if ctx.uav_tasks:
        p.append("<h2>四、无人机航摄任务建议</h2>")
        for i, task in enumerate(ctx.uav_tasks, 1):
            p.append(f"<h3>4.{i} 目标区 {i}</h3>")
            p.append(_table(["参数", "取值"], [(k, v) for k, v in task.get("rows", [])]))
            if task.get("path_img"):
                p.append(f"<div class='fig'><img src='{task['path_img']}'/></div>"
                         f"<div class='cap'>图　目标区 {i} 航摄航线与目标范围</div>")
            for n in task.get("notes", []):
                p.append(f"<div class='note'>{html.escape(n)}</div>")

    # ---- 5 三维重建
    if ctx.reconstruction:
        r = ctx.reconstruction
        p.append("<h2>五、三维高斯泼溅重建与地理配准</h2>")
        p.append(_table(["指标", "取值"], [
            ("重建来源", "COLMAP 稀疏模型" if r.get("source") == "colmap" else "合成多视角演示数据"),
            ("高斯单元数", f"{r.get('num_gaussians', 0):,}"),
            ("球谐阶数", r.get("sh_degree", "-")),
            ("渲染质量 PSNR", f"{r.get('psnr', 0):.2f} dB"),
            ("结构相似度 SSIM", f"{r.get('ssim', 0):.4f}"),
            ("训练耗时", f"{r.get('seconds', 0):.1f} s"),
            ("推理设备", r.get("device", "-")),
            ("模型文件", Path(str(r.get("ply", ""))).name),
        ]))
        if ctx.registration:
            g = ctx.registration
            p.append("<h3>5.1 与卫星影像的地理配准</h3>")
            p.append(_table(["参数", "取值"], [
                ("配准方式", "自动配准（轮廓 IoU 最优化）"),
                ("原点经度", f"{g.get('origin_lon', 0):.7f}"),
                ("原点纬度", f"{g.get('origin_lat', 0):.7f}"),
                ("比例", f"{g.get('meters_per_unit', 0):.5f} m / 局部单位"),
                ("朝向", f"{g.get('heading_deg', 0):.2f}°"),
                ("轮廓重合度 IoU", f"{g.get('iou', 0):.4f}"),
                ("几何偏差 RMSE", f"{g.get('rmse_m', 0):.3f} m"),
            ]))
        for item in (r.get("previews") or [])[:6]:
            rel, cap = _as_fig(item)
            p.append(f"<div class='fig'><img src='{rel}'/></div><div class='cap'>{html.escape(cap)}</div>")

    # ---- 6 模型信息
    if ctx.models:
        p.append("<h2>六、模型与运行环境</h2>")
        p.append(_table(["任务", "模型", "状态", "设备"],
                        [[_task_cn(k), v.get("name", "-"),
                          "降级" if v.get("fallback") else "正常",
                          v.get("device", "-")] for k, v in ctx.models.items()]))

    # ---- 7 结论
    if ctx.conclusion:
        p.append("<h2>七、结论与建议</h2><ul>")
        for c in ctx.conclusion:
            p.append(f"<li>{html.escape(c)}</li>")
        p.append("</ul>")

    p.append(f"<div class='foot'>本报告由 aerosky 空天协同智能监测系统自动生成，"
             f"数据与结论供城市更新核查参考。生成时间 {now}。</div>")
    p.append("</body></html>")

    path = out_dir / "analysis_report.html"
    path.write_text("\n".join(p), encoding="utf-8")
    LOG.info("报告已生成：%s", path)
    return str(path)


def _task_cn(task: str) -> str:
    return {"segmentation": "语义分割", "detection": "目标检测",
            "change_detection": "变化检测"}.get(task, task or "-")


def _as_fig(item) -> tuple:
    """图可以是 (相对路径, 题注) 也可以是纯路径，统一成二元组。"""
    if isinstance(item, (tuple, list)) and len(item) == 2:
        return str(item[0]), str(item[1])
    p = Path(str(item))
    return p.name if p.is_absolute() else str(item), p.stem


def _kv_table(rows) -> str:
    body = "".join(f"<tr><td>{html.escape(str(k))}</td><td class='l'>{html.escape(str(v))}</td></tr>"
                   for k, v in rows)
    return f"<table class='kv'>{body}</table>"


def _table(headers, rows) -> str:
    h = "".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
    body = []
    for r in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in r) + "</tr>")
    return f"<table><thead><tr>{h}</tr></thead><tbody>{''.join(body)}</tbody></table>"


# --------------------------------------------------------------------------- PDF
def html_to_pdf(html_path: str | Path, pdf_path: str | Path,
                page_size_mm=(210, 297), margins_mm=(14, 14, 14, 14)) -> Optional[str]:
    """用 Qt 的富文本引擎把报告渲染成 PDF（不额外依赖 reportlab / weasyprint）。"""
    try:
        from PyQt5.QtCore import QUrl, QSizeF, QMarginsF
        from PyQt5.QtGui import QTextDocument, QPageSize, QPageLayout
        from PyQt5.QtPrintSupport import QPrinter
        from PyQt5.QtWidgets import QApplication
        import sys
    except Exception as e:  # noqa: BLE001
        LOG.warning("Qt 不可用，跳过 PDF 生成：%s", e)
        return None

    html_path = Path(html_path)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    doc = QTextDocument()
    doc.setBaseUrl(QUrl.fromLocalFile(str(html_path.parent) + "/"))
    doc.setHtml(html_path.read_text(encoding="utf-8"))

    printer = QPrinter(QPrinter.HighResolution)
    printer.setOutputFormat(QPrinter.PdfFormat)
    printer.setOutputFileName(str(pdf_path))
    printer.setPageSize(QPageSize(QPageSize.A4))
    # PyQt5 的签名是 setPageMargins(left, top, right, bottom, unit)，不接受 QMarginsF
    printer.setPageMargins(12, 12, 12, 12, QPrinter.Millimeter)
    doc.print_(printer)
    LOG.info("PDF 已生成：%s", pdf_path)
    return str(pdf_path)


def export_report(ctx: ReportContext, out_dir: str | Path,
                  formats: Sequence[str] = ("html", "pdf")) -> Dict[str, str]:
    out_dir = Path(out_dir)
    out: Dict[str, str] = {}
    if "html" in formats or "pdf" in formats:
        out["html"] = build_html(ctx, out_dir)
    if "pdf" in formats:
        pdf = html_to_pdf(out["html"], out_dir / "analysis_report.pdf")
        if pdf:
            out["pdf"] = pdf
    return out
