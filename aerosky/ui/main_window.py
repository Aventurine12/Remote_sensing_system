# -*- coding: utf-8 -*-
"""主窗口：把数据、三大解译任务、三维重建、地理配准、无人机任务与报告串成一个界面。"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIcon, QPixmap
from PyQt5.QtWidgets import (QAction, QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                             QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                             QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                             QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
                             QSlider, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
                             QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from ..config import load_config
from ..pipeline import Pipeline
from ..reconstruction import ReconstructionResult, load_renderer, reconstruct
from ..registration import GeoReference, register_to_raster
from ..report.builder import ReportContext, export_report
from ..report.uav_task import plan_flight
from ..utils.geo import Raster, make_demo_geotiff, save_geojson
from ..utils.logger import get_logger
from ..utils.weights import ensure_building_segmentation, ensure_dota_obb, list_local_weights
from .map_view import MapView
from .viewer3d import SplatView
from .workers import TaskWorker

LOG = get_logger("ui")


class MainWindow(QMainWindow):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or load_config()
        self.pipe = Pipeline(self.cfg)
        self.raster_a: Optional[Raster] = None
        self.raster_b: Optional[Raster] = None
        self.results: Dict[str, object] = {}
        self.recon: Optional[ReconstructionResult] = None
        self.georef: Optional[GeoReference] = None
        self.change_items: List[dict] = []
        self.worker: Optional[TaskWorker] = None
        self.out_dir = Path(self.cfg.get_path("paths.output", "outputs")) / "session"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.setWindowTitle(self.cfg.get_path("app.title"))
        self.resize(*self.cfg.get_path("app.window", [1600, 950]))
        self._build_ui()
        self._build_menu()
        self.log(f"欢迎使用 {self.cfg.get_path('app.title')} v{self.cfg.get_path('app.version')}")
        self.log(f"推理设备：{self._device_text()}　输出目录：{self.out_dir}")
        QTimer.singleShot(600, self._report_model_status)

    # ==================================================================== 界面
    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_center())
        splitter.addWidget(self._build_right())
        splitter.setSizes([320, 760, 460])

        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(6, 2, 6, 4)
        self.progress = QProgressBar()
        self.progress.setMaximumHeight(14)
        self.progress.setTextVisible(False)
        self.logbox = QPlainTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setMaximumBlockCount(3000)
        self.logbox.setFont(QFont("Consolas", 9))
        self.logbox.setMinimumHeight(130)
        bl.addWidget(self.progress)
        bl.addWidget(self.logbox)

        outer = QSplitter(Qt.Vertical)
        outer.addWidget(splitter)
        outer.addWidget(bottom)
        outer.setSizes([780, 170])
        self.setCentralWidget(outer)
        self.statusBar().showMessage("就绪")

    # ---------------------------------------------------------------- 左栏
    def _build_left(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(8, 8, 6, 8)
        lay.setSpacing(8)

        g = QGroupBox("① 数据")
        gl = QVBoxLayout(g)
        self.btn_open_a = QPushButton("打开前时相影像 (GeoTIFF)")
        self.btn_open_b = QPushButton("打开后时相影像 (变化检测用)")
        self.btn_demo = QPushButton("生成演示数据集（离线可用）")
        self.lbl_data = QLabel("未加载数据")
        self.lbl_data.setWordWrap(True)
        self.lbl_data.setStyleSheet("color:#5a6473;font-size:11px;")
        for b in (self.btn_open_a, self.btn_open_b, self.btn_demo):
            gl.addWidget(b)
        gl.addWidget(self.lbl_data)
        lay.addWidget(g)

        g2 = QGroupBox("② 智能解译")
        g2l = QVBoxLayout(g2)
        self.btn_seg = QPushButton("语义分割（建筑/道路/植被/水体）")
        self.btn_det = QPushButton("目标检测（DOTA 航拍 15 类）")
        self.btn_chg = QPushButton("变化检测（双时相）")
        g2l.addWidget(self.btn_seg)
        g2l.addWidget(self.btn_det)
        g2l.addWidget(self.btn_chg)
        g2l.addWidget(QLabel("图层显示"))
        self.chk_seg = QCheckBox("分割结果"); self.chk_seg.setChecked(True)
        self.chk_det = QCheckBox("检测框"); self.chk_det.setChecked(True)
        self.chk_chg = QCheckBox("变化图斑"); self.chk_chg.setChecked(True)
        for c in (self.chk_seg, self.chk_det, self.chk_chg):
            g2l.addWidget(c)
        row = QHBoxLayout()
        row.addWidget(QLabel("透明度"))
        self.sld_alpha = QSlider(Qt.Horizontal)
        self.sld_alpha.setRange(10, 100)
        self.sld_alpha.setValue(60)
        row.addWidget(self.sld_alpha)
        g2l.addLayout(row)
        lay.addWidget(g2)

        g3 = QGroupBox("③ 三维重建（3D 高斯泼溅）")
        g3l = QVBoxLayout(g3)
        self.cmb_backend = QComboBox()
        self._reload_backends()
        g3l.addWidget(QLabel("重建后端（现成的 3DGS 实现）"))
        g3l.addWidget(self.cmb_backend)
        self.btn_dl_3d = QPushButton("① 下载多视角影像（公开数据集）")
        self.btn_recon_demo = QPushButton("② 用内置合成场景重建（流程演示）")
        self.btn_recon_colmap = QPushButton("② 用 COLMAP 结果重建…（真实影像）")
        self.btn_open_ply = QPushButton("打开已有 .ply 模型")
        for b in (self.btn_dl_3d, self.btn_recon_demo, self.btn_recon_colmap, self.btn_open_ply):
            g3l.addWidget(b)
        form = QFormLayout()
        self.sp_iter = QSpinBox(); self.sp_iter.setRange(50, 5000)
        self.sp_iter.setValue(int(self.cfg.get_path("gsplat.iterations", 700)))
        self.sp_iter.setSingleStep(100)
        self.sp_maxg = QSpinBox(); self.sp_maxg.setRange(2000, 1000000)
        self.sp_maxg.setValue(int(self.cfg.get_path("gsplat.max_gaussians", 100000)))
        self.sp_maxg.setSingleStep(10000)
        self.cmb_sh = QComboBox(); self.cmb_sh.addItems(["0（最简单）", "1（推荐）", "2", "3（最精细）"])
        self.cmb_sh.setCurrentIndex(int(self.cfg.get_path("gsplat.sh_degree", 1)))
        form.addRow("迭代次数", self.sp_iter)
        form.addRow("高斯上限", self.sp_maxg)
        form.addRow("球谐阶数", self.cmb_sh)
        g3l.addLayout(form)
        lay.addWidget(g3)

        g4 = QGroupBox("④ 地理配准与任务")
        g4l = QVBoxLayout(g4)
        self.btn_reg = QPushButton("三维模型 ↔ 卫星底图 自动配准")
        self.btn_uav = QPushButton("为选中图斑生成航摄任务")
        self.btn_report = QPushButton("生成结构化分析报告")
        self.btn_geojson = QPushButton("导出 GeoJSON 成果")
        for b in (self.btn_reg, self.btn_uav, self.btn_report, self.btn_geojson):
            g4l.addWidget(b)
        self.lbl_reg = QLabel("未配准")
        self.lbl_reg.setWordWrap(True)
        self.lbl_reg.setStyleSheet("color:#5a6473;font-size:11px;")
        g4l.addWidget(self.lbl_reg)
        lay.addWidget(g4)
        lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(400)

        # 事件
        self.btn_open_a.clicked.connect(lambda: self.open_raster("a"))
        self.btn_open_b.clicked.connect(lambda: self.open_raster("b"))
        self.btn_demo.clicked.connect(self.generate_demo)
        self.btn_seg.clicked.connect(lambda: self.run_task("segmentation"))
        self.btn_det.clicked.connect(lambda: self.run_task("detection"))
        self.btn_chg.clicked.connect(lambda: self.run_task("change_detection"))
        self.btn_dl_3d.clicked.connect(self.download_multiview)
        self.btn_recon_demo.clicked.connect(lambda: self.run_reconstruction(demo=True))
        self.btn_recon_colmap.clicked.connect(lambda: self.run_reconstruction(demo=False))
        self.btn_open_ply.clicked.connect(self.open_ply)
        self.btn_reg.clicked.connect(self.run_registration)
        self.btn_uav.clicked.connect(self.make_uav_task)
        self.btn_report.clicked.connect(self.generate_report)
        self.btn_geojson.clicked.connect(self.export_geojson)
        for c in (self.chk_seg, self.chk_det, self.chk_chg):
            c.stateChanged.connect(self._refresh_layers)
        self.sld_alpha.valueChanged.connect(lambda _: self._refresh_layers())
        return scroll

    # ---------------------------------------------------------------- 中栏
    def _build_center(self) -> QWidget:
        self.map = MapView()
        self.map.clicked.connect(self.on_map_clicked)
        self.map.status.connect(lambda s: self.statusBar().showMessage(s))
        return self.map

    # ---------------------------------------------------------------- 右栏
    def _build_right(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 8, 8, 8)
        tabs = QTabWidget()
        self.view3d = SplatView()
        tabs.addTab(self.view3d, "三维视图")
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["对象", "类型", "面积(m²)", "中心经纬度"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setFont(QFont("Microsoft YaHei", 9))
        self.table.itemSelectionChanged.connect(self.on_table_select)
        tabs.addTab(self.table, "解译结果清单")
        self.info = QTextBrowser()
        self.info.setFont(QFont("Microsoft YaHei", 10))
        self.info.setHtml("<p style='color:#666'>运行解译任务后，这里显示模型、耗时与统计信息。</p>")
        tabs.addTab(self.info, "结果详情")
        tabs.setCurrentIndex(1)
        lay.addWidget(tabs)
        return w

    # ---------------------------------------------------------------- 菜单
    def _build_menu(self) -> None:
        mb = self.menuBar()
        m = mb.addMenu("文件(&F)")
        m.addAction("打开前时相影像…", lambda: self.open_raster("a"))
        m.addAction("打开后时相影像…", lambda: self.open_raster("b"))
        m.addAction("生成演示数据集", self.generate_demo)
        m.addSeparator()
        m.addAction("导出 GeoJSON 成果", self.export_geojson)
        m.addAction("打开输出目录", lambda: os.startfile(str(self.out_dir)))
        m.addSeparator()
        m.addAction("退出", self.close)

        m = mb.addMenu("解译(&I)")
        m.addAction("语义分割", lambda: self.run_task("segmentation"))
        m.addAction("目标检测", lambda: self.run_task("detection"))
        m.addAction("变化检测", lambda: self.run_task("change_detection"))
        m.addSeparator()
        m.addAction("预下载模型权重", self.download_weights)
        m.addAction("查看模型状态", self._report_model_status)

        m = mb.addMenu("三维(&3)")
        m.addAction("生成演示场景并重建", lambda: self.run_reconstruction(demo=True))
        m.addAction("用 COLMAP 结果重建…", lambda: self.run_reconstruction(demo=False))
        m.addAction("打开已有 .ply 模型…", self.open_ply)
        m.addAction("重置视角", self.view3d.reset_view)

        m = mb.addMenu("成果(&R)")
        m.addAction("生成结构化分析报告", self.generate_report)
        m.addAction("生成无人机航摄任务", self.make_uav_task)
        m.addAction("自动地理配准", self.run_registration)

        m = mb.addMenu("帮助(&H)")
        m.addAction("关于", self.about)

    # ==================================================================== 工具
    def log(self, text: str) -> None:
        self.logbox.appendPlainText(text)
        LOG.info(text)

    def _device_text(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return f"CUDA · {torch.cuda.get_device_name(0)}"
            return "CPU"
        except Exception:  # noqa: BLE001
            return "CPU"

    def busy(self, on: bool, text: str = "") -> None:
        for b in (self.btn_open_a, self.btn_open_b, self.btn_demo, self.btn_seg, self.btn_det,
                  self.btn_chg, self.btn_recon_demo, self.btn_recon_colmap, self.btn_open_ply,
                  self.btn_reg, self.btn_uav, self.btn_report, self.btn_geojson):
            b.setEnabled(not on)
        self.progress.setRange(0, 0) if on else self.progress.setRange(0, 100)
        if on:
            self.statusBar().showMessage(text or "处理中…")
        else:
            self.progress.setValue(0)
            self.statusBar().showMessage("就绪")

    def start_worker(self, fn, *args, want_progress: bool = False, on_ok=None,
                     busy_text: str = "", **kwargs) -> None:
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "提示", "已有任务在运行，请稍候")
            return
        self.busy(True, busy_text)
        self.worker = TaskWorker(fn, *args, want_progress=want_progress, **kwargs)
        self.worker.progressed.connect(self._on_progress)
        self.worker.failed.connect(self._on_failed)
        if on_ok:
            self.worker.finished_ok.connect(on_ok)
        self.worker.finished_ok.connect(lambda *_: self.busy(False))
        self.worker.finished.connect(lambda: self.busy(False))
        self.worker.start()

    def _on_progress(self, pct: int, text: str) -> None:
        if pct >= 0:
            self.progress.setRange(0, 100)
            self.progress.setValue(pct)
        self.statusBar().showMessage(text)

    def _on_failed(self, msg: str) -> None:
        self.log("!! 任务失败：" + msg.split("\n")[0])
        self.logbox.appendPlainText(msg)
        QMessageBox.critical(self, "任务失败", msg.split("\n\n")[0])

    # ==================================================================== 数据
    def open_raster(self, which: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择遥感影像", "", "栅格影像 (*.tif *.tiff *.img *.jp2 *.png *.jpg);;所有文件 (*)")
        if not path:
            return
        try:
            r = Raster.open(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "打开失败", str(e))
            return
        if which == "a":
            self.raster_a = r
        else:
            self.raster_b = r
        self.log(f"已载入{'前' if which == 'a' else '后'}时相：{path}　"
                 f"{r.width}×{r.height}×{r.info.count}　分辨率 {r.pixel_size_m()[0]:.2f} m/px")
        self._show_raster()
        self._update_data_label()

    def _show_raster(self) -> None:
        if self.raster_a is None:
            return
        self.map.set_base(self.raster_a.rgb, self.raster_a.pixel_size_m())

    def _update_data_label(self) -> None:
        parts = []
        if self.raster_a is not None:
            b = self.raster_a.bounds_lonlat()
            parts.append(f"前时相 {self.raster_a.width}×{self.raster_a.height}，"
                         f"{self.raster_a.pixel_size_m()[0]:.2f} m/px")
            parts.append(f"范围 {b[0]:.5f}~{b[2]:.5f}E, {b[1]:.5f}~{b[3]:.5f}N")
        if self.raster_b is not None:
            parts.append(f"后时相 {self.raster_b.width}×{self.raster_b.height}")
        self.lbl_data.setText("\n".join(parts) or "未加载数据")

    def generate_demo(self) -> None:
        out = self.out_dir / "demo"
        out.mkdir(parents=True, exist_ok=True)
        a = out / "satellite_before.tif"
        b = out / "satellite_after.tif"
        make_demo_geotiff(a, size=(2048, 1536), seed=2026, with_change=False)
        make_demo_geotiff(b, size=(2048, 1536), seed=2026, with_change=True)
        self.raster_a = Raster.open(a)
        self.raster_b = Raster.open(b)
        self._show_raster()
        self._update_data_label()
        self.log(f"已生成演示数据集：{a} 与 {b}（同一区域的前/后时相，后时相新增了若干建筑）")

    # ==================================================================== 解译
    def run_task(self, task: str) -> None:
        if self.raster_a is None:
            QMessageBox.information(self, "提示", "请先加载影像或生成演示数据集")
            return
        if task == "change_detection" and self.raster_b is None:
            QMessageBox.information(self, "提示", "变化检测需要前、后两个时相影像")
            return
        self.log(f"开始执行：{task} …")

        def work(progress=None):
            if task == "segmentation":
                return self.pipe.segment(self.raster_a, progress=self._wrap(progress))
            if task == "detection":
                return self.pipe.detect(self.raster_a, progress=self._wrap(progress))
            return self.pipe.change(self.raster_a, self.raster_b, progress=self._wrap(progress))

        self.start_worker(work, want_progress=True, busy_text=f"{task} 推理中…",
                          on_ok=self._on_task_done)

    @staticmethod
    def _wrap(progress):
        if progress is None:
            return None
        return lambda cur, total: progress(cur, total)

    def _on_task_done(self, res) -> None:
        self.results[res.task] = res
        self.log(f"完成 {res.task}：模型 {res.model_name}"
                 f"{'（已降级）' if res.fallback else ''}，耗时 {res.elapsed:.1f}s")
        for k, v in res.summary().items():
            self.log(f"    {k}: {v}")
        self._refresh_layers()
        self._refresh_table()
        self._refresh_info()
        self.statusBar().showMessage("解译完成")

    # ---------------------------------------------------------------- 图层
    def _refresh_layers(self) -> None:
        alpha = self.sld_alpha.value() / 100.0
        seg = self.results.get("segmentation")
        if seg is not None and self.chk_seg.isChecked():
            rgba = MapView.mask_to_rgba(seg.mask, alpha=190)
            self.map.set_layer("seg", rgba, True, alpha, order=1)
        else:
            self.map.set_layer("seg", None)

        chg = self.results.get("change_detection")
        if chg is not None and self.chk_chg.isChecked() and chg.prob is not None:
            rgba = MapView.prob_to_rgba(chg.prob, (255, 60, 60), 170)
            self.map.set_layer("chg", rgba, True, alpha, order=2)
        else:
            self.map.set_layer("chg", None)

        det = self.results.get("detection")
        if det is not None:
            boxes = []
            for d in det.detections:
                if d.obb:
                    pts = d.box[:8].reshape(4, 2)
                else:
                    x1, y1, x2, y2 = d.box[:4]
                    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                boxes.append({"pts": pts.tolist(), "label": f"{d.label} {d.score:.2f}",
                              "color": (0, 235, 120)})
            self.map.set_boxes(boxes if self.chk_det.isChecked() else [])

        self._build_change_items()
        self.map.set_polygons([{"id": i, "pts": it["pts"], "color": (255, 200, 60)}
                               for i, it in enumerate(self.change_items)]
                              if self.chk_chg.isChecked() else [])

    def _build_change_items(self) -> None:
        self.change_items = []
        chg = self.results.get("change_detection")
        if chg is None or self.raster_a is None:
            return
        for i, f in enumerate(chg.features):
            coords = f["geometry"]["coordinates"][0]
            lon = np.array([c[0] for c in coords]); lat = np.array([c[1] for c in coords])
            x, y = self.raster_a.lonlat_to_pixel(lon, lat)
            self.change_items.append({
                "id": i, "pts": list(zip(np.ravel(x).tolist(), np.ravel(y).tolist())),
                "lon": float(lon.mean()), "lat": float(lat.mean()),
                "area": float(f["properties"].get("area_px", 0)) *
                        self.raster_a.pixel_size_m()[0] * self.raster_a.pixel_size_m()[1],
                "type": f["properties"].get("change_type", "变化"), "feature": f,
            })

    def _refresh_table(self) -> None:
        rows = []
        for item in self.change_items:
            rows.append((f"变化图斑 #{item['id'] + 1}", item["type"],
                         f"{item['area']:,.1f}", f"{item['lon']:.6f}, {item['lat']:.6f}"))
        det = self.results.get("detection")
        if det is not None:
            from ..pipeline import center_of
            for i, d in enumerate(det.detections[:500]):
                try:
                    lon, lat = center_of(det.features[i])
                except Exception:  # noqa: BLE001
                    lon = lat = float("nan")
                rows.append((f"目标 #{i + 1} {d.label}", d.label, "-", f"{lon:.6f}, {lat:.6f}"))
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                self.table.setItem(r, c, QTableWidgetItem(str(val)))
        self.table.resizeColumnsToContents()

    def _refresh_info(self) -> None:
        html = ["<h3>模型与结果</h3><table border='1' cellspacing='0' cellpadding='4' "
                "style='border-collapse:collapse;font-size:12px'>",
                "<tr><th>任务</th><th>模型</th><th>状态</th><th>耗时(s)</th></tr>"]
        names = {"segmentation": "语义分割", "detection": "目标检测",
                 "change_detection": "变化检测"}
        for t, cn in names.items():
            r = self.results.get(t)
            if r is None:
                html.append(f"<tr><td>{cn}</td><td>-</td><td>未运行</td><td>-</td></tr>")
            else:
                html.append(f"<tr><td>{cn}</td><td>{r.model_name}</td>"
                            f"<td>{'降级' if r.fallback else '正常'}</td>"
                            f"<td>{r.elapsed:.1f}</td></tr>")
        html.append("</table><h4>统计明细</h4><pre style='font-size:11px'>")
        for t, r in self.results.items():
            html.append(f"{t}:\n")
            for k, v in r.summary().items():
                html.append(f"  {k} = {v}\n")
        html.append("</pre>")
        self.info.setHtml("".join(html))

    # ---------------------------------------------------------------- 交互
    def on_map_clicked(self, x: float, y: float) -> None:
        if not self.change_items:
            return
        best, bd = None, 1e18
        for it in self.change_items:
            pts = np.asarray(it["pts"])
            d = float(((pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2).min())
            if d < bd:
                best, bd = it, d
        if best is None or bd > 60:
            return
        self.map.select(best)
        self.log(f"选中变化图斑 #{best['id'] + 1}：{best['type']}，"
                 f"面积 {best['area']:.1f} m²，中心 {best['lon']:.6f}E {best['lat']:.6f}N")
        self._jump_to_3d(best)

    def on_table_select(self) -> None:
        for idx in self.table.selectionModel().selectedRows():
            r = idx.row()
            if r < len(self.change_items):
                item = self.change_items[r]
                self.map.select(item)
                self._jump_to_3d(item)
            break

    def _jump_to_3d(self, item: dict) -> None:
        """点击二维图斑 -> 三维视图高亮对应位置（2D-3D 联动）。"""
        if self.recon is None or self.georef is None:
            return
        try:
            x, y = self.georef.lonlat_to_local(item["lon"], item["lat"])
            self.view3d.set_highlight(np.array([[float(np.ravel(x)[0]),
                                                 float(np.ravel(y)[0]), 0.0]]))
            self.log("已联动三维视图：高亮该图斑在地理配准后的模型位置")
        except Exception as e:  # noqa: BLE001
            LOG.warning("三维联动失败：%s", e)

    # ==================================================================== 三维
    def _reload_backends(self) -> None:
        """把可用的 3DGS 后端填进下拉框（auto = 按优先级自动挑）。"""
        self.cmb_backend.clear()
        self.cmb_backend.addItem("自动选择（推荐）", None)
        try:
            from ..gsbackend.registry import detect_backends
            for d in detect_backends(self.cfg, verbose=False):
                mark = "✓" if d["available"] else "×"
                self.cmb_backend.addItem(f"{mark} {d['name']} — {d['title']}", d["name"])
        except Exception as e:  # noqa: BLE001
            LOG.warning("后端探测失败：%s", e)

    def download_multiview(self) -> None:
        """下载公开的多视角影像数据集，供 3DGS 重建使用。"""
        items = ["lego（NeRF 合成，43MB，自带位姿）",
                 "ship（NeRF 合成，自带位姿）",
                 "drums（NeRF 合成，自带位姿）",
                 "south-building（真实照片 + COLMAP 位姿，约 270MB）",
                 "bigsur（真实航拍照片，需自己跑 SfM）"]
        keys = ["lego", "ship", "drums", "south-building", "bigsur"]
        from PyQt5.QtWidgets import QInputDialog
        choice, ok = QInputDialog.getItem(self, "下载多视角影像", "选择数据集：", items, 0, False)
        if not ok:
            return
        name = keys[items.index(choice)]
        out = Path(self.cfg.get_path("paths.output", "outputs")).parent / "data"

        def work(progress=None):
            from ..gsbackend.data import fetch_preset
            kw = {"max_images": 40, "resize": 1200} if name in ("south-building", "bigsur") else {}
            if name == "bigsur":
                kw["consecutive"] = True
                kw["max_images"] = 30
            return fetch_preset(name, out, progress=progress, **kw)

        self.start_worker(work, want_progress=True, busy_text=f"下载 {name} …",
                          on_ok=self._on_multiview_downloaded)

    def _on_multiview_downloaded(self, info: dict) -> None:
        self._mv_info = info
        self.log(f"多视角影像已就绪：{info['images_dir']}（{info['files']} 个文件）")
        if info.get("sparse_dir"):
            self.log(f"    相机位姿：{info['sparse_dir']}")
            self._mv_ready = (info["images_dir"], info["sparse_dir"])
        else:
            self.log("    该数据集只有照片，请先跑 SfM 计算位姿："
                     f"python tools\\run_sfm.py --images \"{info['images_dir']}\" "
                     f"--out \"{Path(info['root']) / 'colmap'}\"")
            self._mv_ready = None

    def run_reconstruction(self, demo: bool = True) -> None:
        gs = dict(self.cfg.get_path("gsplat", {}))
        gs["iterations"] = self.sp_iter.value()
        gs["max_gaussians"] = self.sp_maxg.value()
        gs["sh_degree"] = self.cmb_sh.currentIndex()
        backend = self.cmb_backend.currentData()
        out = self.out_dir / ("3dgs_demo" if demo else "3dgs_colmap")
        images_dir = sparse_dir = None
        if not demo:
            ready = getattr(self, "_mv_ready", None)
            if ready:
                images_dir, sparse_dir = ready
                self.log(f"使用已下载的多视角数据：{images_dir}")
            else:
                images_dir = QFileDialog.getExistingDirectory(self, "选择影像目录（多视角照片）")
                if not images_dir:
                    return
                sparse_dir = QFileDialog.getExistingDirectory(
                    self, "选择 COLMAP 稀疏模型目录（含 cameras.bin/txt）")
                if not sparse_dir:
                    return
        self.log(f"开始三维重建：后端={backend or '自动'}，迭代 {gs['iterations']}，"
                 f"高斯上限 {gs['max_gaussians']}，球谐阶数 {gs['sh_degree']}")

        def work(progress=None):
            return reconstruct(out, gs, device=self._torch_device(), images_dir=images_dir,
                               sparse_dir=sparse_dir, iterations=gs["iterations"],
                               progress=progress, backend=backend, app_cfg=self.cfg)

        self.start_worker(work, want_progress=True, busy_text="3D 高斯泼溅重建中…",
                          on_ok=self._on_recon_done)

    def _torch_device(self) -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"

    def _on_recon_done(self, res: ReconstructionResult) -> None:
        self.recon = res
        tag = "（内置兜底实现，非 3DGS 主路径）" if not res.is_real_backend else ""
        self.log(f"三维重建完成：后端 {res.backend} {tag}")
        self.log(f"    {res.toolkit}；{res.num_gaussians:,} 个高斯；"
                 f"PSNR {res.psnr if res.psnr is None else round(res.psnr, 2)} dB；"
                 f"SSIM {res.ssim if res.ssim is None else round(res.ssim, 4)}；"
                 f"耗时 {res.seconds:.1f}s")
        self.log(f"模型文件：{res.ply}")
        if res.log_file:
            self.log(f"训练日志：{res.log_file}")
        try:
            self.view3d.set_renderer(load_renderer(res.ply))
            self.view3d.reset_view()
            self.log("三维视图已载入模型，可拖拽旋转 / 滚轮缩放")
        except Exception as e:  # noqa: BLE001
            self.log(f"三维视图载入失败：{e}")

    def open_ply(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开 3DGS 模型", str(self.out_dir),
                                              "3D 高斯模型 (*.ply)")
        if not path:
            return
        try:
            self.view3d.set_renderer(load_renderer(path))
            self.view3d.reset_view()
            self.log(f"已载入三维模型：{path}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "载入失败", str(e))

    # ==================================================================== 配准
    def run_registration(self) -> None:
        if self.recon is None or self.raster_a is None:
            QMessageBox.information(self, "提示", "需要先完成三维重建并加载卫星影像")
            return
        from ..gsplat.ply_io import read_ply
        d = read_ply(self.recon.ply)
        means = np.stack([d["x"], d["y"], d["z"]], axis=1)
        ref = None
        seg = self.results.get("segmentation")
        if seg is not None and seg.mask is not None:
            ref = (seg.mask == 1).astype(np.uint8)
        out_png = self.out_dir / "registration_overlay.png"

        def work(progress=None):
            return register_to_raster(means, self.raster_a, ref_mask=ref,
                                      overlay_path=str(out_png), progress=progress)

        self.start_worker(work, want_progress=True, busy_text="地理配准中…",
                          on_ok=self._on_reg_done)

    def _on_reg_done(self, res) -> None:
        self.georef = res.georef
        self.lbl_reg.setText(res.georef.describe())
        self.log("地理配准完成：" + res.georef.describe())
        self.log(f"    轮廓重合度 IoU={res.iou:.4f}，几何偏差 RMSE={res.rmse_m:.3f} m")
        if res.overlay:
            self.log(f"    配准核验图：{res.overlay}")
        self._reg_result = res

    # ==================================================================== 任务/报告
    def make_uav_task(self) -> None:
        item = self.map.selected or (self.change_items[0] if self.change_items else None)
        if item is None:
            QMessageBox.information(self, "提示", "请先在图上点击一个变化图斑")
            return
        pts = item["feature"]["geometry"]["coordinates"][0]
        cfg = dict(self.cfg.get_path("uav_task", {}))
        task = plan_flight(pts, cfg)
        self._uav_tasks = getattr(self, "_uav_tasks", [])
        self._uav_tasks.append({"task": task, "item": item})
        lines = task.summary_lines()
        self.log("已生成无人机航摄任务建议：")
        for l in lines:
            self.log("    " + l)
        p = self.out_dir / f"uav_task_{len(self._uav_tasks)}.geojson"
        save_geojson(p, task.geojson())
        self.log(f"    航线文件：{p}")
        QMessageBox.information(self, "无人机航摄任务建议", "\n".join(lines))

    def generate_report(self) -> None:
        if not self.results and self.recon is None:
            QMessageBox.information(self, "提示", "请先运行至少一个解译任务或三维重建")
            return
        ctx = self._build_context()
        paths = export_report(ctx, self.out_dir)
        self.log("分析报告已生成：")
        for k, v in paths.items():
            self.log(f"    {k.upper()}: {v}")
        if "html" in paths:
            try:
                os.startfile(paths["html"])
            except Exception:  # noqa: BLE001
                pass

    def _build_context(self) -> ReportContext:
        meta = []
        for name, r in (("前时相", self.raster_a), ("后时相", self.raster_b)):
            if r is None:
                continue
            b = r.bounds_lonlat()
            meta.append({"name": name, "size": f"{r.width}×{r.height}", "count": r.info.count,
                         "res": f"{r.pixel_size_m()[0]:.2f}",
                         "crs": r.info.crs or "无(占位)", 
                         "bounds": f"{b[0]:.5f}~{b[2]:.5f}E, {b[1]:.5f}~{b[3]:.5f}N"})
        tasks = []
        for r in self.results.values():
            s = r.summary()
            s["model"] = r.model_name
            tasks.append(s)
        changes = [{"type": it["type"], "area_m2": it["area"], "center_lon": it["lon"],
                    "center_lat": it["lat"],
                    "advice": "现场核查并比对规划许可" if it["type"] == "新增" else "核实拆除/损毁情况"}
                   for it in self.change_items]
        uav = []
        for i, u in enumerate(getattr(self, "_uav_tasks", []), 1):
            t = u["task"]
            rows = [(l.split("：")[0], "：".join(l.split("：")[1:])) for l in t.summary_lines()]
            uav.append({"rows": rows, "notes": t.notes})
        rec = None
        if self.recon is not None:
            rec = self.recon.summary()
            rec["previews"] = [(Path(p).name, f"图　三维高斯模型环绕视图 {i + 1}")
                               for i, p in enumerate(self.recon.previews)]
        reg = None
        res = getattr(self, "_reg_result", None)
        if res is not None:
            reg = res.summary()
        images = []
        seg = self.results.get("segmentation")
        if seg is not None and self.raster_a is not None:
            from ..utils.vis import colorize_mask
            import cv2
            cv2.imwrite(str(self.out_dir / "fig_segmentation.png"),
                        colorize_mask(seg.mask)[:, :, ::-1])
            images.append(("fig_segmentation.png", "图　语义分割结果（地物分类图）"))
        det = self.results.get("detection")
        if det is not None and self.raster_a is not None:
            from ..utils.vis import draw_boxes
            import cv2
            img = draw_boxes(self.raster_a.rgb, [d.box for d in det.detections],
                             [d.label for d in det.detections], [d.score for d in det.detections],
                             obb=any(d.obb for d in det.detections))
            cv2.imwrite(str(self.out_dir / "fig_detection.png"), img[:, :, ::-1])
            images.append(("fig_detection.png", "图　目标检测结果（DOTA 航拍目标）"))
        chg = self.results.get("change_detection")
        if chg is not None and self.raster_a is not None:
            from ..utils.geo import overlay_mask
            import cv2
            cv2.imwrite(str(self.out_dir / "fig_change.png"),
                        overlay_mask(self.raster_a.rgb, chg.mask)[:, :, ::-1])
            images.append(("fig_change.png", "图　变化检测结果（红色为检出的变化区域）"))
        if res is not None and res.overlay:
            images.append((Path(res.overlay).name, "图　三维模型与卫星底图配准核验"))
        conclusion = [
            "系统完成了从多光谱遥感影像输入、三大深度学习解译、三维高斯泼溅重建到"
            "结构化报告输出的端到端流程。",
            f"本次共检出变化图斑 {len(changes)} 个"
            + (f"，合计变化面积 {sum(c['area_m2'] for c in changes):,.0f} m²" if changes else "") + "。",
            "所有解译结果均已通过 GDAL 仿射变换映射至 WGS84 经纬度，可直接用于无人机航摄"
            "与现场核查。",
            "建议对新增建筑类图斑优先开展现场核查，核实是否符合规划许可；"
            "对拆除/损毁类图斑核实是否需要修复或重建。",
        ]
        return ReportContext(
            area_name="哈尔滨市典型城市更新场景（演示区）",
            backend=self._device_text(),
            rasters={"前时相": str(self.raster_a.info.path) if self.raster_a else "",
                     "后时相": str(self.raster_b.info.path) if self.raster_b else ""},
            raster_meta=meta, tasks=tasks, changes=changes, uav_tasks=uav,
            reconstruction=rec, registration=reg, models=self.pipe.hub.status(),
            images=images, conclusion=conclusion)

    def export_geojson(self) -> None:
        if not self.results:
            QMessageBox.information(self, "提示", "还没有可导出的解译结果")
            return
        paths = []
        for task, res in self.results.items():
            p = self.out_dir / f"{task}.geojson"
            paths.append(self.pipe.save_geojson(res, p))
        self.log("已导出 GeoJSON：")
        for p in paths:
            self.log("    " + p)

    # ==================================================================== 其他
    def download_weights(self) -> None:
        def work(progress=None):
            if progress:
                progress(0, 3)
            progress(1, "拉取建筑分割权重…") if progress else None
            seg = ensure_building_segmentation(self.cfg.get_path("paths.weights"))
            if progress:
                progress(1, "下载 DOTA 检测权重…")
            det = ensure_dota_obb(self.cfg.get_path("paths.weights"))
            return {"segmentation": seg, "detection": det}

        self.start_worker(work, want_progress=True, busy_text="下载模型权重…",
                          on_ok=self._on_weights)

    def _on_weights(self, res) -> None:
        for k, v in res.items():
            self.log(f"    {k}: {v or '失败（将使用降级方案）'}")
        QMessageBox.information(self, "完成", "权重检查完成，详见日志")

    def _report_model_status(self) -> None:
        st = self.pipe.hub.status()
        self.log("模型状态：" + "；".join(
            f"{k}={v['name'] or '未载入'}{'(降级)' if v['fallback'] else ''}"
            for k, v in st.items()))
        ws = list_local_weights(self.cfg.get_path("paths.weights"))
        self.log("本地权重：" + (", ".join(f"{w['name']}({w['size_mb']}MB)" for w in ws) or "无"))

    def about(self) -> None:
        QMessageBox.about(self, "关于", (
            f"<b>{self.cfg.get_path('app.title')}</b><br>版本 {self.cfg.get_path('app.version')}"
            "<br><br>哈尔滨工业大学 大学生创新训练计划项目<br>"
            "<br>语义分割：YOLOv8-seg 卫星影像建筑提取 + 光谱指数<br>"
            "目标检测：YOLO-OBB（DOTAv1 航拍 15 类）<br>"
            "变化检测：建筑掩膜差分 / SiamUnet-diff<br>"
            "三维重建：自研可微分 3D 高斯泼溅光栅化器"))
