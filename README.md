# aerosky —— 空天协同的多光谱遥感影像智能解译与三维监测系统

面向城市更新的「深度学习驱动筛查 — 无人机定向详查 — 三维交互验证 — 智能报告生成」闭环系统。
哈尔滨工业大学大学生创新训练计划项目。

```
卫星影像 (GeoTIFF)
   │
   ├─► ① 智能解译后端   语义分割 / 目标检测 / 变化检测（分块切割—并行推理—无缝拼接）
   │        │
   │        ├─► GDAL 仿射变换 → WGS84 经纬度 → GeoJSON 图斑 / 检测框
   │        │
   ├─► ② 空天协同联动   点击图斑 → 中心坐标 + 覆盖范围 → 无人机航摄任务建议（航高/航带/像片数/航线）
   │
   ├─► ③ 三维高斯泼溅   多视角影像 → COLMAP(SfM) → 3DGS 优化 → 轻量化 .ply
   │        │
   │        └─► 与卫星底图自动地理配准（相似变换 + 轮廓 IoU 最优化 + RMSE 评价）
   │
   └─► ④ 结构化报告     地理位置 · 变化类型 · 量化指标 · 趋势研判 · 处置建议 → HTML / PDF
```

---

## 1. 快速开始

```powershell
cd D:\Code\dc\aerosky

# ① 建环境（约 5~10 分钟，国内镜像；默认 CPU 版 PyTorch）
python -m venv .venv
powershell -ExecutionPolicy Bypass -File tools\setup_env.ps1

# ② 环境自检
.venv\Scripts\python run_app.py --check

# ③ 一条命令跑完整个端到端流程（无需任何外部数据，结果落在 outputs\demo_run）
.venv\Scripts\python run_app.py --demo

# ④ 启动图形界面
.venv\Scripts\python run_app.py
```

> **想要 GPU 加速？** 机器上的 RTX 4050 可用，但 CUDA 版 PyTorch 约 2.4 GB，
> 国内直连较慢。网络条件允许时执行 `tools\upgrade_to_cuda.ps1`（支持断点续传），
> 之后系统会自动使用 `cuda`，三维重建速度可提升一个数量级。

---

## 2. 界面功能

| 区域 | 功能 |
|---|---|
| 左栏 ① 数据 | 打开 GeoTIFF 前/后时相；一键生成演示数据集 |
| 左栏 ② 智能解译 | 语义分割 / 目标检测 / 变化检测三个按钮 + 图层开关 + 透明度 |
| 左栏 ③ 三维重建 | 演示场景重建 / COLMAP 结果重建 / 打开 .ply；迭代次数、高斯上限、球谐阶数可调 |
| 左栏 ④ 配准与成果 | 自动地理配准、生成无人机航摄任务、生成分析报告、导出 GeoJSON |
| 中间 | 二维地图：滚轮缩放、拖动平移、双击放大，点击图斑触发 2D→3D 联动 |
| 右侧 | 三维高斯泼溅实时浏览（拖拽旋转 / 滚轮缩放 / 右键平移）、解译结果清单、结果详情 |
| 底部 | 进度条与运行日志 |

---

## 3. 三大智能解译模型

| 任务 | 默认实现 | 权重 | 说明 |
|---|---|---|---|
| 语义分割 | **YOLOv8-seg 建筑实例分割** + 光谱指数合成 5 类 | `keremberke/yolov8s-building-segmentation`（23.8 MB，首次运行自动下载） | 真实卫星影像建筑提取；再结合 ExG 植被指数、NDWI 水体指数、灰度/纹理规则得到 建筑/道路/植被/水体/裸地 |
| 目标检测 | **YOLO11n-OBB（DOTAv1）** | `yolo11n-obb.pt`（约 6 MB，自动下载） | 航拍 15 类：飞机/船舶/储罐/棒球场/网球场/游泳池/田径场/港口/桥梁/大型车辆/小型车辆/直升机/环岛/足球场/篮球场 |
| 变化检测 | **建筑掩膜差分**（默认）/ SiamUnet-diff（可训练） | 复用建筑分割权重；SiamUnet 需自行训练 | 前后时相分别解译后取掩膜差集，直接对应「新增违建 / 拆除损毁」 |

三种任务都带**离线降级路径**（光谱+形态学、连通域、变化向量分析），
权重缺失或网络不通时系统仍可运行，不会因为下载失败而不可用。

**自己训练权重**（把配置里的 `backend` 改成对应的自训练后端即可启用）：

```powershell
# 语义分割：U-Net(ResNet34)，数据集目录 images/ + masks/（0/255）
.venv\Scripts\python tools\train_segmentation.py --data D:\data\buildings --epochs 30

# 变化检测：SiamUnet-diff，数据集目录 A/ + B/ + label/
.venv\Scripts\python tools\train_change_detection.py --data D:\data\WHU-CD --epochs 40
```

---

## 4. 三维高斯泼溅（对接现成实现）

三维重建**不自研算法**，而是对接业界成熟的 3DGS 实现，通过统一适配层
`aerosky/gsbackend/` 调用；系统负责数据准备、流程编排、地理配准与成果输出。

| 后端 | 实现 | 编译要求 | 本机 |
|---|---|---|---|
| `taichi` | [taichi_3d_gaussian_splatting](https://github.com/wanmeihuali/taichi_3d_gaussian_splatting)（纯 Python + Taichi，CUDA 后端） | 免编译 | ✅ 已跑通 |
| `gsplat` | [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) | 有 Windows 预编译轮子（仅 cp310） | 需 Python 3.10 |
| `official` | [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)（原论文官方实现） | 需 nvcc + MSVC | 本机无 MSVC |
| `opensplat` | [OpenSplat](https://github.com/pierotofy/OpenSplat) | 需 CMake | — |
| `builtin` | 系统内置极简实现（`aerosky/gsplat/`） | 免编译，CPU 可跑 | 无 CUDA 时的流程兜底，**非主路径** |

```powershell
python tools\run_3dgs.py --list                 # 看当前环境哪些后端可用
python tools\setup_3dgs.py --backend taichi     # 克隆/安装某个后端
python tools\download_3d_data.py --list         # 看有哪些公开多视角数据集
python tools\run_3dgs.py --data data\south-building --out outputs\3dgs --iters 900
```

详细说明（各后端差异、数据流、踩过的坑）见 **[3DGS后端说明.md](3DGS后端说明.md)**。

`aerosky/gsplat/` 里保留的纯 PyTorch 实现只在两种情况用：没有可用 CUDA 后端时把流程跑通，
以及给前端三维视图提供不依赖 torch/CUDA 的快速预览渲染器（`render_np.py`）。

---

## 5. 地理配准

`aerosky/registration/` 把三维模型的局部坐标系锚定到 WGS84：

1. 取模型**抬高表面（屋顶）**的俯视占据图作为模型侧轮廓；
2. 取卫星底图的建筑掩膜（可由本系统分割结果提供）作为参考侧轮廓；
3. 在 **(尺度 k, 朝向 θ)** 上做网格搜索，每个候选用 **FFT 互相关 + 积分图**求最优平移与 IoU；
4. 取最优候选做两轮细化，解出 `(origin_lon, origin_lat, meters_per_unit, heading)`；
5. 用**距离变换**统计模型轮廓到参考轮廓的残差，给出几何偏差 RMSE（米）。

配准结果落盘为 `georeference.json`，据此实现「点击二维图斑 → 跳转三维模型对应位置」。

---

## 6. 无人机航摄任务与报告

`aerosky/report/uav_task.py` 用标准航空摄影测量公式生成可直接执行的作业建议：

```
GSD = H · p / f          → 航高 H = GSD · f / p
航带间距 = GSD · 像幅宽(px) · (1 − 旁向重叠)
曝光间距 = GSD · 像幅高(px) · (1 − 航向重叠)
```

输出：目标区中心/面积、推荐航高、重叠度、航带数与像片数、航程与作业时间、
蛇形航线航点（GeoJSON LineString）。

`aerosky/report/builder.py` 生成结构化分析报告（HTML + PDF）：地理位置、变化类型、
量化指标、趋势研判、处置建议，并内嵌解译结果图与配准核验图。

---

## 7. 目录结构

```
aerosky/
├── run_app.py                 启动入口（--check / --demo / GUI）
├── configs/default.yaml       全部可调参数
├── requirements.txt
├── aerosky/
│   ├── config.py              配置加载（YAML + 默认值深合并）
│   ├── pipeline.py            端到端编排（影像 → 模型 → 地理坐标 → 图斑/报告）
│   ├── reconstruction.py      三维重建门面
│   ├── demo.py                无人值守端到端演示
│   ├── utils/                 地理坐标（GDAL/rasterio）、可视化、指标、权重下载
│   ├── inference/             模型基类、分块推理引擎、三大模型、注册表
│   ├── gsplat/                3D 高斯泼溅（光栅化 / 模型 / 训练 / COLMAP / PLY / 合成数据）
│   ├── registration/          局部坐标系 ↔ WGS84 与自动配准
│   ├── report/                无人机任务 + 结构化报告
│   └── ui/                    PyQt5 界面（地图视图 / 三维视图 / 后台线程 / 主窗口）
├── tools/                     环境搭建、演示数据、真实样本抓取、权重下载、训练脚本、自检
└── tests/                     离线自动化测试
```

---

## 8. 命令行速查

```powershell
python run_app.py --check                 # 环境自检
python run_app.py --demo                  # 端到端演示（真实卫星影像 + 三维重建 + 报告）
python tools\run_cli.py segment --image a.tif
python tools\run_cli.py detect  --image a.tif
python tools\run_cli.py change  --before a.tif --after b.tif
python tools\run_cli.py recon   --out outputs\3d
python tools\fetch_real_sample.py --out outputs\real --n 9     # 抓真实卫星影像样本
python tools\download_weights.py                                # 预下载模型权重
python tools\selftest_gs.py                                     # 三维重建自检
python tests\test_aerosky.py                                    # 自动化测试
```

---

## 9. 注意事项与已知限制

* **Windows 上 OpenCV 与 PyQt5 的 Qt 插件冲突**：`run_app.py` 已在导入 `cv2` 之前设置
  `QT_QPA_PLATFORM_PLUGIN_PATH`，请勿在别处先导入 `cv2`。
* **PyPI 上的 Windows PyTorch 是 CPU 版**，三维重建与推理在 CPU 上可用但较慢
  （160×120、8 千高斯约 0.5 s/迭代）；GPU 版见 `tools/upgrade_to_cuda.ps1`。
* **Ultralytics 采用 AGPL-3.0 许可**，DOTA 数据集为 research-only；若用于商业分发需替换。
* **WHU-CD 是变化检测数据集**，不含「建筑/道路/植被」多类标注，不能直接用于多类语义分割；
  本系统因此用卫星影像建筑分割权重 + 光谱指数合成地物类别。
* 演示数据若来自公开数据集（`keremberke/satellite-building-segmentation`），
  其地理参考是按 0.3 m/px、哈尔滨原位坐标**模拟赋值**的，仅用于流程演示。
* 系统输出为辅助决策信息，正式执法/修复决策仍需现场核实。
