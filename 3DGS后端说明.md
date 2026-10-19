# 3D 高斯泼溅后端说明

本系统的三维重建**不自研算法**，而是对接业界现成的 3D 高斯泼溅实现，
通过统一适配层（`aerosky/gsbackend/`）调用。这样：
算法质量由成熟实现保证，系统只负责数据准备、流程编排、地理配准与成果输出。

## 一、支持的后端

| 后端 | 实现 | 语言/依赖 | 是否需要编译 | 本机可用性 |
|---|---|---|---|---|
| `gsplat` | [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) | PyTorch + CUDA 扩展 | 有 Windows 预编译轮子（**仅 cp310**） | 未装（需 Python 3.10） |
| `official` | [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)（原论文官方实现） | PyTorch + CUDA 扩展 | 需要 nvcc + MSVC | 未装（本机无 MSVC） |
| `opensplat` | [pierotofy/OpenSplat](https://github.com/pierotofy/OpenSplat) | C++ / libtorch | 需要 CMake + 编译器 | 未装 |
| `taichi` | [wanmeihuali/taichi_3d_gaussian_splatting](https://github.com/wanmeihuali/taichi_3d_gaussian_splatting) | 纯 Python + Taichi 语言（CUDA 后端） | **不需要编译** | ✅ **已跑通** |
| `builtin` | 系统内置极简实现 | 纯 PyTorch，CPU 即可 | 不需要 | ✅（无 CUDA 时的流程兜底，非主路径） |

**本机实际使用 `taichi` 后端**：它把高斯光栅化用 Taichi 语言写成、由 Taichi 自己 JIT 到 CUDA，
不需要 MSVC / nvcc 编译 CUDA 扩展，因此在这台机器上可以直接跑真实 3DGS 训练。

## 二、环境

3DGS 后端跑在独立的 Python 环境里（系统的遥感部分在另一个 venv，互不影响）：

```
D:\Code\dc\gs3d_env\          ← 由已有的 rs_seg（PyTorch 2.7.1+cu118 + CUDA）派生
  torch 2.7.1+cu118  (CUDA 可用，RTX 4050)
  taichi 1.7.4
  dataclass-wizard 0.32.0     ← 必须 < 1.0，新版删掉了 YAMLWizard
D:\Code\dc\3dgs_repos\taichi_3d_gaussian_splatting\   ← 克隆的现成实现
```

系统的后端适配层会自动探测这些路径（`aerosky/gsbackend/registry.py`），
也可以用配置项或环境变量指定：

```yaml
# configs/default.yaml
gsbackend:
  backend: auto            # auto / gsplat / official / opensplat / taichi / builtin
  taichi_python: D:\Code\dc\gs3d_env\Scripts\python.exe
  taichi_repo: D:\Code\dc\3dgs_repos\taichi_3d_gaussian_splatting
```

## 三、怎么用

```powershell
# 看当前环境有哪些后端可用
python tools\run_3dgs.py --list

# 一键安装/克隆某个后端
python tools\setup_3dgs.py --backend gsplat --python C:\miniconda3\envs\gs310\python.exe
python tools\setup_3dgs.py --backend official
python tools\setup_3dgs.py --backend taichi

# 下载公开多视角影像（不用无人机）
python tools\download_3d_data.py lego                       # 自带位姿，43MB
python tools\download_3d_data.py south-building --max-images 48   # 真实照片 + 现成 COLMAP 位姿

# 只有照片时，自己算位姿（pycolmap，免装 COLMAP）
python tools\run_sfm.py --images data\bigsur\images --out data\bigsur\colmap --matcher sequential

# 三维重建
python tools\run_3dgs.py --data data\south-building --out outputs\3dgs --iters 1200 --downscale 2

# 查看结果
python tools\view_3dgs.py --model outputs\3dgs\run_taichi\point_cloud.ply
```

在图形界面里：左栏「③ 三维重建」→ 选后端 → 先「① 下载多视角影像」→ 再「② 用 COLMAP 结果重建」。

## 四、数据流

```
公开多视角影像（网上下载）或你自己的照片
        │
        ├─ 有现成位姿 → 直接用
        └─ 只有照片  → pycolmap 跑 SfM（特征提取 → 匹配 → 增量重建）
        │
        ▼
   COLMAP 稀疏模型（cameras / images / points3D）
        │
        ▼
   后端适配层 aerosky/gsbackend/
        │  taichi  → 转成 train.json/val.json/point_cloud.parquet + 配置 yaml → 训练
        │  gsplat  → 直接读 COLMAP 目录训练
        │  官方    → 直接读 COLMAP 目录训练
        ▼
   标准 3DGS .ply（可拖进 https://supersplat.dev 查看）
        │
        ├─ 前端三维视图实时浏览（内置 numpy 预览渲染器，只负责"看图"）
        └─ 与卫星影像自动地理配准 → 2D-3D 联动 → 写入分析报告
```

## 五、踩过的坑（都已修掉）

1. **Taichi 版模型只在验证时落盘**：`val-interval` 必须小于总迭代数，
   否则跑完一次模型都不会保存。适配层已自动设成 `iters/4`。
2. **`dataclass-wizard` 新版删掉了 `YAMLWizard`**，必须装 `<1.0`（本机用 0.32.0）。
3. **COLMAP 文本格式的相机模型是字符串**（`SIMPLE_RADIAL`），二进制里才是整数 id，
   两种格式的解析要分开处理。
4. **影像降采样后内参必须同步缩放**，否则重建会整体错位（适配层已处理）。
5. Python 3.9 的 urllib 不认 HTTP 308，HuggingFace CDN 会返回它，需要自定义重定向处理器。
6. gsplat 的 Windows 预编译轮子**只有 cp310**；Python 3.9 环境只能源码编译（需要 MSVC）。

## 六、关于内置的 `builtin` 后端

`builtin` 是系统自带的极简纯 PyTorch 实现（`aerosky/gsplat/`），存在的意义有两个：

* 在没有可用 CUDA 后端时把流程跑通（数据准备 → 重建 → 配准 → 报告），不至于整条链路不可用；
* 给前端三维视图提供一个不依赖 torch/CUDA 的快速预览渲染器。

它**不是本系统的 3DGS 主路径**，重建结果在报告里会明确标注 `backend=builtin`。
要做正式的三维重建，请用 `taichi` / `gsplat` / `official` 这些现成实现。
