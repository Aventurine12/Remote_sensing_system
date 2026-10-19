# -*- coding: utf-8 -*-
"""环境诊断器：先检测这台电脑的环境是否齐备，再决定要不要装依赖。

启动脚本会先调用本脚本；只有它报告"环境不完整"时才会触发安装。

用法：
    python tools/env_doctor.py              完整检测（真的逐个导入依赖），打印报告
    python tools/env_doctor.py --quick      快速检测（只查包在不在，不导入），供启动脚本秒判
    python tools/env_doctor.py --json       机器可读输出

退出码（启动脚本据此分支）：
    0  环境完备，可以直接启动
    1  依赖不完整，需要安装
    4  Python 版本过低（低于 3.9）

本脚本刻意**不依赖任何第三方库**，因此用系统里任意一个 Python 都能执行——
哪怕这个环境里什么都没装。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

MIN_PY = (3, 9)
ROOT = Path(__file__).resolve().parent.parent

# (导入名, 展示名, 分组)
DEPS = [
    ("numpy", "numpy", "核心"),
    ("torch", "torch (PyTorch)", "核心"),
    ("torchvision", "torchvision", "核心"),
    ("cv2", "opencv-python", "图像"),
    ("PIL", "Pillow", "图像"),
    ("skimage", "scikit-image", "图像"),
    ("rasterio", "rasterio (含 GDAL)", "地理"),
    ("pyproj", "pyproj", "地理"),
    ("shapely", "shapely", "地理"),
    ("ultralytics", "ultralytics (YOLO)", "模型"),
    ("timm", "timm", "模型"),
    ("segmentation_models_pytorch", "segmentation-models-pytorch", "模型"),
    ("einops", "einops", "模型"),
    ("huggingface_hub", "huggingface-hub", "模型"),
    ("PyQt5.QtWidgets", "PyQt5 (图形界面)", "界面"),
    ("matplotlib", "matplotlib", "其它"),
    ("pandas", "pandas", "其它"),
    ("scipy", "scipy", "其它"),
    ("sklearn", "scikit-learn", "其它"),
    ("yaml", "pyyaml", "其它"),
    ("plyfile", "plyfile", "其它"),
]

# 缺了会直接导致程序不能用（缺任意一个就不能启动）
REQUIRED = {"numpy", "torch", "cv2", "rasterio", "PyQt5.QtWidgets", "yaml", "shapely"}

WEIGHTS = [
    ("yolo11n-obb.pt", "DOTAv1 旋转框检测权重", False),
    ("yolov8s-building-segmentation.pt", "建筑实例分割权重", False),
]


def _human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def check_python() -> dict:
    ver = sys.version_info
    ok = ver[:2] >= MIN_PY
    return {
        "version": "%d.%d.%d" % ver[:3],
        "ok": ok,
        "required": "%d.%d" % MIN_PY,
        "executable": sys.executable,
        "in_venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        "venv_dir": sys.prefix,
        "bits": 64 if sys.maxsize > 2 ** 32 else 32,
    }


def check_venv_dir() -> dict:
    v = ROOT / ".venv"
    exe = v / "Scripts" / "python.exe"
    if not exe.exists():
        exe = v / "bin" / "python"           # Linux/macOS
    marker = (v / ".deps_ok").exists()
    link = ""
    try:
        if v.exists() and v.is_symlink():
            link = os.readlink(v)
    except OSError:
        pass
    return {"path": str(v), "exists": exe.exists(), "verified": marker, "link": link}


def check_deps(quick: bool) -> dict:
    import importlib
    import importlib.util
    rows, missing, broken = [], [], []
    for mod, label, group in DEPS:
        ver = ""
        state = "OK"
        try:
            if quick:
                # 只做路径查找，不执行模块代码 —— 毫秒级
                if importlib.util.find_spec(mod) is None:
                    raise ModuleNotFoundError(mod)
            else:
                m = importlib.import_module(mod)
                ver = str(getattr(m, "__version__", "") or "")
                if not ver:
                    for attr in ("__VERSION__", "version"):
                        if hasattr(m, attr):
                            ver = str(getattr(m, attr))
                            break
        except Exception as e:  # noqa: BLE001
            state = "MISS"
            missing.append(label)
            ver = e.__class__.__name__
        rows.append({"module": mod, "label": label, "group": group,
                     "state": state, "version": ver, "required": mod in REQUIRED})
    broken = [r["label"] for r in rows if r["state"] != "OK" and r["required"]]
    return {"rows": rows, "missing": missing, "missing_required": broken}


def check_hardware(quick: bool) -> dict:
    import multiprocessing
    info = {
        "cpu_count": multiprocessing.cpu_count(),
        "python_bits": 64 if sys.maxsize > 2 ** 32 else 32,
    }
    # 内存
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            info["mem_total_gb"] = round(st.ullTotalPhys / 1024 ** 3, 1)
            info["mem_avail_gb"] = round(st.ullAvailPhys / 1024 ** 3, 1)
    except Exception:  # noqa: BLE001
        pass
    # 磁盘（放得下依赖与成果即可）
    try:
        u = shutil.disk_usage(str(ROOT))
        info["disk_free_gb"] = round(u.free / 1024 ** 3, 1)
    except Exception:  # noqa: BLE001
        pass
    # NVIDIA GPU：用 nvidia-smi，不依赖 torch，所以任何环境都能查
    info["gpu"] = None
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
        if out.returncode == 0 and out.stdout.strip():
            info["gpu"] = out.stdout.strip().splitlines()[0].strip()
    except Exception:  # noqa: BLE001
        pass
    if not quick:
        try:
            import torch
            info["torch_cuda"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                info["cuda_device"] = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            info["torch_cuda"] = None
    return info


def check_weights() -> dict:
    rows, missing = [], []
    for name, desc, required in WEIGHTS:
        p = ROOT / "weights" / name
        ok = p.exists() and p.stat().st_size > 1024
        rows.append({"file": name, "desc": desc, "ok": ok,
                     "size": _human(p.stat().st_size) if p.exists() else "-",
                     "required": required})
        if not ok:
            missing.append(name)
    return {"rows": rows, "missing": missing}


def collect(quick: bool) -> dict:
    return {
        "root": str(ROOT),
        "python": check_python(),
        "venv": check_venv_dir(),
        "deps": check_deps(quick),
        "hardware": check_hardware(quick),
        "weights": check_weights(),
        "quick": quick,
    }


def verdict(r: dict) -> tuple:
    """返回 (退出码, 一句话结论)。"""
    if not r["python"]["ok"]:
        return 4, f"Python 版本过低（当前 {r['python']['version']}，需要 {r['python']['required']}+）"
    if r["deps"]["missing_required"]:
        n = len(r["deps"]["missing"])
        return 1, f"依赖不完整（缺少 {n} 项，其中关键项：{'、'.join(r['deps']['missing_required'])}），需要安装"
    if r["deps"]["missing"]:
        return 0, f"核心依赖齐备，但缺少 {len(r['deps']['missing'])} 个可选组件（不影响启动）"
    return 0, "环境完备，可以直接启动"


def render(r: dict) -> None:
    p, v, d, h, w = r["python"], r["venv"], r["deps"], r["hardware"], r["weights"]
    line = "=" * 66
    print(line)
    print("  空天协同遥感三维监测系统 —— 环境检测" + ("（快速）" if r["quick"] else ""))
    print(line)
    print(f"  程序目录 : {r['root']}")
    print()
    print("【Python】")
    flag = "OK  " if p["ok"] else "过低"
    print(f"  {flag} {p['version']}（{p['bits']} 位）  要求 >= {p['required']}")
    print(f"       解释器 : {p['executable']}")
    print(f"       虚拟环境: {'是' if p['in_venv'] else '否'}  {p['venv_dir']}")
    print()
    print("【虚拟环境目录】")
    print(f"       {v['path']}")
    print(f"       存在 : {'是' if v['exists'] else '否'}"
          f"   安装校验标记 : {'有' if v['verified'] else '无'}"
          + (f"   (目录联接 -> {v['link']})" if v["link"] else ""))
    print()
    print("【依赖包】")
    group = None
    for row in d["rows"]:
        if row["group"] != group:
            group = row["group"]
            print(f"  -- {group} --")
        mark = "OK  " if row["state"] == "OK" else "缺失"
        star = " *" if row["required"] else "  "
        print(f"  {mark}{star} {row['label']:<34} {row['version']}")
    print("  （带 * 的是关键依赖，缺任何一个都无法启动）")
    print()
    print("【模型权重】")
    for row in w["rows"]:
        print(f"  {'OK  ' if row['ok'] else '缺失'} {row['file']:<36} {row['size']}")
    print()
    print("【硬件】")
    print(f"  CPU 核心 : {h['cpu_count']}")
    if "mem_total_gb" in h:
        print(f"  内存     : 共 {h['mem_total_gb']} GB，可用 {h['mem_avail_gb']} GB")
    if "disk_free_gb" in h:
        print(f"  磁盘可用 : {h['disk_free_gb']} GB")
    print(f"  NVIDIA   : {h['gpu'] if h['gpu'] else '未检测到独显（三维重建会退回 CPU 简化实现）'}")
    if h.get("torch_cuda") is not None:
        print(f"  PyTorch  : CUDA {'可用' if h['torch_cuda'] else '不可用'}"
              + (f"（{h.get('cuda_device')}）" if h.get("cuda_device") else ""))
    print()
    code, msg = verdict(r)
    print(line)
    print(f"  结论：{msg}")
    if code == 0:
        print("  下一步：双击「一键启动.bat」即可运行。")
    elif code == 1:
        print("  下一步：双击「安装依赖.bat」，或直接双击「一键启动.bat」（会自动安装）。")
    else:
        print("  下一步：请先安装 Python 3.9 或更高版本（安装时勾选 Add Python to PATH）。")
        print("          下载地址：https://www.python.org/downloads/")
    print(line)


def main() -> int:
    ap = argparse.ArgumentParser(description="环境检测")
    ap.add_argument("--quick", action="store_true", help="只检查包在不在，不导入（快）")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--quiet", action="store_true", help="只打印结论一行")
    a = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    r = collect(a.quick)
    code, msg = verdict(r)
    if a.json:
        r["verdict"] = {"code": code, "message": msg}
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif a.quiet:
        print(msg)
    else:
        render(r)
    return code


if __name__ == "__main__":
    sys.exit(main())
