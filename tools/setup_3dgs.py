# -*- coding: utf-8 -*-
"""一键安装/克隆 3D 高斯泼溅后端。

    python tools/setup_3dgs.py --list                  # 看当前环境哪些后端可用
    python tools/setup_3dgs.py --backend gsplat        # 装 gsplat（推荐，有 Windows 预编译轮子）
    python tools/setup_3dgs.py --backend official      # 克隆官方实现（需要 CUDA 工具链）
    python tools/setup_3dgs.py --backend opensplat     # 克隆并编译 OpenSplat（支持 CPU）
    python tools/setup_3dgs.py --backend all

说明
----
* **gsplat**（nerfstudio）是最推荐的选择：pip 直接装，Windows 有预编译 CUDA 轮子，
  但**只提供 cp310 的 Windows 轮子**，所以需要一个 Python 3.10 的环境，例如
  `conda create -n gs310 python=3.10`。
* **official** 是原论文实现，必须编译 CUDA 扩展（nvcc + MSVC）。
* **opensplat** 是 C++ 实现，支持 CPU 训练，但要 CMake + 编译器。
* 三者都吃同一套 COLMAP 数据格式，也就是遥感流水线里 pycolmap 直接产出的那套。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aerosky.gsbackend.registry import detect_backends  # noqa: E402
from aerosky.utils.logger import get_logger, setup  # noqa: E402

LOG = get_logger("setup_3dgs")
REPO_ROOT = Path(os.environ.get("AEROSKY_3DGS_REPOS", r"D:\Code\dc\3dgs_repos"))

GSPLAT_WHEEL_INDEX = "https://docs.gsplat.studio/whl"
GIT = {
    "official": ("https://github.com/graphdeco-inria/gaussian-splatting.git",
                 REPO_ROOT / "gaussian-splatting"),
    "taichi": ("https://github.com/wanmeihuali/taichi_3d_gaussian_splatting.git",
               REPO_ROOT / "taichi_3d_gaussian_splatting"),
    "opensplat": ("https://github.com/pierotofy/OpenSplat.git", REPO_ROOT / "OpenSplat"),
}


def _run(cmd, cwd=None, check=True):
    print("$ " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None)
    if check and r.returncode != 0:
        LOG.error("命令失败（退出码 %d）", r.returncode)
    return r.returncode


def clone(name: str) -> bool:
    if name not in GIT:
        return False
    url, dst = GIT[name]
    if (dst / ".git").exists():
        LOG.info("%s 已存在，执行 git pull", dst)
        _run(["git", "pull", "--ff-only"], cwd=dst, check=False)
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("克隆 %s -> %s", url, dst)
    if _run(["git", "clone", "--recursive", url, str(dst)]) != 0:
        return False
    LOG.info("克隆完成：%s", dst)
    return True


def install_gsplat(python_exe: str, torch_tag: str = "pt22cu121") -> bool:
    """装 gsplat。优先用官方预编译轮子（免编译），失败则尝试源码安装。"""
    py = python_exe
    if not Path(py).exists():
        LOG.error("解释器不存在：%s", py)
        return False
    ver = subprocess.run([py, "-c", "import sys;print('%d%d'%sys.version_info[:2])"],
                         capture_output=True, text=True).stdout.strip()
    LOG.info("目标解释器 Python %s：%s", ver, py)
    if ver != "310":
        LOG.warning("gsplat 的 Windows 预编译轮子只提供 cp310；当前是 cp%s。"
                    "建议 conda create -n gs310 python=3.10 后再装。", ver)
    # 依赖
    _run([py, "-m", "pip", "install", "-q", "-i",
          "https://pypi.tuna.tsinghua.edu.cn/simple",
          "numpy==1.26.4", "opencv-python-headless", "tqdm"], check=False)
    # 预编译轮子
    url = f"{GSPLAT_WHEEL_INDEX}/{torch_tag}/gsplat/"
    try:
        import re
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "aerosky/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            html = r.read().decode("utf-8", "ignore")
        whls = re.findall(r">([^<>]+\.whl)<", html)
        win = [w for w in whls if f"cp{ver}" in w and "win_amd64" in w]
        if win:
            wheel = sorted(win)[-1]
            LOG.info("发现预编译轮子：%s", wheel)
            rc = _run([py, "-m", "pip", "install", f"{url}{wheel}"], check=False)
            if rc == 0:
                LOG.info("gsplat 安装完成（预编译，免编译）")
                return True
    except Exception as e:  # noqa: BLE001
        LOG.warning("查询预编译轮子失败：%s", e)
    LOG.info("改从源码安装 gsplat（需要 nvcc + MSVC）…")
    return _run([py, "-m", "pip", "install", "gsplat"]) == 0


def main() -> int:
    ap = argparse.ArgumentParser(description="安装 3DGS 后端")
    ap.add_argument("--backend", default="gsplat",
                    choices=["gsplat", "official", "opensplat", "taichi", "all", "list"])
    ap.add_argument("--python", default=None, help="用于安装的目标解释器")
    ap.add_argument("--torch-tag", default="pt22cu121",
                    help="gsplat 预编译轮子的 torch/cuda 组合，如 pt22cu121 / pt23cu121")
    a = ap.parse_args()
    setup()

    if a.backend == "list":
        print("当前环境的后端可用性：\n")
        for d in detect_backends({}, verbose=False):
            print(f"  [{'✓' if d['available'] else '×'}] {d['name']:<9} {d['title']}")
            print(f"      {d['reason']}")
            if d.get("python"):
                print(f"      解释器：{d['python']}")
        print("\n仓库目录：", REPO_ROOT)
        print("安装示例：python tools/setup_3dgs.py --backend gsplat --python "
              r"C:\miniconda3\envs\gs310\python.exe")
        return 0

    py = a.python or sys.executable
    ok = True
    if a.backend in ("gsplat", "all"):
        ok &= install_gsplat(py, a.torch_tag)
    if a.backend in ("official", "all"):
        ok &= clone("official")
        if ok:
            print("\n官方实现还需要编译 CUDA 扩展：")
            print(f"  cd {REPO_ROOT / 'gaussian-splatting'}")
            print("  pip install submodules/diff-gaussian-rasterization")
            print("  pip install submodules/simple-knn")
    if a.backend in ("opensplat", "all"):
        ok &= clone("opensplat")
        if ok:
            print("\nOpenSplat 需要 CMake 编译（支持 CPU / CUDA / Metal）：")
            print(f"  cd {REPO_ROOT / 'OpenSplat'}")
            print("  cmake -B build -DCMAKE_BUILD_TYPE=Release")
            print("  cmake --build build --config Release")
    if a.backend in ("taichi", "all"):
        ok &= clone("taichi")
        if ok:
            _run([py, "-m", "pip", "install", "-q", "-i",
                  "https://pypi.tuna.tsinghua.edu.cn/simple", "taichi", "pandas", "pyarrow"],
                 check=False)

    print("\n=== 安装结果 ===")
    for d in detect_backends({}, verbose=False):
        print(f"  [{'✓' if d['available'] else '×'}] {d['name']:<9} {d['reason']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
