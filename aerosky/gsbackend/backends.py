# -*- coding: utf-8 -*-
"""各 3DGS 现成实现的后端适配。

每个后端只做三件事：探测可用性、把数据集喂给它、把产物（.ply / 指标）取回来。
真正的高斯泼溅光栅化与训练全部由这些成熟实现负责，本系统不做算法重写。

数据格式：所有后端都吃 COLMAP 稀疏重建目录（images/ + sparse/0/），
也就是遥感流水线里用 pycolmap 直接产出的那套东西。
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from .base import GSBackend, GSResult
from ..utils.logger import get_logger

LOG = get_logger("gsbackend.impl")

# 仓库默认克隆位置
REPO_ROOT = Path(os.environ.get("AEROSKY_3DGS_REPOS", r"D:\Code\dc\3dgs_repos"))


def _find_python(cands: List[str]) -> Optional[str]:
    for c in cands:
        if not c:
            continue
        p = Path(c)
        if p.is_file():
            return str(p)
    return None


def _probe_torch(python_exe: str) -> dict:
    """在一个 Python 解释器里问一下 torch / CUDA 的情况。"""
    import subprocess
    code = ("import json;"
            "import torch;"
            "print(json.dumps({'ver': torch.__version__,"
            " 'cuda': torch.cuda.is_available(),"
            " 'dev': torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}))")
    try:
        out = subprocess.run([python_exe, "-c", code], capture_output=True, text=True,
                             timeout=120)
        if out.returncode != 0:
            return {}
        import json
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {}


# --------------------------------------------------------------------------- gsplat
class GSplatBackend(GSBackend):
    """nerfstudio 的 gsplat —— 工业界广泛使用的 3DGS 实现，有 Windows 预编译轮子。

    本后端调用 :mod:`aerosky.gsbackend.gsplat_train`（在拥有 gsplat 的解释器里运行）。
    """

    name = "gsplat"
    title = "gsplat (nerfstudio)"
    homepage = "https://github.com/nerfstudio-project/gsplat"

    def available(self) -> tuple:
        py = self.python_exe
        if not py or not Path(py).exists():
            return False, f"解释器不存在：{py}"
        try:
            import subprocess
            r = subprocess.run([py, "-c", "import gsplat,torch;print(gsplat.__version__,"
                                           "torch.cuda.is_available())"],
                               capture_output=True, text=True, timeout=180)
            if r.returncode != 0:
                return False, "该解释器里没有装 gsplat（pip install gsplat）"
            line = r.stdout.strip().splitlines()[-1].split()
            if len(line) >= 2 and line[1] == "True":
                return True, f"gsplat {line[0]}，CUDA 可用"
            return False, f"gsplat {line[0]} 已装，但该解释器没有 CUDA"
        except Exception as e:  # noqa: BLE001
            return False, str(e).splitlines()[0]

    def train(self, images_dir, sparse_dir, out_dir, iterations=7000, downscale=1,
              sh_degree=3, device="cuda", extra=None, progress_cb=None) -> GSResult:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        log = out_dir / "train.log"
        entry = Path(__file__).with_name("gsplat_train.py")
        cmd = [self.python_exe, str(entry),
               "--images", str(images_dir), "--sparse", str(sparse_dir),
               "--out", str(out_dir), "--iters", str(iterations),
               "--downscale", str(downscale), "--sh", str(sh_degree),
               "--device", device]
        extra = extra or {}
        if extra.get("max_gaussians"):
            cmd += ["--max-gaussians", str(extra["max_gaussians"])]
        if extra.get("time_budget"):
            cmd += ["--time-budget", str(extra["time_budget"])]
        t0 = time.time()
        rc = self._run(cmd, Path(__file__).resolve().parents[2], log,
                       progress_cb=progress_cb)
        res = GSResult(backend=self.name, out_dir=str(out_dir),
                       ply=self.out_ply(out_dir), seconds=time.time() - t0,
                       iterations=iterations, log_file=str(log), device=device)
        res = self._collect_metrics(res, out_dir)
        if rc != 0:
            res.extra["exit_code"] = rc
        return res

    @staticmethod
    def _collect_metrics(res: GSResult, out_dir: Path) -> GSResult:
        m = out_dir / "metrics.json"
        if m.exists():
            import json
            try:
                d = json.loads(m.read_text(encoding="utf-8"))
                res.num_gaussians = int(d.get("num_gaussians", 0))
                res.psnr = d.get("PSNR")
                res.ssim = d.get("SSIM")
                res.previews = [str(p) for p in d.get("previews", [])]
                res.extra.update({k: v for k, v in d.items()
                                  if k in ("train_seconds", "final_loss", "stopped_early")})
            except Exception:  # noqa: BLE001
                pass
        return res


# --------------------------------------------------------------------------- 官方
class OfficialBackend(GSBackend):
    """原论文官方实现 graphdeco-inria/gaussian-splatting。

    需要能编译 CUDA 扩展（nvcc + MSVC）。命令行接口：
        python train.py -s <colmap数据目录> -m <输出目录>
    """

    name = "official"
    title = "3D Gaussian Splatting (官方实现)"
    homepage = "https://github.com/graphdeco-inria/gaussian-splatting"

    def repo_dir(self) -> Path:
        return Path(self.cfg.get("official_repo") or (REPO_ROOT / "gaussian-splatting"))

    def available(self) -> tuple:
        repo = self.repo_dir()
        if not (repo / "train.py").exists():
            return False, f"仓库未克隆：{repo} （运行 tools/setup_3dgs.py --backend official）"
        r = _probe_torch(self.python_exe)
        if not r.get("cuda"):
            return False, "该解释器没有可用的 CUDA"
        try:
            import subprocess
            chk = subprocess.run([self.python_exe, "-c",
                                  "import diff_gaussian_rasterization as d; print('ok')"],
                                 capture_output=True, text=True, timeout=180)
            if chk.returncode != 0:
                return False, "CUDA 扩展未编译（见仓库 README 的 submodules 安装步骤）"
        except Exception as e:  # noqa: BLE001
            return False, str(e).splitlines()[0]
        return True, f"仓库就绪：{repo}"

    def train(self, images_dir, sparse_dir, out_dir, iterations=30000, downscale=1,
              sh_degree=3, device="cuda", extra=None, progress_cb=None) -> GSResult:
        repo = self.repo_dir()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        log = out_dir / "train.log"
        # 官方实现要求数据目录是 <root>/images + <root>/sparse/0 的结构
        data_root = _ensure_colmap_layout(images_dir, sparse_dir, out_dir / "dataset")
        cmd = [self.python_exe, str(repo / "train.py"), "-s", str(data_root),
               "-m", str(out_dir / "model"), "--iterations", str(iterations),
               "--sh_degree", str(sh_degree)]
        t0 = time.time()
        rc = self._run(cmd, repo, log, progress_cb=progress_cb)
        ply = out_dir / "model" / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
        res = GSResult(backend=self.name, out_dir=str(out_dir),
                       ply=str(ply) if ply.exists() else self.out_ply(out_dir / "model"),
                       seconds=time.time() - t0, iterations=iterations, log_file=str(log))
        if rc != 0:
            res.extra["exit_code"] = rc
        return res

    def render(self, ply, out_dir, **kw) -> List[str]:
        return []


# --------------------------------------------------------------------------- OpenSplat
class OpenSplatBackend(GSBackend):
    """OpenSplat：C++/libtorch 实现，支持 CPU / CUDA / Metal，产物为 .ply。

        opensplat <colmap数据目录> -n <迭代数> -o <输出.ply>
    """

    name = "opensplat"
    title = "OpenSplat"
    homepage = "https://github.com/pierotofy/OpenSplat"

    def available(self) -> tuple:
        exe = self.cfg.get("opensplat_exe") or self.which("opensplat") or \
            str(REPO_ROOT / "OpenSplat" / "build" / "opensplat.exe")
        if not Path(exe).exists():
            return False, f"未找到 opensplat 可执行文件（{exe}）"
        return True, f"可执行文件：{exe}"

    def train(self, images_dir, sparse_dir, out_dir, iterations=2000, downscale=1,
              sh_degree=3, device="cpu", extra=None, progress_cb=None) -> GSResult:
        exe = self.cfg.get("opensplat_exe") or self.which("opensplat") or \
            str(REPO_ROOT / "OpenSplat" / "build" / "opensplat.exe")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        data_root = _ensure_colmap_layout(images_dir, sparse_dir, out_dir / "dataset")
        log = out_dir / "train.log"
        ply = out_dir / "splat.ply"
        cmd = [exe, str(data_root), "-n", str(iterations), "-o", str(ply)]
        if device == "cpu":
            cmd += ["-d", "cpu"]
        t0 = time.time()
        rc = self._run(cmd, out_dir, log, progress_cb=progress_cb)
        res = GSResult(backend=self.name, out_dir=str(out_dir),
                       ply=str(ply) if ply.exists() else self.out_ply(out_dir),
                       seconds=time.time() - t0, iterations=iterations,
                       log_file=str(log), device=device)
        if rc != 0:
            res.extra["exit_code"] = rc
        return res


# --------------------------------------------------------------------------- Taichi
class TaichiBackend(GSBackend):
    """taichi_3d_gaussian_splatting：纯 Python + Taichi 的 3D 高斯泼溅实现。

    高斯光栅化用 Taichi 语言写成（CUDA 后端），不需要编译 C++/CUDA 扩展，
    因此只要有一个带 CUDA 的 PyTorch + taichi 就能跑。
    仓库：https://github.com/wanmeihuali/taichi_3d_gaussian_splatting
    """

    name = "taichi"
    title = "taichi_3d_gaussian_splatting（Taichi 语言实现的 3DGS）"
    homepage = "https://github.com/wanmeihuali/taichi_3d_gaussian_splatting"
    default_repo = r"D:\Code\dc\3dgs_repos\taichi_3d_gaussian_splatting"

    def repo_dir(self) -> Path:
        return Path(self.cfg.get("taichi_repo") or os.environ.get("TAICHI_3DGS_REPO")
                    or self.default_repo)

    def available(self) -> tuple:
        repo = self.repo_dir()
        if not (repo / "gaussian_point_train.py").exists():
            return False, (f"仓库未克隆：{repo} "
                           f"（python tools/setup_3dgs.py --backend taichi）")
        py = self.python_exe
        if not Path(py).exists():
            return False, f"解释器不存在：{py}"
        try:
            import subprocess
            r = subprocess.run(
                [py, "-c", "import taichi,torch,json;print(json.dumps("
                           "{'ti':taichi.__version__,'cuda':torch.cuda.is_available(),"
                           "'dev':torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}))"],
                capture_output=True, text=True, timeout=300)
            if r.returncode != 0:
                return False, "该解释器缺少 taichi 或 torch（见 requirements.txt）"
            import json as _j
            d = _j.loads(r.stdout.strip().splitlines()[-1])
            if not d.get("cuda"):
                return False, "该解释器没有可用的 CUDA（Taichi 版需要 CUDA 后端）"
            return True, f"taichi {d['ti']}，GPU：{d['dev']}"
        except Exception as e:  # noqa: BLE001
            return False, str(e).splitlines()[0]

    def train(self, images_dir, sparse_dir, out_dir, iterations=7000, downscale=1,
              sh_degree=3, device="cuda", extra=None, progress_cb=None) -> GSResult:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        log = out_dir / "train.log"
        entry = Path(__file__).with_name("taichi_train.py")
        cmd = [self.python_exe, str(entry),
               "--images", str(images_dir), "--sparse", str(sparse_dir),
               "--out", str(out_dir), "--repo", str(self.repo_dir()),
               "--iters", str(iterations), "--downscale", str(max(1, downscale)),
               "--device", device]
        extra = extra or {}
        if extra.get("max_side"):
            cmd += ["--max-side", str(extra["max_side"])]
        t0 = time.time()
        rc = self._run(cmd, Path(__file__).resolve().parents[2], log,
                       progress_cb=progress_cb)
        res = GSResult(backend=self.name, out_dir=str(out_dir),
                       ply=self.out_ply(out_dir), seconds=time.time() - t0,
                       iterations=iterations, log_file=str(log), device=device)
        m = out_dir / "metrics.json"
        if m.exists():
            import json
            try:
                d = json.loads(m.read_text(encoding="utf-8"))
                res.num_gaussians = int(d.get("num_gaussians", 0))
                res.previews = list(d.get("previews", []) or [])
                res.extra["dataset"] = d.get("dataset")
                res.extra["train_seconds"] = d.get("train_seconds")
            except Exception:  # noqa: BLE001
                pass
        if rc != 0 and not res.ply:
            raise RuntimeError(f"taichi 3DGS 训练失败（退出码 {rc}），详见 {log}")
        if rc != 0:
            res.extra["exit_code"] = rc
        return res


# --------------------------------------------------------------------------- 内置兜底
class BuiltinBackend(GSBackend):
    """内置极简实现（纯 PyTorch，无需 CUDA）。

    **不是**本系统的 3DGS 主路径：它只用于在没有可用 CUDA 后端时跑通流程、
    生成可直接查看的模型，以及给前端提供离线预览。真实重建请用 gsplat / 官方实现。
    """

    name = "builtin"
    title = "内置极简实现（无 CUDA 兜底，仅供流程演示）"

    def available(self) -> tuple:
        try:
            import torch  # noqa: F401
            return True, "纯 PyTorch 实现，CPU 可用（非 3DGS 主路径）"
        except Exception as e:  # noqa: BLE001
            return False, str(e).splitlines()[0]

    def train(self, images_dir, sparse_dir, out_dir, iterations=700, downscale=1,
              sh_degree=1, device="cpu", extra=None, progress_cb=None) -> GSResult:
        from ..reconstruction import _train_builtin
        t0 = time.time()
        res = _train_builtin(images_dir, sparse_dir, out_dir, iterations=iterations,
                             downscale=downscale, sh_degree=sh_degree, device=device,
                             progress_cb=progress_cb, cfg=extra or {})
        res.backend = self.name
        res.seconds = time.time() - t0
        return res


# --------------------------------------------------------------------------- 工具
def _ensure_colmap_layout(images_dir, sparse_dir, root: Path) -> Path:
    """把 (images_dir, sparse_dir) 组织成 COLMAP 标准目录 <root>/images + <root>/sparse/0。"""
    root = Path(root)
    if root.exists() and (root / "images").exists() and (root / "sparse" / "0").exists():
        return root
    (root / "sparse").mkdir(parents=True, exist_ok=True)
    img_dst = root / "images"
    if not img_dst.exists():
        try:
            img_dst.symlink_to(Path(images_dir).resolve(), target_is_directory=True)
        except Exception:  # noqa: BLE001
            shutil.copytree(images_dir, img_dst, dirs_exist_ok=True)
    sp_dst = root / "sparse" / "0"
    if sp_dst.exists():
        shutil.rmtree(sp_dst, ignore_errors=True)
    sp_dst.mkdir(parents=True, exist_ok=True)
    for f in Path(sparse_dir).glob("*"):
        if f.is_file():
            shutil.copy2(f, sp_dst / f.name)
    return root
