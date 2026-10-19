# -*- coding: utf-8 -*-
"""后端基类与统一结果结构。"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..utils.logger import get_logger

LOG = get_logger("gsbackend")


@dataclass
class GSResult:
    """一次三维重建的结果。"""

    backend: str = ""
    out_dir: str = ""
    ply: str = ""
    num_gaussians: int = 0
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    seconds: float = 0.0
    iterations: Optional[int] = None
    previews: List[str] = field(default_factory=list)
    log_file: str = ""
    device: str = ""
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return dict(self.__dict__)


class GSBackend:
    """所有 3DGS 后端的公共接口。"""

    name = "base"
    title = "未命名后端"
    requires: Sequence[str] = ()
    homepage = ""
    uses_cuda = True

    def __init__(self, cfg: dict | None = None, python_exe: str | None = None):
        self.cfg = dict(cfg or {})
        self.python_exe = python_exe or self.cfg.get("python_exe") or sys.executable

    # ------------------------------------------------------------------ 探测
    def available(self) -> tuple:
        """返回 (是否可用, 说明)。子类实现。"""
        return False, "未实现"

    def describe(self) -> str:
        ok, why = self.available()
        return f"{self.name:<10} {'可用' if ok else '不可用':<4} {why}"

    # ------------------------------------------------------------------ 训练
    def train(self, images_dir: str, sparse_dir: str, out_dir: str,
              iterations: int = 7000, downscale: int = 1, sh_degree: int = 3,
              device: str = "cuda", extra: Dict | None = None,
              progress_cb=None) -> GSResult:
        raise NotImplementedError

    def render(self, ply: str, out_dir: str, **kw) -> List[str]:
        return []

    # ------------------------------------------------------------------ 工具
    def _run(self, cmd: Sequence[str], cwd: str | Path, log_file: str | Path,
             env: Dict | None = None, timeout: Optional[float] = None,
             progress_cb=None) -> int:
        """执行外部命令并把输出同时写进日志文件。"""
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        e = dict(os.environ)
        e.setdefault("PYTHONUNBUFFERED", "1")
        if env:
            e.update(env)
        LOG.info("执行：%s", " ".join(str(c) for c in cmd))
        with open(log_file, "w", encoding="utf-8", errors="replace") as f:
            f.write("$ " + " ".join(str(c) for c in cmd) + "\n\n")
            f.flush()
            p = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd), env=e,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace",
                                 bufsize=1)
            t0 = time.time()
            for line in p.stdout:              # 实时写日志，方便看进度
                f.write(line)
                f.flush()
                if progress_cb is not None:
                    progress_cb(line)
                elif "iter" in line.lower() or "%" in line:
                    LOG.info("  %s", line.rstrip()[:160])
            p.wait(timeout=timeout)
            f.write(f"\n[exit] {p.returncode}\n")
        LOG.info("命令结束（退出码 %d，耗时 %.1fs）", p.returncode, time.time() - t0)
        return p.returncode

    def out_ply(self, out_dir: str | Path) -> str:
        """在输出目录里找 .ply（不同后端的文件名不一样）。"""
        d = Path(out_dir)
        cands = sorted(d.rglob("*.ply"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in cands:
            if "point_cloud" in p.name or "splat" in p.name.lower():
                return str(p)
        return str(cands[0]) if cands else ""

    @staticmethod
    def which(name: str) -> Optional[str]:
        return shutil.which(name)

    def write_meta(self, out_dir: str | Path, **kw) -> str:
        p = Path(out_dir) / "gs_meta.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(kw, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(p)
