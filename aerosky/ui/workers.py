# -*- coding: utf-8 -*-
"""后台任务线程：让推理/重建这类重活不阻塞界面。"""
from __future__ import annotations

import traceback
from typing import Any, Callable, Optional

from PyQt5.QtCore import QThread, pyqtSignal


class TaskWorker(QThread):
    """把任意可调用对象放到子线程执行；被调用函数可接收 progress 回调。"""

    progressed = pyqtSignal(int, str)          # 百分比, 文本
    message = pyqtSignal(str)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn: Callable, *args, want_progress: bool = False, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.want_progress = want_progress
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def _progress(self, *a) -> None:
        if not a:
            return
        if len(a) == 1:
            self.progressed.emit(-1, str(a[0]))
        else:
            cur, total = int(a[0]), int(a[1])
            pct = int(cur * 100 / max(total, 1))
            self.progressed.emit(pct, f"{cur}/{total}")

    def run(self) -> None:
        try:
            if self.want_progress:
                self.kwargs.setdefault("progress", self._progress)
            out = self.fn(*self.args, **self.kwargs)
            self.finished_ok.emit(out)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")
