# -*- coding: utf-8 -*-
"""推理层：模型基类、结果数据结构与统一注册表。"""
from .base import BaseModel, Detection, TaskResult  # noqa: F401
from .tiler import Tiler, blend_window  # noqa: F401
from .registry import build_model, available_models  # noqa: F401
