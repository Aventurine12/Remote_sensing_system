# -*- coding: utf-8 -*-
"""aerosky —— 空天协同的多光谱遥感影像智能解译与三维监测系统。

模块划分：
    aerosky.utils      地理坐标、日志等基础工具
    aerosky.inference  三大智能解译模型（语义分割 / 目标检测 / 变化检测）与分块推理
    aerosky.gsplat     3D 高斯泼溅重建、渲染与 .ply 读写
    aerosky.registration 三维模型与卫星底图的地理配准
    aerosky.report     无人机航摄任务建议与结构化分析报告
    aerosky.ui         PyQt5 集成化前端
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
