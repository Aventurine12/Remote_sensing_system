# -*- coding: utf-8 -*-
"""地理配准模块：把三维高斯模型锚定到 WGS84，并与卫星底图做互检。"""
from .georeference import GeoReference  # noqa: F401
from .registrar import RegistrationResult, register_to_raster  # noqa: F401

__all__ = ["GeoReference", "RegistrationResult", "register_to_raster"]
