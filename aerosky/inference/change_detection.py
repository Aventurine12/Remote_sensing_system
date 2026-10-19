# -*- coding: utf-8 -*-
"""变化检测后端（双时相）。

三种可选实现：

1. ``mask_diff``    —— 用同一个建筑分割模型分别解译前后时相，对建筑掩膜做差集，
   得到"新增建筑 / 拆除建筑"。这是最稳、开箱可用的方案（依赖已预训练的建筑提取模型）。
2. ``siamunet_diff``—— SiamUnet-diff（Chen & Shi, ICIP 2018）双流网络，
   可用 ``tools/train_change_detection.py`` 在 WHU-CD / LEVIR-CD 上训练后载入权重。
3. ``cva``          —— 变化向量分析（CVA）+ Otsu 阈值 + 形态学，无权重兜底。

输出统一为变化前景概率图 (H, W)。
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

from ..utils.logger import get_logger
from .base import BaseModel

LOG = get_logger("inference.cd")


# --------------------------------------------------------------------- SiamUnet
def _make_siamunet(base: int = 16):
    """构造 SiamUnet-diff（延迟导入 torch，未安装时返回 None）。"""
    try:
        import torch
        import torch.nn as nn
    except Exception:  # noqa: BLE001
        return None

    def conv_bn(in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )

    class SiamUnetDiff(nn.Module):
        """两个共享权重的编码器分支，特征做逐元素相减后进解码器。"""

        def __init__(self, in_ch=3, base_ch=16):
            super().__init__()
            b = base_ch
            self.enc1 = conv_bn(in_ch, b)
            self.enc2 = conv_bn(b, b * 2)
            self.enc3 = conv_bn(b * 2, b * 4)
            self.enc4 = conv_bn(b * 4, b * 8)
            self.pool = nn.MaxPool2d(2)
            self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
            self.dec3 = conv_bn(b * 4, b * 4)
            self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
            self.dec2 = conv_bn(b * 2, b * 2)
            self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
            self.dec1 = conv_bn(b, b)
            self.head = nn.Conv2d(b, 1, 1)

        def forward(self, x1, x2):
            f1_1, f2_1 = self.enc1(x1), self.enc1(x2)
            f1_2, f2_2 = self.enc2(self.pool(f1_1)), self.enc2(self.pool(f2_1))
            f1_3, f2_3 = self.enc3(self.pool(f1_2)), self.enc3(self.pool(f2_2))
            f1_4, f2_4 = self.enc4(self.pool(f1_3)), self.enc4(self.pool(f2_3))
            d = self.dec3(self.up3(torch.abs(f1_4 - f2_4))) + torch.abs(f1_3 - f2_3)
            d = self.dec2(self.up2(d)) + torch.abs(f1_2 - f2_2)
            d = self.dec1(self.up1(d)) + torch.abs(f1_1 - f2_1)
            return self.head(d)

    return SiamUnetDiff(in_ch=3, base_ch=base)


class ChangeDetectionModel(BaseModel):
    task = "change_detection"

    def __init__(self, cfg=None, weights_dir="weights", device=None,
                 segmenter: Optional[BaseModel] = None):
        super().__init__(cfg, weights_dir, device)
        self.backend = ""
        self.segmenter = segmenter
        self.threshold = float(self.cfg.get("threshold", 0.5))
        self.min_area_px = int(self.cfg.get("min_area_px", 64))

    # ------------------------------------------------------------------ 载入
    def load(self) -> "ChangeDetectionModel":
        wanted = str(self.cfg.get("backend", "mask_diff"))
        order = [wanted] + [b for b in ("mask_diff", "siamunet_diff", "cva") if b != wanted]
        for b in order:
            if b == "mask_diff" and self.segmenter is not None \
                    and getattr(self.segmenter, "model", None) is not None:
                self.backend = "mask_diff"
                self.name = "建筑掩膜差分(双时相)"
                LOG.info("变化检测后端：%s", self.backend)
                return self
            if b == "siamunet_diff" and self._load_siam():
                self.backend = "siamunet_diff"
                LOG.info("变化检测后端：%s", self.backend)
                return self
            if b == "cva":
                self.backend = "cva"
                self.fallback = True
                LOG.info("变化检测后端：cva（变化向量分析兜底）")
                return self
        self.backend = "cva"
        self.fallback = True
        return self

    def _load_siam(self) -> bool:
        try:
            import torch
        except Exception as e:  # noqa: BLE001
            self.load_error = f"torch 不可用: {e}"
            return False
        ckpt = self.weight_path(self.cfg.get("checkpoint", "siamunet_diff_whucd.pth"))
        if not os.path.isfile(ckpt):
            self.load_error = f"未找到权重 {ckpt}"
            return False
        net = _make_siamunet(int(self.cfg.get("base_channels", 16)))
        if net is None:
            return False
        state = torch.load(ckpt, map_location="cpu")
        state = state.get("state_dict", state.get("model", state))
        state = {k.replace("module.", ""): v for k, v in state.items()}
        net.load_state_dict(state, strict=False)
        net.eval().to(self.device)
        self.model = net
        self.name = "SiamUnet-diff"
        return True

    # ------------------------------------------------------------------ 推理
    def predict_pair(self, rgb_a: np.ndarray, rgb_b: np.ndarray) -> np.ndarray:
        """返回 (H, W) float32 变化概率图。"""
        if self.backend == "mask_diff":
            return self._predict_mask_diff(rgb_a, rgb_b)
        if self.backend == "siamunet_diff":
            return self._predict_siam(rgb_a, rgb_b)
        return self._predict_cva(rgb_a, rgb_b)

    # ---- 1) 建筑掩膜差分
    def _predict_mask_diff(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        pa = self._build_prob(a)
        pb = self._build_prob(b)
        diff_new = np.clip(pb - pa, 0, 1)          # 新增建筑
        diff_del = np.clip(pa - pb, 0, 1)          # 拆除建筑
        prob = np.maximum(diff_new, diff_del * 0.8)
        return self._clean(prob)

    def _build_prob(self, rgb: np.ndarray) -> np.ndarray:
        seg = self.segmenter
        p = seg.predict_tile([rgb])[0] if seg is not None else np.zeros(rgb.shape[:2], np.float32)
        return np.asarray(p, np.float32)

    # ---- 2) SiamUnet-diff
    def _predict_siam(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        import torch
        import cv2
        size = tuple(self.cfg.get("input_size", [256, 256]))
        ha, wa = a.shape[:2]
        xa = cv2.resize(a, (size[1], size[0]), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        xb = cv2.resize(b, (size[1], size[0]), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        xa = (xa - mean) / std
        xb = (xb - mean) / std
        with torch.no_grad():
            t1 = torch.from_numpy(xa.transpose(2, 0, 1))[None].to(self.device)
            t2 = torch.from_numpy(xb.transpose(2, 0, 1))[None].to(self.device)
            logit = self.model(t1, t2)
            prob = torch.sigmoid(logit)[0, 0].cpu().numpy()
        prob = cv2.resize(prob, (wa, ha), interpolation=cv2.INTER_LINEAR)
        return self._clean(prob)

    # ---- 3) 变化向量分析
    def _predict_cva(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        import cv2
        x = a.astype(np.float32) - b.astype(np.float32)
        mag = np.sqrt((x ** 2).sum(axis=2))
        mag = cv2.GaussianBlur(mag, (5, 5), 0)
        mag_n = np.clip((mag - mag.min()) / (mag.ptp() + 1e-6), 0, 1)
        # Otsu 自适应阈值
        u8 = (mag_n * 255).astype(np.uint8)
        th, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        prob = (u8 >= max(th, 25)).astype(np.float32)
        return self._clean(prob)

    # ---- 后处理：面积过滤 + 形态学，去掉零碎噪点
    def _clean(self, prob: np.ndarray) -> np.ndarray:
        import cv2
        m = (prob >= self.threshold).astype(np.uint8)
        if m.sum() == 0:
            return prob
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        out = np.zeros_like(m)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area_px:
                out[lab == i] = 1
        return np.maximum(prob * out, out).astype(np.float32)


DEFAULT_CFG = {"input_size": [256, 256], "base_channels": 16, "threshold": 0.5, "min_area_px": 64}
