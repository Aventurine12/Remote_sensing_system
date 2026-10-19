# -*- coding: utf-8 -*-
"""抓取公开的真实卫星影像，拼成带地理参考的演示 GeoTIFF（前/后时相）。

数据来源：HuggingFace 数据集 keremberke/satellite-building-segmentation（真实卫星影像）。
后时相 = 前时相 + 若干"新增建筑"图块（从影像中已有建筑处拷贝粘贴），
用于演示变化检测。网络不可用时自动回退到程序化合成数据。

    python tools/fetch_real_sample.py --out outputs/real --n 6
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from aerosky.utils.geo import Raster, WGS84, make_demo_geotiff

DATASET = "keremberke/satellite-building-segmentation"


def list_rows(split: str, offset: int, length: int):
    url = ("https://datasets-server.huggingface.co/rows?dataset="
           + DATASET.replace("/", "%2F")
           + f"&config=full&split={split}&offset={offset}&length={length}")
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r).get("rows", [])


def fetch_image(url: str) -> np.ndarray:
    req = urllib.request.Request(url, headers={"User-Agent": "aerosky/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        buf = np.frombuffer(r.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("影像解码失败")
    img = img[:, :, ::-1]
    # 过滤掉全黑 / 近乎纯色的无效影像
    if float(img.mean()) < 25 or float(img.std()) < 8:
        raise ValueError("影像无效（过暗或近似纯色）")
    return img


def build(out: Path, n: int = 9, split: str = "validation", cols: int = 3) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    imgs = []
    offset = 0
    while len(imgs) < n and offset < n * 4:
        rows = list_rows(split, offset, min(6, n))
        if not rows:
            break
        for row in rows:
            if len(imgs) >= n:
                break
            try:
                imgs.append(fetch_image(row["row"]["image"]["src"]))
            except Exception as e:  # noqa: BLE001
                print("  跳过一张：", e)
        offset += len(rows)
    if len(imgs) < 2:
        raise RuntimeError("真实影像获取失败")
    print(f"  已获取 {len(imgs)} 景真实卫星影像")

    cell = max(im.shape[0] for im in imgs)
    target = (cell, cell)
    tiles = [cv2.resize(im, target[::-1], interpolation=cv2.INTER_AREA) for im in imgs]
    rows_n = int(np.ceil(len(tiles) / cols))
    mosaic = np.zeros((rows_n * cell, cols * cell, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        mosaic[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = t

    h, w = mosaic.shape[:2]
    after = mosaic.copy()
    rng = np.random.default_rng(2026)
    inserted = []
    for _ in range(7):                     # 模拟"新增建筑"：从已有建筑密集处取块贴到空白处
        bh, bw = int(rng.integers(60, 130)), int(rng.integers(70, 150))
        sy = int(rng.integers(0, max(1, h - bh)))
        sx = int(rng.integers(0, max(1, w - bw)))
        src = mosaic[sy:sy + bh, sx:sx + bw].astype(np.float32)
        if src.std() < 12:                 # 纹理太弱的区域不作为"建筑源"
            continue
        dy = int(rng.integers(0, max(1, h - bh)))
        dx = int(rng.integers(0, max(1, w - bw)))
        if abs(dy - sy) < bh and abs(dx - sx) < bw:
            continue
        after[dy:dy + bh, dx:dx + bw] = np.clip(src * 1.06 + 12, 0, 255).astype(np.uint8)
        inserted.append((dx, dy, bw, bh))

    lon0, lat0 = 126.6320, 45.7520
    gsd = 0.30                                  # 0.3 m/px，与城市航拍影像量级一致
    m_per_deg_lon = 111320.0 * np.cos(np.radians(lat0))
    transform = (gsd / m_per_deg_lon, 0.0, lon0, 0.0, -gsd / 110540.0, lat0)

    pa = out / "satellite_before.tif"
    pb = out / "satellite_after.tif"
    Raster.from_array(mosaic, transform, crs=WGS84).save(pa, dtype="uint8")
    Raster.from_array(after, transform, crs=WGS84).save(pb, dtype="uint8")
    cv2.imwrite(str(out / "preview_before.png"), mosaic[:, :, ::-1])
    cv2.imwrite(str(out / "preview_after.png"), after[:, :, ::-1])
    info = {"before": str(pa), "after": str(pb), "size": [w, h], "gsd_m": gsd,
            "inserted": inserted, "source": DATASET, "split": split, "n": len(imgs)}
    (out / "sample.json").write_text(json.dumps(info, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(f"  GeoTIFF：{pa} / {pb}　尺寸 {w}×{h}　分辨率 {gsd} m/px　"
          f"新增建筑块 {len(inserted)} 处")
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/real")
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--fallback-synthetic", action="store_true", default=True)
    a = ap.parse_args()
    out = Path(a.out)
    try:
        build(out, a.n, a.split)
    except Exception as e:  # noqa: BLE001
        print("获取真实影像失败：", e)
        if not a.fallback_synthetic:
            return 1
        print("回退到程序化合成数据…")
        out.mkdir(parents=True, exist_ok=True)
        make_demo_geotiff(out / "satellite_before.tif", size=(2048, 1536), with_change=False)
        make_demo_geotiff(out / "satellite_after.tif", size=(2048, 1536), with_change=True)
        print("已生成合成演示数据 ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
