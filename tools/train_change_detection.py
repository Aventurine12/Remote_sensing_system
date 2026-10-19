# -*- coding: utf-8 -*-
"""变化检测微调脚本：SiamUnet-diff 在双时相数据集（WHU-CD / LEVIR-CD / CDD）上训练。

数据目录约定（CDD / LEVIR-CD 转换后的通用结构）：

    dataset/
      train/A/*.png   train/B/*.png   train/label/*.png   # label: 0/255
      val/A/*.png     val/B/*.png     val/label/*.png

    python tools/train_change_detection.py --data D:/data/WHU-CD --epochs 40 --size 256

权重保存到 weights/siamunet_diff_whucd.pth；把配置里的
models.change_detection.backend 改成 siamunet_diff 即可启用。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from aerosky.inference.change_detection import _make_siamunet
from aerosky.utils.vis import recall_precision_at_mask

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class ChangeDataset(Dataset):
    def __init__(self, root: Path, split: str, size: int = 256, augment: bool = False):
        self.a = root / split / "A"
        self.b = root / split / "B"
        self.l = root / split / "label"
        if not self.a.exists():
            self.l = root / split / "labels"
        self.files = sorted([p for p in self.a.glob("*")
                             if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")])
        if not self.files:
            raise FileNotFoundError(f"{self.a} 下没有影像")
        self.size, self.augment = size, augment

    def __len__(self):
        return len(self.files)

    def _read(self, folder: Path, f: Path):
        for ext in (".png", ".jpg", ".tif", ".tiff"):
            p = folder / (f.stem + ext)
            if p.exists():
                return cv2.imread(str(p), cv2.IMREAD_COLOR)[:, :, ::-1]
        return None

    def __getitem__(self, i):
        f = self.files[i]
        ia = self._read(self.a, f)
        ib = self._read(self.b, f)
        lb = self._read(self.l, f)
        if ib is None:
            ib = ia.copy()
        if lb is None:
            lb = np.zeros_like(ia)
        ia = cv2.resize(ia, (self.size, self.size), interpolation=cv2.INTER_AREA)
        ib = cv2.resize(ib, (self.size, self.size), interpolation=cv2.INTER_AREA)
        lab = cv2.resize(lb[:, :, 0], (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            if np.random.rand() < 0.5:
                ia, ib, lab = ia[:, ::-1], ib[:, ::-1], lab[:, ::-1]
            if np.random.rand() < 0.5:
                ia, ib, lab = ia[::-1], ib[::-1], lab[::-1]
            k = np.random.randint(4)
            ia, ib, lab = np.rot90(ia, k), np.rot90(ib, k), np.rot90(lab, k)
            g = np.random.uniform(0.88, 1.12)
            ia = np.clip(ia * g, 0, 255)
            ib = np.clip(ib * np.random.uniform(0.88, 1.12), 0, 255)
        ia = (ia.astype(np.float32) / 255.0 - MEAN) / STD
        ib = (ib.astype(np.float32) / 255.0 - MEAN) / STD
        y = (lab > 127).astype(np.float32)
        return (torch.from_numpy(ia.transpose(2, 0, 1).copy()),
                torch.from_numpy(ib.transpose(2, 0, 1).copy()),
                torch.from_numpy(y)[None])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="weights/siamunet_diff_whucd.pth")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备：{dev}")
    root = Path(a.data)
    tr = ChangeDataset(root, "train", a.size, augment=True)
    va = ChangeDataset(root, "val", a.size) if (root / "val").exists() else tr
    dl_tr = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=a.workers)
    dl_va = DataLoader(va, batch_size=a.batch, shuffle=False, num_workers=a.workers)
    print(f"训练 {len(tr)} 对，验证 {len(va)} 对")

    model = _make_siamunet(a.base).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    bce = nn.BCEWithLogitsLoss()
    best = 0.0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for xa, xb, y in dl_tr:
            xa, xb, y = xa.to(dev), xb.to(dev), y.to(dev)
            opt.zero_grad()
            loss = bce(model(xa, xb), y)
            loss.backward()
            opt.step()
            tot += float(loss.item())
        sched.step()
        model.eval()
        ms = []
        with torch.no_grad():
            for xa, xb, y in dl_va:
                pr = (torch.sigmoid(model(xa.to(dev), xb.to(dev))) > 0.5).cpu().numpy()
                gt = (y.numpy() > 0.5).astype(np.uint8)
                for i in range(pr.shape[0]):
                    ms.append(recall_precision_at_mask(pr[i, 0].astype(np.uint8), gt[i, 0]))
        f1 = float(np.mean([m["F1"] for m in ms]))
        miou = float(np.mean([m["IoU"] for m in ms]))
        miss = float(np.mean([m["漏检率"] for m in ms]))
        fa = float(np.mean([m["误检率"] for m in ms]))
        print(f"epoch {ep:3d}/{a.epochs}  loss={tot / max(len(dl_tr), 1):.4f}  F1={f1:.4f}  "
              f"IoU={miou:.4f}  漏检={miss:.4f}  误检={fa:.4f}  ({time.time() - t0:.0f}s)")
        if f1 >= best:
            best = f1
            torch.save({"state_dict": model.state_dict(), "base_channels": a.base,
                        "f1": f1, "iou": miou}, a.out)
            print(f"    已保存 -> {a.out}（最好 F1 {best:.4f}）")
    print(f"训练结束，最佳验证 F1 {best:.4f}，权重：{a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
