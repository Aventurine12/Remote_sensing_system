# -*- coding: utf-8 -*-
"""语义分割微调脚本：U-Net（ResNet 编码器）在建筑提取数据集上训练。

数据目录约定（与 WHU Building / Massachusetts Buildings 等公开集转换后一致）：

    dataset/
      train/images/*.png   train/masks/*.png     # mask: 0=背景 255=建筑
      val/images/*.png     val/masks/*.png

    python tools/train_segmentation.py --data D:/data/buildings --epochs 30 --size 512

训练完成后权重保存到 weights/unet_building.pth，系统会自动优先加载它
（配置 models.segmentation.backend 改成 smp_unet 即可）。
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

from aerosky.utils.vis import iou_f1


class BuildingDataset(Dataset):
    def __init__(self, root: Path, split: str, size: int = 512, augment: bool = False):
        self.img_dir = root / split / "images"
        self.msk_dir = root / split / "masks"
        self.files = sorted([p for p in self.img_dir.glob("*")
                             if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")])
        if not self.files:
            raise FileNotFoundError(f"{self.img_dir} 下没有影像")
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)[:, :, ::-1]
        msk = None
        for ext in (".png", ".jpg", ".tif", ".tiff"):
            p = self.msk_dir / (f.stem + ext)
            if p.exists():
                msk = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                break
        if msk is None:
            msk = np.zeros(img.shape[:2], np.uint8)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            if np.random.rand() < 0.5:
                img, msk = img[:, ::-1], msk[:, ::-1]
            if np.random.rand() < 0.5:
                img, msk = img[::-1], msk[::-1]
            k = np.random.randint(4)
            img, msk = np.rot90(img, k), np.rot90(msk, k)
            img = np.clip(img.astype(np.float32) * np.random.uniform(0.85, 1.15), 0, 255)
        x = (img.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406], np.float32)) / \
            np.array([0.229, 0.224, 0.225], np.float32)
        y = (msk > 127).astype(np.float32)
        return torch.from_numpy(x.transpose(2, 0, 1)), torch.from_numpy(y)[None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="weights/unet_building.pth")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()

    import segmentation_models_pytorch as smp
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备：{dev}")

    root = Path(a.data)
    tr = BuildingDataset(root, "train", a.size, augment=True)
    va = BuildingDataset(root, "val", a.size, augment=False) if (root / "val").exists() else tr
    dl_tr = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=a.workers)
    dl_va = DataLoader(va, batch_size=max(1, a.batch // 2), shuffle=False, num_workers=a.workers)
    print(f"训练 {len(tr)} 张，验证 {len(va)} 张")

    model = smp.Unet(encoder_name=a.encoder, encoder_weights="imagenet",
                     in_channels=3, classes=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    dice = smp.losses.DiceLoss(mode="binary")
    bce = nn.BCEWithLogitsLoss()
    best = 0.0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            logit = model(x)
            loss = bce(logit, y) + dice(logit, y)
            loss.backward()
            opt.step()
            tot += float(loss.item())
        sched.step()
        model.eval()
        metrics = []
        with torch.no_grad():
            for x, y in dl_va:
                pr = (torch.sigmoid(model(x.to(dev))) > 0.5).cpu().numpy()
                gt = (y.numpy() > 0.5).astype(np.uint8)
                for i in range(pr.shape[0]):
                    metrics.append(iou_f1(pr[i, 0].astype(np.uint8), gt[i, 0]))
        iou = float(np.mean([m["IoU"] for m in metrics]))
        f1 = float(np.mean([m["F1"] for m in metrics]))
        print(f"epoch {ep:3d}/{a.epochs}  loss={tot / max(len(dl_tr), 1):.4f}  "
              f"IoU={iou:.4f}  F1={f1:.4f}  ({time.time() - t0:.0f}s)")
        if iou >= best:
            best = iou
            torch.save({"state_dict": model.state_dict(), "encoder": a.encoder,
                        "iou": iou, "f1": f1}, a.out)
            print(f"    已保存 -> {a.out}（最好 IoU {best:.4f}）")
    print(f"训练结束，最佳验证 IoU {best:.4f}，权重：{a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
