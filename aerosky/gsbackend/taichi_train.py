# -*- coding: utf-8 -*-
"""Taichi 3D 高斯泼溅训练入口（跑在装了 taichi + CUDA torch 的解释器里）。

这是对接现成实现 `wanmeihuali/taichi_3d_gaussian_splatting` 的适配脚本：
    1. 读 COLMAP 稀疏模型，转成它要求的 train.json / val.json / point_cloud.parquet
    2. 生成训练配置 yaml
    3. 调用它的 gaussian_point_train.py 做训练（Taichi 语言 + CUDA 后端的 3DGS）
    4. 用它的 parquet_to_ply.py 导出标准 3DGS .ply，并渲染预览图

    python taichi_train.py --images <照片目录> --sparse <sparse/0> --out <输出目录>
                          --repo <taichi 仓库路径> --iters 7000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gsplat_train import read_colmap, _cam_params  # noqa: E402  （纯 numpy，无额外依赖）

DEFAULT_CONFIG = {
    "num-iterations": 7000,
    "log-loss-interval": 10,
    "log-metrics-interval": 200,
    "log-image-interval": 1000,
    "val-interval": 2000,
    "print-metrics-to-console": True,
    "enable-taichi-kernel-profiler": False,
    "increase-color-max-sh-band-interval": 1000.0,
    "feature_learning_rate": 0.005,
    "position_learning_rateo": 5e-5,
    "position_learning_rate_decay_rate": 0.9947,
    "position_learning_rate_decay_interval": 100,
    "loss-function-config": {"lambda-value": 0.2, "enable_regularization": False,
                             "regularization_weight": 0.005},
    "rasterisation-config": {"near-plane": 0.8, "far-plane": 1000.0,
                             "depth-to-sort-key-scale": 100.0},
    "gaussian-point-cloud-scene-config": {**{k: v for k, v in {
        "num-of-features": 56, "max_initial_covariance": 3000.0,
        "initial_covariance_ratio": 0.1, "initial_alpha": 0.05, "add_sphere": True,
        # ★ 关键：这个实现的高斯缓冲区是**预分配定长**的，新点只能填进被剪枝腾出的空槽。
        #   这个比值就是预留多少倍的空槽。设成 None 会把致密化的路彻底堵死
        #   （实测日志里 num_fillable_densify_points 恒为 0，高斯数从头到尾不变）。
        "max-num-points-ratio": 10.0}.items()}},
    "adaptive-controller-config": {
        "densification-view-space-position-gradients-threshold": 3e-6,
        "densification_view_avg_space_position_gradients_threshold": 1e3,
        "densification_multi_frame_view_space_position_gradients_threshold": 1e3,
        "densification_multi_frame_view_pixel_avg_space_position_gradients_threshold": 1e3,
        "densification_multi_frame_position_gradients_threshold": 1e3,
        "gaussian-split-factor-phi": 1.6,
        "num-iterations-densify": 100,
        "num-iterations-reset-alpha": 4000,
        "num-iterations-warm-up": 1000,
        "reset-alpha-value": -1.9,
        "transparent-alpha-threshold": -2.0,
        "floater_num_pixels_threshold": 400000,
        "floater_near_camrea_num_pixels_threshold": 300000,
        "iteration_start_remove_floater": 2000,
        "under_reconstructed_num_pixels_threshold": 32,
        "enable_sample_from_point": True,
        "under_reconstructed_move_factor": 10.0,
    },
}


# --------------------------------------------------------------------- 数据准备
def prepare_dataset(images_dir: Path, sparse_dir: Path, work: Path,
                    max_side: int = 1000, val_every: int = 8,
                    white_background: bool = False) -> dict:
    """COLMAP -> Taichi 训练格式；顺带把影像降采样并同步缩放内参。"""
    import pandas as pd
    from PIL import Image

    work.mkdir(parents=True, exist_ok=True)
    img_out = work / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    cams, imgs, pts, cols = read_colmap(sparse_dir)
    if not imgs:
        raise RuntimeError(f"{sparse_dir} 里没有影像位姿")
    print(f"[prepare] COLMAP：{len(imgs)} 个视角，{pts.shape[0]} 个点")

    records, kept = [], 0
    for iid, q, t, cid, name in imgs:
        if cid not in cams:
            continue
        src = images_dir / name
        if not src.exists():
            cand = list(images_dir.rglob(Path(name).name))
            if not cand:
                print(f"  缺少影像 {name}，跳过")
                continue
            src = cand[0]
        model, w, h, params = cams[cid]
        fx, fy, cx, cy = _cam_params(model, params)

        im = Image.open(src).convert("RGB")
        W, H = im.size
        scale = 1.0
        if max_side and max(W, H) > max_side:
            scale = max_side / max(W, H)
            im = im.resize((max(1, int(W * scale)), max(1, int(H * scale))), Image.LANCZOS)
        # COLMAP 的 width/height 若与实际影像不一致，也要按比例校正
        if (w, h) != (W, H):
            sx, sy = W / max(w, 1), H / max(h, 1)
            fx, cx = fx * sx, cx * sx
            fy, cy = fy * sy, cy * sy
        dst = img_out / (Path(name).stem + ".jpg")
        im.save(dst, quality=92)
        W2, H2 = im.size
        K = np.array([[fx * scale, 0, cx * scale],
                      [0, fy * scale, cy * scale],
                      [0, 0, 1]], np.float64)

        # Taichi 版要的是「点云 -> 相机」的逆，也就是相机->世界（c2w）
        R_wc = _q2r(q)                      # 世界->相机
        c2w = np.eye(4)
        c2w[:3, :3] = R_wc.T
        c2w[:3, 3] = -R_wc.T @ np.asarray(t, np.float64)
        records.append({
            "image_path": str(dst.resolve()),
            "T_pointcloud_camera": c2w.tolist(),
            "camera_intrinsics": K.tolist(),
            "camera_height": int(H2),
            "camera_width": int(W2),
            "camera_id": int(iid),
        })
        kept += 1

    df = pd.DataFrame(records)
    df["is_train"] = df.index % val_every != 0
    train_df = df[df["is_train"]].drop(columns=["is_train"]).copy()
    val_df = df[~df["is_train"]].drop(columns=["is_train"]).copy()
    train_df.to_json(work / "train.json", orient="records")
    val_df.to_json(work / "val.json", orient="records")

    pc = pd.DataFrame({"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2],
                       "r": cols[:, 0], "g": cols[:, 1], "b": cols[:, 2]})
    pc.to_parquet(work / "point_cloud.parquet")
    print(f"[prepare] 训练 {len(train_df)} / 验证 {len(val_df)}，影像缩放系数 {scale:.3f}，"
          f"输出 {work}")
    return {"train": len(train_df), "val": len(val_df), "points": int(pts.shape[0]),
            "work": str(work), "scale": scale}


def _q2r(q):
    q = np.asarray(q, np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def write_config(path: Path, work: Path, out_dir: Path, iters: int) -> dict:
    import yaml
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg["num-iterations"] = int(iters)
    # 关键：这个实现的模型**只在验证时落盘**，val-interval 必须小于总迭代数，
    # 否则跑完一次模型都不会保存（踩过的坑）
    cfg["val-interval"] = max(100, min(2000, iters // 4))
    cfg["log-metrics-interval"] = max(50, cfg["val-interval"] // 4)
    cfg["log-loss-interval"] = max(10, min(50, iters // 40))
    cfg["adaptive-controller-config"]["num-iterations-warm-up"] = max(
        100, min(1000, iters // 8))
    cfg["pointcloud-parquet-path"] = str((work / "point_cloud.parquet").resolve())
    cfg["train-dataset-json-path"] = str((work / "train.json").resolve())
    cfg["val-dataset-json-path"] = str((work / "val.json").resolve())
    cfg["summary-writer-log-dir"] = str((out_dir / "logs").resolve())
    cfg["output-model-dir"] = str((out_dir / "model").resolve())
    # 上游实现只在 output_model_dir 为 None 时才创建目录（bug），
    # 这里替它把两个目录都建好，否则训练到验证阶段保存模型会直接报错
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "model").mkdir(parents=True, exist_ok=True)
    cfg.pop("v", None)
    # ★ 这里绝不能再把 max-num-points-ratio 覆盖成 None：
    #   该字段决定预分配多少倍的空槽给致密化用，None = 一个点都加不进去（曾经的 bug）
    ratio = cfg["gaussian-point-cloud-scene-config"].get("max-num-points-ratio")
    assert ratio is None or ratio > 1.0, "max-num-points-ratio 必须 >1，否则致密化失效"
    path.write_text(yaml.dump(cfg, allow_unicode=True, default_flow_style=False),
                    encoding="utf-8")
    return cfg


# --------------------------------------------------------------------- 训练
def run_training(repo: Path, config_path: Path, cwd: Path) -> int:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [sys.executable, str(repo / "gaussian_point_train.py"),
           "--train_config", str(config_path.resolve())]
    print("[train] " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(cwd)).returncode


def export_ply(repo: Path, model_dir: Path, out_ply: Path) -> str:
    """把训练出的 parquet 模型转成标准 3DGS .ply。"""
    cands = []
    for d in (model_dir, model_dir.parent / "logs"):
        if d.exists():
            cands += list(d.rglob("*.parquet"))
    cands = [p for p in cands if "point_cloud" not in p.name.lower()]  # 排除输入的点云
    cands = sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        print("[export] 没找到 parquet 模型")
        return ""
    src = cands[0]
    print(f"[export] {src} -> {out_ply}")
    code = (
        "import sys, argparse\n"
        "from taichi_3d_gaussian_splatting.GaussianPointCloudScene import GaussianPointCloudScene\n"
        f"s = GaussianPointCloudScene.from_parquet(r'{src}', "
        "config=GaussianPointCloudScene.PointCloudSceneConfig(max_num_points_ratio=None))\n"
        f"s.to_ply(r'{out_ply}')\n"
        "print('ok')\n")
    tmp = out_ply.parent / "_export_ply.py"
    tmp.write_text(code, encoding="utf-8")
    r = subprocess.run([sys.executable, str(tmp)], cwd=str(repo))
    return str(out_ply) if r.returncode == 0 and out_ply.exists() else ""


def count_gaussians(ply: Path) -> int:
    try:
        with open(ply, "rb") as f:
            for _ in range(40):
                line = f.readline().decode("ascii", "ignore").strip()
                if line.startswith("element vertex"):
                    return int(line.split()[-1])
                if line == "end_header":
                    break
    except Exception:  # noqa: BLE001
        pass
    return 0


def render_previews(ply: Path, out_dir: Path, n: int = 6) -> list:
    """用系统内置的 numpy 预览器渲染环绕图（只是看图，不参与训练）。"""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from aerosky.gsplat.camera import OrbitCamera
        from aerosky.gsplat.ply_io import ply_to_gaussians
        from aerosky.gsplat.render_np import SplatRenderer
        import cv2
        d = ply_to_gaussians(ply)
        deg = int(round(math.sqrt(d["shs"].shape[1]))) - 1
        r = SplatRenderer(d["means"], d["scales"], d["quats"], d["opacities"], d["shs"], deg)
        c, s = r.center_scale()
        paths = []
        for i in range(n):
            ang = 2 * math.pi * i / n
            e = math.radians(20)
            cam = OrbitCamera(target=c, distance=s * 1.3, yaw=math.degrees(ang),
                              pitch=20.0).to_camera(640, 480)
            img, _ = r.render(cam)
            p = out_dir / f"orbit_{i:02d}.png"
            cv2.imwrite(str(p), img[:, :, ::-1])
            paths.append(str(p))
        print(f"[preview] {len(paths)} 张环绕图")
        return paths
    except Exception as e:  # noqa: BLE001
        print(f"[preview] 渲染失败：{e}")
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--sparse", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=os.environ.get("TAICHI_3DGS_REPO",
                                                     r"D:\Code\dc\3dgs_repos\taichi_3d_gaussian_splatting"))
    ap.add_argument("--iters", type=int, default=7000)
    ap.add_argument("--downscale", type=int, default=1, help="影像最长边再除以该值")
    ap.add_argument("--max-side", type=int, default=None, help="影像最长边缩到多少像素")
    ap.add_argument("--previews", type=int, default=6)
    ap.add_argument("--time-budget", type=float, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(a.repo)
    if not (repo / "gaussian_point_train.py").exists():
        print(f"[error] 找不到仓库：{repo}")
        return 2

    import torch
    print(f"[env] python {sys.version.split()[0]}  torch {torch.__version__}  "
          f"CUDA={torch.cuda.is_available()}  "
          f"设备={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    import taichi as ti
    print(f"[env] taichi {ti.__version__}")

    max_side = a.max_side or (1000 // max(1, a.downscale))
    t0 = time.time()
    info = prepare_dataset(Path(a.images), Path(a.sparse), out / "taichi_data",
                           max_side=max_side)
    cfg_path = out / "train_config.yaml"
    write_config(cfg_path, out / "taichi_data", out, a.iters)
    print(f"[config] {cfg_path}")

    rc = run_training(repo, cfg_path, cwd=repo)
    seconds = time.time() - t0
    ply = export_ply(repo, out / "model", out / "point_cloud.ply")
    n = count_gaussians(Path(ply)) if ply else 0
    previews = render_previews(Path(ply), out, a.previews) if ply else []

    metrics = {"backend": "taichi", "num_gaussians": n, "iterations": a.iters,
               "device": a.device, "train_seconds": round(seconds, 1),
               "exit_code": rc, "dataset": info,
               "ply": ply, "previews": previews, "log_file": str(cfg_path)}
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(f"[done] 高斯 {n}，耗时 {seconds:.1f}s，模型 {ply}")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
