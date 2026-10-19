# -*- coding: utf-8 -*-
"""环境自检：报告 aerosky 依赖是否就绪、GPU 是否可用。"""
import importlib
import platform
import sys

MODULES = [
    ("numpy", "numpy"), ("torch", "torch"), ("torchvision", "torchvision"),
    ("cv2", "opencv"), ("PIL", "Pillow"), ("rasterio", "rasterio"),
    ("pyproj", "pyproj"), ("shapely", "shapely"), ("PyQt5.QtWidgets", "PyQt5"),
    ("plyfile", "plyfile"), ("timm", "timm"),
    ("segmentation_models_pytorch", "smp"), ("ultralytics", "ultralytics"),
    ("einops", "einops"), ("skimage", "scikit-image"), ("sklearn", "scikit-learn"),
    ("scipy", "scipy"), ("matplotlib", "matplotlib"), ("pandas", "pandas"),
    ("yaml", "pyyaml"), ("huggingface_hub", "huggingface-hub"),
]


def main():
    print("Python  :", sys.version.split()[0], platform.architecture()[0])
    print("Exe     :", sys.executable)
    bad = []
    for mod, label in MODULES:
        try:
            m = importlib.import_module(mod)
            print("  OK   %-28s %s" % (label, getattr(m, "__version__", "")))
        except Exception as e:  # noqa: BLE001
            bad.append(label)
            print("  MISS %-28s %s" % (label, e.__class__.__name__))
    try:
        import torch
        print("torch   :", torch.__version__, "| cuda build:", torch.version.cuda)
        print("cuda可用:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU     :", torch.cuda.get_device_name(0),
                  "%.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1024 ** 3))
    except Exception as e:  # noqa: BLE001
        print("torch 自检失败:", e)
    print("\n缺失：", ", ".join(bad) if bad else "无")
    return 0


if __name__ == "__main__":
    sys.exit(main())
