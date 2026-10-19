# ============================================================
# aerosky —— 一键搭建开发环境（Windows / Python 3.9）
#   powershell -ExecutionPolicy Bypass -File tools\setup_env.ps1
#
# 默认装 CPU 版 PyTorch（PyPI 上的 Windows 轮子即 CPU 版，约 200MB，国内镜像几分钟装完），
# 系统所有功能都能跑；想把推理/三维重建换成 GPU 加速，装完后再执行：
#   powershell -ExecutionPolicy Bypass -File tools\upgrade_to_cuda.ps1
# ============================================================
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Py   = Join-Path $Root ".venv\Scripts\python.exe"
$TUNA = "https://pypi.tuna.tsinghua.edu.cn/simple"
New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs") | Out-Null
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

function Step($name, $block) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $block
    Write-Host ("--- $name 完成，用时 {0:N1}s" -f $sw.Elapsed.TotalSeconds) -ForegroundColor Green
}

if (-not (Test-Path $Py)) { throw "未找到虚拟环境 $Py，请先执行：python -m venv `"$Root\.venv`"" }

Step "PyTorch（CPU 版，约 200MB）" {
    & $Py -m pip install -i $TUNA "numpy==1.26.4"
    & $Py -m pip install -i $TUNA "torch==2.2.2" "torchvision==0.17.2"
}

Step "地理空间库（GDAL 能力）" {
    & $Py -m pip install -i $TUNA "rasterio==1.3.11" "pyproj==3.6.1" "shapely==2.0.7" "affine"
}

Step "GUI 与三维" {
    & $Py -m pip install -i $TUNA "PyQt5==5.15.11"
}

Step "深度学习模型库" {
    & $Py -m pip install -i $TUNA "opencv-python==4.10.0.84" "Pillow==10.4.0"
    & $Py -m pip install -i $TUNA "timm==1.0.11" "einops==0.8.0"
    & $Py -m pip install -i $TUNA "scikit-image==0.24.0" "scikit-learn==1.6.1"
    & $Py -m pip install -i $TUNA "ultralytics==8.3.40" "huggingface-hub"
    & $Py -m pip install -i $TUNA "segmentation-models-pytorch==0.3.4"
}

Step "杂项" {
    & $Py -m pip install -i $TUNA "requests" "tqdm" "pyyaml" "matplotlib==3.9.2" "pandas==2.2.3" "psutil"
    & $Py -m pip install -i $TUNA "plyfile" "openpyxl"
}

Write-Host "`n=== 环境自检 ===" -ForegroundColor Cyan
& $Py (Join-Path $PSScriptRoot "check_env.py")
