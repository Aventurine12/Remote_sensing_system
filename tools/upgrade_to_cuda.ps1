# ============================================================
# aerosky —— 可选：把 PyTorch 换成 CUDA 版（约 2.4GB，国内直连较慢）
#   powershell -ExecutionPolicy Bypass -File tools\upgrade_to_cuda.ps1
# 支持断点续传：curl 下载到 .wheels\ 后本地安装，中断后重跑即可。
# ============================================================
param([string]$Cuda = "cu121")
$ErrorActionPreference = "Stop"
$Root  = Split-Path -Parent $PSScriptRoot
$Py    = Join-Path $Root ".venv\Scripts\python.exe"
$Whl   = Join-Path $Root ".wheels"
New-Item -ItemType Directory -Force -Path $Whl | Out-Null

$mirrors = @(
    "https://mirror.sjtu.edu.cn/pytorch-wheels/$Cuda",
    "https://download.pytorch.org/whl/$Cuda"
)
$files = @(
    "torch-2.2.2%2B$Cuda-cp39-cp39-win_amd64.whl",
    "torchvision-0.17.2%2B$Cuda-cp39-cp39-win_amd64.whl"
)

foreach ($f in $files) {
    $dest = Join-Path $Whl ($f -replace "%2B", "+")
    if ((Test-Path $dest) -and ((Get-Item $dest).Length -gt 10MB)) {
        Write-Host "已存在：$dest"
        continue
    }
    $done = $false
    foreach ($m in $mirrors) {
        Write-Host "curl 下载 $m/$f"
        curl.exe -L --retry 5 --retry-delay 3 -C - -o $dest "$m/$f"
        if ((Test-Path $dest) -and ((Get-Item $dest).Length -gt 10MB)) { $done = $true; break }
    }
    if (-not $done) { throw "下载失败：$f" }
}

Write-Host "`n安装 CUDA 版 PyTorch ..."
& $Py -m pip install --no-deps (Join-Path $Whl "torch-2.2.2+$Cuda-cp39-cp39-win_amd64.whl") `
                              (Join-Path $Whl "torchvision-0.17.2+$Cuda-cp39-cp39-win_amd64.whl")
& $Py (Join-Path $PSScriptRoot "check_env.py")
