@echo off
chcp 65001 >nul
title 安装依赖 - 空天协同遥感三维监测系统
cd /d "%~dp0"
set PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

echo ============================================================
echo   空天协同的多光谱遥感影像智能解译与三维监测系统
echo   首次使用：创建运行环境并安装依赖
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 没有找到 Python 命令。
    echo        请先安装 Python 3.9 或更高版本，
    echo        安装时务必勾选 "Add Python to PATH"。
    echo        下载地址: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] 创建虚拟环境 .venv ...
    python -m venv .venv
    if errorlevel 1 ( echo [错误] 创建虚拟环境失败 & pause & exit /b 1 )
) else (
    echo [1/3] 虚拟环境已存在，跳过
)

echo.
echo [2/3] 升级 pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip -i %PIP_INDEX% --quiet

echo.
echo [3/3] 安装依赖（约 2 GB，可能需要 10~30 分钟，请耐心等待）...
".venv\Scripts\python.exe" -m pip install -i %PIP_INDEX% -r requirements.txt
if errorlevel 1 (
    echo.
    echo [错误] 依赖安装失败。请检查网络后重新运行本脚本（pip 支持续传）。
    pause
    exit /b 1
)

echo.
echo ------------------------------------------------------------
echo   安装完成，正在校验依赖是否齐全 ...
echo ------------------------------------------------------------
".venv\Scripts\python.exe" -c "import torch, torchvision, cv2, numpy, yaml, rasterio, shapely, pyproj, PyQt5, ultralytics, matplotlib, plyfile"
if errorlevel 1 (
    echo.
    echo [错误] 关键依赖仍缺失，安装未完成。
    echo        请重新运行本脚本（pip 支持断点续传，已下过的不会重复下载）。
    pause
    exit /b 1
)
echo ok> ".venv\.deps_ok"

echo.
echo ------------------------------------------------------------
echo   环境自检 ...
echo ------------------------------------------------------------
".venv\Scripts\python.exe" run_app.py --check

echo.
echo 全部就绪！以后双击 "一键启动.bat" 即可运行。
pause