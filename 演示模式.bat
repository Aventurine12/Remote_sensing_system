@echo off
chcp 65001 >nul
title 端到端演示 - 空天协同遥感三维监测系统
cd /d "%~dp0"

set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY (
    where python >nul 2>nul
    if not errorlevel 1 set "PY=python"
)
if not defined PY (
    echo [错误] 本机没有找到 Python，请先安装 Python 3.9+ 并勾选 Add Python to PATH。
    pause
    exit /b 1
)

"%PY%" tools\env_doctor.py --quick --quiet
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    if "%RC%"=="4" (
        echo [错误] 本机 Python 版本过低，需要 3.9 或更高版本。
        pause
        exit /b 1
    )
    echo 检测结果：本机运行环境不完整，现在开始安装依赖。
    echo.
    call "安装依赖.bat"
    if not exist ".venv\.deps_ok" (
        echo [错误] 依赖安装未成功完成，请查看上面的输出后重试。
        pause
        exit /b 1
    )
)

echo.
echo 正在无人值守跑一遍完整流程（解译 - 任务 - 三维 - 配准 - 报告）...
echo 结果会写到 outputs\demo_run 目录。
echo.
".venv\Scripts\python.exe" run_app.py --demo --reuse-3d
echo.
echo 跑完了，成果在 outputs\demo_run
pause