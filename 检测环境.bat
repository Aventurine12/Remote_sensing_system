@echo off
chcp 65001 >nul
title 环境检测 - 空天协同遥感三维监测系统
cd /d "%~dp0"

set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY (
    where python >nul 2>nul
    if not errorlevel 1 set "PY=python"
)
if not defined PY (
    echo.
    echo [结果] 本机没有找到 Python，无法检测。
    echo        请先安装 Python 3.9 或更高版本，安装时勾选 "Add Python to PATH"。
    echo        下载地址: https://www.python.org/downloads/
    echo.
    pause
    exit /b 3
)

"%PY%" tools\env_doctor.py
pause