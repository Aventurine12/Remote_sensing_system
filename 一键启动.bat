@echo off
chcp 65001 >nul
title 空天协同遥感三维监测系统
cd /d "%~dp0"

rem ============ 1) 找一个可用的 Python 解释器 ============
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY (
    where python >nul 2>nul
    if not errorlevel 1 set "PY=python"
)
if not defined PY goto :nopython

rem ============ 2) 先检测本机环境（快速模式，约 0.3 秒）============
"%PY%" tools\env_doctor.py --quick --quiet
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" goto :launch
if "%RC%"=="4" goto :oldpython

rem ============ 3) 环境不完整，才安装依赖 ============
echo.
echo 检测结果：本机运行环境不完整，现在开始安装依赖。
echo （如果只是缺个别组件，pip 会自动只补缺的部分）
echo.
call "安装依赖.bat"
if not exist ".venv\.deps_ok" (
    echo.
    echo [错误] 依赖安装未成功完成，请查看上面的输出后重试。
    pause
    exit /b 1
)

:launch
echo.
echo 正在启动图形界面，请稍候 ...
".venv\Scripts\python.exe" run_app.py %*
if errorlevel 1 (
    echo.
    echo 程序异常退出。可以双击「检测环境.bat」查看环境哪里有问题。
    pause
)
exit /b 0

:oldpython
echo.
echo [错误] 本机 Python 版本过低，本系统需要 3.9 或更高版本。
echo        请到 https://www.python.org/downloads/ 安装新版 Python
echo        （安装时勾选 "Add Python to PATH"），然后重新双击本文件。
echo.
pause
exit /b 1

:nopython
echo.
echo [错误] 本机没有找到 Python，无法运行本系统。
echo        请先安装 Python 3.9 或更高版本，安装时务必勾选 "Add Python to PATH"。
echo        下载地址: https://www.python.org/downloads/
echo.
pause
exit /b 1