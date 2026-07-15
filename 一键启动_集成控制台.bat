@echo off
chcp 65001 >nul
title FunClip Pro 集成控制台

echo [INFO] 正在解析 config.json 提取 Python 解释器路径...

for /f "usebackq tokens=*" %%i in (`powershell -NoProfile -Command "(Get-Content -Raw -Path '%~dp0config.json' | ConvertFrom-Json).offline_python"`) do set "PYTHON_PATH=%%i"

if "%PYTHON_PATH%"=="" (
    echo [WARNING] 无法从 config.json 中解析 offline_python 路径，使用系统 PATH python
    set "PYTHON_PATH=python"
) else (
    echo [INFO] 找到 Python: %PYTHON_PATH%
)

echo [INFO] 正在运行 app_control.py...
"%PYTHON_PATH%" "%~dp0app_control.py"
pause
