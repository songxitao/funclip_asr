@echo off
chcp 65001 >nul
title FunClip Pro 本地实时字幕

if not defined CONDA_PREFIX (
    where python >nul 2>nul
    if errorlevel 1 (
        echo [WARNING] Python not found in PATH. Please activate your conda environment first.
        pause
        exit /b 1
    )
    echo [INFO] Using system python from PATH.
) else (
    echo [INFO] Conda env already active: %CONDA_PREFIX%
)

echo [INFO] 正在运行 app_live_local.py...
python "%~dp0app_live_local.py" --overlay
pause
