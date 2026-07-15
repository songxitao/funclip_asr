@echo off
chcp 65001 >nul
title FunClip Pro ASR API Service

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

echo [INFO] ÕýÔÚÆô¶¯ asr_onnx_service.py...
python "%~dp0asr_onnx_service.py"
pause
