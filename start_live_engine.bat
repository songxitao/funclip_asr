@echo off
chcp 65001 >nul
title Qwen3 Real-time Subtitles

if not defined CONDA_PREFIX (
    if exist "D:\program files\Miniconda\Scripts\activate.bat" (
        echo Activating Env: asr_ui_env...
        call "D:\program files\Miniconda\Scripts\activate.bat" asr_ui_env
    ) else (
        where python >nul 2>nul
        if errorlevel 1 (
            echo [WARNING] Python not found in PATH. Please activate your conda environment first.
            pause
            exit /b 1
        )
        echo [WARNING] Conda not found at configured path, using system python.
        echo To customize, please set config.json.
    )
) else (
    echo Conda env already active: %CONDA_PREFIX%
)

echo.
echo  Launching Engine...
echo  Command: python "%~dp0app_live_ws.py" --mode mic --lang auto
echo ==========================================
python "%~dp0app_live_ws.py" --mode mic --lang auto

if %errorlevel% neq 0 (
    echo.
    echo Error occurred! Press any key to exit...
    pause
)
