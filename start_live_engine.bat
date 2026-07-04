@echo off
chcp 65001 >nul
title Qwen3 Real-time Subtitles
echo ==========================================
echo  🚀 Activating Env: asr_ui_env...
echo ==========================================
call "D:\program files\Miniconda\Scripts\activate.bat" asr_ui_env

echo.
echo  🚀 Launching Engine...
echo  Command: python "E:\project\funclip-pro\app_live_ws.py" --mode mic --lang auto
echo ==========================================
python "E:\project\funclip-pro\app_live_ws.py" --mode mic --lang auto

if %errorlevel% neq 0 (
    echo.
    echo ❌ Error occurred! Press any key to exit...
    pause
)
