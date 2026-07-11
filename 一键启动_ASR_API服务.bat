@echo off
title FunClip Pro ASR API Service
echo [INFO] 正在从 config.json 读取 Conda 路径和环境...

for /f "usebackq tokens=*" %%i in (`powershell -NoProfile -Command "(Get-Content -Raw -Path '%~dp0config.json' | ConvertFrom-Json).conda_root"`) do set "CONDA_ROOT=%%i"
for /f "usebackq tokens=*" %%i in (`powershell -NoProfile -Command "(Get-Content -Raw -Path '%~dp0config.json' | ConvertFrom-Json).live_env_name"`) do set "LIVE_ENV_NAME=%%i"

if "%CONDA_ROOT%"=="" (
    echo [ERROR] 无法从 config.json 读取 conda_root 路径
    pause
    exit /b
)
if "%LIVE_ENV_NAME%"=="" (
    echo [ERROR] 无法从 config.json 读取 live_env_name
    pause
    exit /b
)

echo [INFO] Conda Root: %CONDA_ROOT%
echo [INFO] Live Env Name: %LIVE_ENV_NAME%

set "ACTIVATE_PATH=%CONDA_ROOT%\Scripts\activate.bat"
if not exist "%ACTIVATE_PATH%" (
    set "ACTIVATE_PATH=%CONDA_ROOT%\condabin\conda.bat"
)

if not exist "%ACTIVATE_PATH%" (
    echo [ERROR] 找不到 conda 脚本，请检查 config.json 中的 conda_root 路径是否正确。
    pause
    exit /b
)

echo [INFO] 正在激活 Conda 环境: %LIVE_ENV_NAME%...
call "%ACTIVATE_PATH%" activate %LIVE_ENV_NAME%

echo [INFO] 正在启动 asr_onnx_service.py...
python "%~dp0asr_onnx_service.py"
pause
