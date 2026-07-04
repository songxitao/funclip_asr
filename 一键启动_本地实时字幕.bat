@echo off
title FunClip Pro 本地实时字幕
echo [INFO] 正在解析 config.json 提取 Conda 路径和环境配置...

for /f "usebackq tokens=*" %%i in (`powershell -NoProfile -Command "(Get-Content -Raw -Path '%~dp0config.json' | ConvertFrom-Json).conda_root"`) do set "CONDA_ROOT=%%i"
for /f "usebackq tokens=*" %%i in (`powershell -NoProfile -Command "(Get-Content -Raw -Path '%~dp0config.json' | ConvertFrom-Json).live_env_name"`) do set "LIVE_ENV_NAME=%%i"

if "%CONDA_ROOT%"=="" (
    echo [ERROR] 无法从 config.json 中提取 conda_root 路径
    pause
    exit /b
)
if "%LIVE_ENV_NAME%"=="" (
    echo [ERROR] 无法从 config.json 中提取 live_env_name 配置
    pause
    exit /b
)

echo [INFO] Conda Root: %CONDA_ROOT%
echo [INFO] Live Env Name: %LIVE_ENV_NAME%

set "ACTIVATE_PATH=%CONDA_ROOT%\Scriptsctivate.bat"
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

echo [INFO] 正在运行 app_live_local.py...
python "%~dp0app_live_local.py" --overlay
pause