@echo off
title Qwen3-ASR Service Manager
echo [INFO] 正在检查 Qwen3-ASR 容器状态...

docker ps -a -q -f "name=qwen3-asr" | findstr . >nul
if %errorlevel% neq 0 goto :CREATE_CONTAINER

docker ps -q -f "name=qwen3-asr" | findstr . >nul
if %errorlevel% == 0 goto :RUNNING

:START_CONTAINER
echo [INFO] 容器已存在但已停止。正在启动...
docker start qwen3-asr
goto :CHECK_START

:CREATE_CONTAINER
echo [INFO] 找不到容器。正在创建并启动新容器...
echo [INFO] 正在配置端口 28000 和 GPU 显卡访问权限...

docker run -d ^
  --name qwen3-asr ^
  --gpus all ^
  -p 28000:80 ^
  --shm-size 10g ^
  -v "%~dp0model\models\Qwen:/data/shared" ^
  -v "%~dp0qwen_server:/app/server" ^
  qwenllm/qwen3-asr:latest ^
  sleep infinity

if %errorlevel% neq 0 (
    echo [ERROR] 容器创建失败。请检查 Docker 是否已启动及镜像是否存在。
    pause
    exit /b
)
goto :LAUNCH_SERVER

:CHECK_START
if %errorlevel% neq 0 (
    echo [ERROR] 无法启动容器 "qwen3-asr"。
    pause
    exit /b
)
echo [INFO] 正在等待容器初始化 (5秒)...
timeout /t 5 >nul
goto :LAUNCH_SERVER

:RUNNING
echo [INFO] Qwen 容器已经处于运行状态。

:LAUNCH_SERVER
echo [INFO] 正在容器内启动 ASR 服务端程序...
echo [INFO] 提示: 请保持此窗口打开以维持服务运行。
docker exec qwen3-asr python3 /app/server/custom_server.py

echo.
echo [WARN] 服务端进程已退出。
pause