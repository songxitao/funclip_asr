@echo off
title Stop Qwen3-ASR Service
echo [INFO] 正在停止 Qwen3-ASR 容器...

docker stop qwen3-asr
docker rm qwen3-asr

echo.
echo [SUCCESS] Qwen3-ASR 容器已成功停止并清理。
echo [INFO] GPU 显存已完全释放。
timeout /t 3