import os
import time
import requests
import json
from .base import ASREngine

# 自动获取根目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 假设当前文件在 FunClip/core/asr/qwen.py，向上三级到根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))

SHARED_HOST_DIR = os.path.join(PROJECT_ROOT, "qwen_server", "shared_tmp")
SHARED_DOCKER_DIR = "/app/server/shared_tmp"

class QwenEngine(ASREngine):
    def __init__(self, device="cuda", **kwargs):
        super().__init__(device, **kwargs)
        self.model_name = "Qwen2-Audio-7B"

    def load_model(self):
        self.log("🚀 Qwen3 使用 Docker 服务 (無需本地加载模型)")
        # Check health
        try:
            res = requests.get(f"{DOCKER_HOST}/health", timeout=2)
            if res.status_code == 200:
                self.log("✅ Qwen3 Docker 服务在线")
            else:
                self.log(f"⚠️ Qwen3 服务状态码: {res.status_code}")
        except:
            self.log("❌ 警告: 无法连接 Qwen3 Docker 服务，请确保 start_qwen_backend.bat 已运行")

    def transcribe(self, audio_path, language="auto", batch_size=None, **kwargs):
        self.log(f"🚀 [Qwen3] 开始处理: {os.path.basename(audio_path)}")
        t_start = time.time()
        
        # 1. 尝试使用 Shared Volume 优化
        file_name = os.path.basename(audio_path)
        shared_path = os.path.join(SHARED_HOST_DIR, file_name)
        docker_path = f"{SHARED_DOCKER_DIR}/{file_name}"
        
        use_shared = False
        try:
            import shutil
            if not os.path.exists(SHARED_HOST_DIR):
                os.makedirs(SHARED_HOST_DIR, exist_ok=True)
            shutil.copy(audio_path, shared_path)
            use_shared = True
            self.log("⚡ [IO加速] 已启用共享存储卷模式")
        except Exception as e:
            self.log(f"⚠️ 无法复制文件到共享目录: {e}, 将使用 Base64 慢速模式")

        # 2. 构造请求
        payload = {
            "language": language,
            "return_timestamps": True
        }
        
        if use_shared:
            payload["audio_paths"] = [docker_path] # 新版 API 支持直接传路径
        else:
            # Fallback to base64
            import base64
            with open(audio_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            payload["audio_data"] = [b64]

        # 3. 发送请求
        try:
            # 使用 batch 大口径
            res = requests.post(f"{DOCKER_HOST}/v1/audio/batch_transcriptions", json=payload, timeout=1200)
            if res.status_code != 200:
                raise Exception(f"API Error {res.status_code}: {res.text}")
            
            data = res.json()
            # Batch API returns a list of results
            result_item = data[0] if isinstance(data, list) and len(data) > 0 else data
            
            text = result_item.get("text", "")
            timestamps = result_item.get("timestamps", [])
            
            dur = time.time() - t_start
            self.log(f"✅ Qwen3 处理完成! 耗时: {dur:.2f}s")
            
            # 4. 生成 SRT (Qwen 返回的是字级别的 timestamp)
            srt_content = self._build_srt(timestamps, text)
            
            return {
                "text": text,
                "srt": srt_content,
                "raw": result_item
            }
            
        except Exception as e:
            self.log(f"❌ Qwen3 请求失败: {e}")
            raise e
        finally:
            # Cleanup shared file
            if use_shared and os.path.exists(shared_path):
                try: os.remove(shared_path)
                except: pass

    def _build_srt(self, timestamps, full_text):
        if not timestamps:
            return f"1\n00:00:00,000 --> 00:00:10,000\n{full_text}\n"
        
        # 简单策略：每 30 个字切一行，或者利用时间戳间隔切分
        res = ""
        idx = 1
        # TODO: Implement smarter splitting logic here if needed
        # For now, simplistic mapping
        return f"1\n00:00:00,000 --> 00:00:10,000\n{full_text}\n"
