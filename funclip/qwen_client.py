import requests
import base64
import os
import json

class QwenASRClient:
    def __init__(self, host="127.0.0.1", port=28000):
        self.base_url = f"http://{host}:{port}"
        self.batch_transcribe_endpoint = f"{self.base_url}/v1/audio/batch_transcriptions"

    def _preprocess_and_encode(self, audio_path, preprocess=True, verbose=False):
        """Helper to process single file to base64"""
        if not os.path.exists(audio_path):
             raise FileNotFoundError(f"Audio file not found: {audio_path}")

        temp_wav = None
        audio_base64 = None
        
        try:
            if preprocess:
                import subprocess
                import uuid
                import tempfile
                temp_wav = os.path.join(tempfile.gettempdir(), f"qwen_client_{uuid.uuid4()}.wav")
                
                if verbose: print(f"🔄 [Client] Pre-processing {os.path.basename(audio_path)}...")
                cmd = [
                    "ffmpeg", "-y", 
                    "-i", audio_path,
                    "-vn", "-ac", "1", "-ar", "16000", 
                    "-f", "wav", 
                    temp_wav
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                target_path = temp_wav
            else:
                target_path = audio_path

            with open(target_path, "rb") as f:
                audio_bytes = f.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                
        finally:
            if temp_wav and os.path.exists(temp_wav):
                try: os.remove(temp_wav)
                except: pass
                
        return audio_base64

    def transcribe(self, audio_path, language=None, return_timestamps=False, preprocess=True, verbose=True):
        """Single file transcription (Legacy)"""
        try:
            b64 = self._preprocess_and_encode(audio_path, preprocess, verbose)
            
            payload = {
                "audio_base64": b64,
                "return_timestamps": return_timestamps
            }
            if language and language != "auto":
                payload["language"] = language
            
            if verbose: print(f"🚀 [Client] Sending request to {self.transcribe_endpoint}...")
            resp = requests.post(self.transcribe_endpoint, json=payload, timeout=300)
            if verbose: print(f"📥 [Client] Response: {resp.status_code}")
            
            if resp.status_code == 200:
                return resp.json()
            else:
                raise Exception(f"Server Error {resp.status_code}: {resp.text}")
        except Exception as e:
            if verbose: print(f"❌ Error: {e}")
            return None

    def transcribe_batch(self, audio_paths, language=None, return_timestamps=False, preprocess=True, verbose=True):
        """Batch transcription (Optimized with Shared Volume Support)"""
        import concurrent.futures
        import shutil
        
        # 🔥 共享目录配置
        HOST_SHARE_DIR = r"E:\FunClip\qwen_server\shared_tmp"
        DOCKER_SHARE_DIR = "/app/server/shared_tmp"
        
        use_shared_volume = os.path.exists(HOST_SHARE_DIR)
        
        if verbose: print(f"📦 [Client] Preparing batch of {len(audio_paths)} files (Mode: {'Shared Volume 🚀' if use_shared_volume else 'HTTP Base64 🐢'})...")
        
        # 1. 极速模式 (直接文件IO)
        if use_shared_volume:
            docker_paths = []
            try:
                # 简单复制文件到共享目录 (比 Base64 快100倍)
                for path in audio_paths:
                    if not os.path.exists(path): continue
                    
                    filename = os.path.basename(path)
                    target_host = os.path.join(HOST_SHARE_DIR, filename)
                    
                    # 只有当源文件不在共享目录时才复制
                    # 注意：当前 path 应该已经在 temp 里了，所以直接 move 或 copy 都行
                    if os.path.abspath(path) != os.path.abspath(target_host):
                        shutil.copy2(path, target_host)
                    
                    # 构造 Docker 内部路径
                    target_docker = f"{DOCKER_SHARE_DIR}/{filename}"
                    docker_paths.append(target_docker)
                    
            except Exception as e:
                print(f"❌ Shared Volume Copy Failed: {e}, falling back to Base64")
                use_shared_volume = False # 降级处理
            
            if use_shared_volume:
                 payload = {
                    "audio_paths": docker_paths,
                    "return_timestamps": return_timestamps
                 }
                 if language and language != "auto": payload["language"] = language
                 
                 # 发送轻量级请求
                 try:
                    if verbose: print(f"🚀 [Client] Sending PATH request to {self.batch_transcribe_endpoint}...")
                    resp = requests.post(self.batch_transcribe_endpoint, json=payload, timeout=600)
                    if verbose: print(f"📥 [Client] Batch Response: {resp.status_code}")
                    
                    if resp.status_code == 200:
                        # 成功后可以在后台清理一下共享文件? 暂时保留以便调试
                        return resp.json().get("results", [])
                    else:
                        raise Exception(f"Server Error {resp.status_code}: {resp.text}")
                 except Exception as e:
                    print(f"❌ [Client] Batch Request Failed: {e}")
                    return None

        # 2. 传统模式 (Base64) - 只有当共享目录不可用或失败时才走到这里
        # Parallel preprocessing (CPU bound)
        batch_b64 = [None] * len(audio_paths)
        
        def process_one(idx, path):
            try:
                return idx, self._preprocess_and_encode(path, preprocess, verbose=False)
            except Exception as e:
                print(f"❌ Failed to process {path}: {e}")
                return idx, None

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_one, i, p) for i, p in enumerate(audio_paths)]
            for f in concurrent.futures.as_completed(futures):
                i, b64 = f.result()
                if b64: batch_b64[i] = b64
        
        # Filter failures? Or fail all? Qwen expects list.
        # Check if any failed
        if any(b is None for b in batch_b64):
            raise Exception("Some files failed to preprocess in batch.")

        payload = {
            "audio_batch_base64": batch_b64,
            "return_timestamps": return_timestamps
        }
        if language and language != "auto":
            payload["language"] = language
            
        try:
            if verbose: print(f"🚀 [Client] Sending BATCH request to {self.batch_transcribe_endpoint}...")
            resp = requests.post(self.batch_transcribe_endpoint, json=payload, timeout=600) # Longer timeout for batch
            if verbose: print(f"📥 [Client] Batch Response: {resp.status_code}")
            
            if resp.status_code == 200:
                return resp.json().get("results", [])
            else:
                raise Exception(f"Server Error {resp.status_code}: {resp.text}")
                
        except Exception as e:
            print(f"❌ [Client] Batch Request Failed: {e}")
            return None

# Simple test if run directly
if __name__ == "__main__":
    client = QwenASRClient()
    # Mock test
    pass
