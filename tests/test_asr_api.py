import os
import tempfile
import wave
import struct
import requests
import pytest

def generate_silent_wav(path):
    """动态生成 1 秒钟的 16kHz 单声道静音 WAV 文件用于测试"""
    with wave.open(path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        for _ in range(16000):
            data = struct.pack('<h', 0)
            wav_file.writeframesraw(data)

def test_transcribe_endpoint():
    # 1. 动态生成临时测试音频
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_path = temp_audio.name
    try:
        generate_silent_wav(temp_path)
        
        # 2. 发起 API 请求 (预期微服务监听 8001 端口)
        url = "http://127.0.0.1:8001/transcribe"
        with open(temp_path, 'rb') as f:
            files = {'file': (os.path.basename(temp_path), f, 'audio/wav')}
            response = requests.post(url, files=files, data={'language': 'auto'})
            
        # 3. 校验返回值 structure 与状态
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "latency_ms" in data
        assert data["latency_ms"] > 0
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
