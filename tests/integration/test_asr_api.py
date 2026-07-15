import os
import tempfile
import wave
import struct
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.slow

# 引入微服务应用
from asr_onnx_service import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def generate_silent_wav(path):
    """动态生成 1 秒钟的 16kHz 单声道静音 WAV 文件用于测试"""
    with wave.open(path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        for _ in range(16000):
            data = struct.pack('<h', 0)
            wav_file.writeframesraw(data)

def test_transcribe_endpoint_with_mock(client):
    """单元测试：使用 Mock 模拟 ASR & VAD 模型，彻底解耦物理 GPU 环境"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_path = temp_audio.name
    try:
        generate_silent_wav(temp_path)
        
        # 1. 模拟 ASR MODEL 与 VAD MODEL
        with patch("asr_service.MODEL") as mock_model, patch("asr_service.VAD_MODEL") as mock_vad:
            # 模拟 ASR 极速模式的返回值
            mock_model.generate.return_value = [{"text": "<|happy|>测试静音文本。<|applause|>"}]
            # 模拟 VAD 分割返回值
            mock_vad.generate.return_value = [{"value": [[0, 500]]}]
            
            with open(temp_path, 'rb') as f:
                files = {'file': (os.path.basename(temp_path), f, 'audio/wav')}
                # 测试极速模式 (vad_split=False)
                response = client.post("/transcribe", files=files, data={"vad_split": "false"})
                
            assert response.status_code == 200
            data = response.json()
            assert "text" in data
            assert data["text"] == "测试静音文本。"
            assert "latency_ms" in data
            assert data["latency_ms"] > 0
            
            # 测试 VAD 切片模式 (vad_split=True)
            with open(temp_path, 'rb') as f:
                files = {'file': (os.path.basename(temp_path), f, 'audio/wav')}
                response_vad = client.post("/transcribe", files=files, data={"vad_split": "true"})
                
            assert response_vad.status_code == 200
            data_vad = response_vad.json()
            assert "text" in data_vad
            assert "latency_ms" in data_vad
            
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# 检测当前机器是否具备运行真实模型测试的条件（如具备 GPU 并有本地模型目录）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "models", "iic", "SenseVoiceSmall")
HAS_PHYSICAL_ENV = os.path.exists(MODEL_PATH)

@pytest.mark.skipif(not HAS_PHYSICAL_ENV, reason="没有检测到本地 SenseVoice 模型，跳过真实集成测试")
def test_transcribe_endpoint_real():
    """集成测试：如果在具备模型和 GPU 物理环境的机器上，执行真实模型转写"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_path = temp_audio.name
    try:
        generate_silent_wav(temp_path)
        
        # 验证极速单句模式 (vad_split=False)
        with TestClient(app) as real_client:
            with open(temp_path, 'rb') as f:
                files = {'file': (os.path.basename(temp_path), f, 'audio/wav')}
                response = real_client.post("/transcribe", files=files, data={"vad_split": "false"})
                
            assert response.status_code == 200
            data = response.json()
            assert "text" in data
            assert "latency_ms" in data
            assert data["latency_ms"] > 0
            
            # 验证 VAD 长音频切句模式 (vad_split=True)
            with open(temp_path, 'rb') as f:
                files = {'file': (os.path.basename(temp_path), f, 'audio/wav')}
                response_vad = real_client.post("/transcribe", files=files, data={"vad_split": "true"})
                
            assert response_vad.status_code == 200
            data_vad = response_vad.json()
            assert "text" in data_vad
            assert data_vad["latency_ms"] > 0
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
