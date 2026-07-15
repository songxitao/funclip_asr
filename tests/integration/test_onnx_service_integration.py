import subprocess
import time
import os
import sys
import tempfile
import wave
import struct
import requests
import pytest


def _generate_sine_wav(path, duration_sec=1.0, freq=440, sr=16000):
    """动态生成 1 秒 16kHz 单声道正弦波 WAV 文件用于测试"""
    n_samples = int(sr * duration_sec)
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        for i in range(n_samples):
            val = int(16000 * 0.3 * __import__('math').sin(2 * __import__('math').pi * freq * i / sr))
            w.writeframesraw(struct.pack('<h', val))


@pytest.fixture(scope="module", autouse=True)
def run_service():
    # 启动后台 ASR-ONNX 服务
    process = subprocess.Popen(
        [sys.executable, "asr_onnx_service.py"],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
    )
    # 给予充足的模型加载初始化时间 (15秒)
    time.sleep(15)

    yield

    # 测试结束后优雅终止并彻底清理
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


@pytest.mark.skip(reason="需要启动外部 ASR 服务并依赖真实模型文件")
def test_transcribe_api():
    url = "http://127.0.0.1:8002/transcribe"

    # 生成临时测试音频
    tmp_wav = os.path.join(tempfile.gettempdir(), "onnx_test_sine.wav")
    try:
        _generate_sine_wav(tmp_wav)

        with open(tmp_wav, "rb") as f:
            files = {"file": f}
            data = {"vad_split": "true"}
            resp = requests.post(url, files=files, data=data)

        assert resp.status_code == 200
        res_json = resp.json()
        assert "text" in res_json
    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)
