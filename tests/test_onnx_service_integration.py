import subprocess
import time
import requests
import pytest

@pytest.fixture(scope="module", autouse=True)
def run_service():
    # 启动后台 ASR-ONNX 服务，因为是在 Windows 上运行
    # 请使用 E:\conda\envs\asr_ui_env\python.exe 启动
    process = subprocess.Popen(
        [r"E:\conda\envs\asr_ui_env\python.exe", "asr_onnx_service.py"],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP  # 方便完整清理进程组
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

def test_transcribe_api():
    url = "http://127.0.0.1:8002/transcribe"
    audio_file = r"E:\下载\下载\李雪花2.wav"
    
    with open(audio_file, "rb") as f:
        files = {"file": f}
        data = {"vad_split": "true"}
        resp = requests.post(url, files=files, data=data)
        
    assert resp.status_code == 200
    res_json = resp.json()
    assert "text" in res_json
    assert len(res_json["text"]) > 0
    
    text = res_json["text"]
    print("ASR Result Sample:", text[:100])
    
    # 验证结果中是否成功恢复了标点符号（，。！？）
    has_punc = any(p in text for p in ["，", "。", "？", "！"])
    assert has_punc, "转写文本未带上任何标点符号"
