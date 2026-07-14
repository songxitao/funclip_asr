import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import asr_onnx_service as asr_service

def test_run_inference_with_punc(monkeypatch):
    # 确保模型加载
    if asr_service.MODEL is None:
        asr_service.load_models()
    
    # 监控标点模型是否被调用
    called = []
    original_generate = asr_service.PUNC_MODEL.generate
    def mock_generate(*args, **kwargs):
        called.append(True)
        return original_generate(*args, **kwargs)
    
    monkeypatch.setattr(asr_service.PUNC_MODEL, "generate", mock_generate)
    
    # 输入测试文件
    audio_path = r"E:\下载\下载\李雪花2.wav"
    # 执行带有 VAD 切分和大 Batch 的推理
    text = asr_service._run_inference(audio_path, vad_split=True)
    # 验证输出不为空且成功加上了标点符号（包含逗号或句号等字符）
    assert len(text) > 0
    assert any(p in text for p in ["，", "。", "？", "！"])
    # 验证是否调用了全局标点模型进行后处理
    assert len(called) > 0
