import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import asr_onnx_service as asr_service

pytestmark = pytest.mark.slow

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
    
    # 生成临时测试音频（正弦波代替真实音频文件）
    import tempfile, wave, struct, math
    audio_path = os.path.join(tempfile.gettempdir(), "pytorch_infer_test.wav")
    n_samples = int(16000 * 2)
    with wave.open(audio_path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        for i in range(n_samples):
            val = int(16000 * 0.3 * math.sin(2 * math.pi * 440 * i / 16000))
            w.writeframesraw(struct.pack('<h', val))
    try:
        # 执行带有 VAD 切分和大 Batch 的推理
        text = asr_service._run_inference(audio_path, vad_split=True)
        # 验证输出不为空且成功加上了标点符号（包含逗号或句号等字符）
        assert len(text) > 0
        assert any(p in text for p in ["，", "。", "？", "！"])
        # 验证是否调用了全局标点模型进行后处理
        assert len(called) > 0
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)
