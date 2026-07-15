import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_punc_model_loaded():
    import asr_onnx_service as asr_service
    # 调用加载模型逻辑
    asr_service.load_models()
    assert asr_service.PUNC_MODEL is not None
    # 确保加载在 CPU 侧
    assert asr_service.PUNC_MODEL.device == "cpu"
