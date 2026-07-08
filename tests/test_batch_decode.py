import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmall")
from asr_onnx_service import SenseVoiceSmall

def test_batch_decode_multi():
    model_path = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmall-ONNX"
    # 初始化 batch_size = 2
    model = SenseVoiceSmall(model_dir=model_path, batch_size=2, quantize=True, device_id="0")
    
    # 制造 2 个短静音片段
    wave1 = np.zeros(16000, dtype=np.float32)
    wave2 = np.zeros(16000, dtype=np.float32)
    
    res = model([wave1, wave2])
    print("ASR Batch Result:", res)
    # 验证返回的结果列表长度为 2
    assert isinstance(res, list)
    assert len(res) == 2
