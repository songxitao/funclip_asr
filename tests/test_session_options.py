import os
import sys
import pytest

# 添加 SenseVoiceSmall 目录到 PYTHONPATH
sys.path.append(r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmall")
from utils.model_bin import SenseVoiceSmallONNX
import onnxruntime as ort

def test_session_options():
    model_dir = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmall-ONNX"
    model = SenseVoiceSmallONNX(model_dir, batch_size=1, quantize=True, device_id="-1", intra_op_num_threads=4)
    session = model.ort_infer.session
    options = session.get_session_options()
    
    # 验证我们注入的高级 SessionOptions
    assert options.intra_op_num_threads == 6
    assert options.inter_op_num_threads == 1
    assert options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    print("Session Options 注入成功！")

if __name__ == "__main__":
    test_session_options()
