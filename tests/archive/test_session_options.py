import os
import sys
import pytest

# 项目根目录
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 添加 SenseVoiceSmall 目录到 PYTHONPATH
sensevoice_dir = os.path.join(BASE, "model", "models", "iic", "SenseVoiceSmall")
if os.path.isdir(sensevoice_dir):
    sys.path.append(sensevoice_dir)

from utils.model_bin import SenseVoiceSmallONNX
import onnxruntime as ort

@pytest.mark.skip(reason="需要真实 ONNX 模型文件")
def test_session_options():
    model_dir = os.path.join(BASE, "model", "models", "iic", "SenseVoiceSmall-ONNX")
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
