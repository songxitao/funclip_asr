import os
import sys
# 导入 ASR 服务模块，这会触发头部 affinity 绑定
import asr_onnx_service
import psutil

def test_cpu_affinity():
    affinity = psutil.Process().cpu_affinity()
    print("Current CPU Affinity:", affinity)
    assert set(affinity).issubset({0, 1, 2, 3, 4, 5})
