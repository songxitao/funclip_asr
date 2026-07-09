import os
import sys
import psutil
import pytest

def test_cpu_affinity():
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import asr_onnx_service
    
    affinity = psutil.Process().cpu_affinity()
    print("Current CPU Affinity:", affinity)
    assert set(affinity).issubset({0, 1, 2, 3, 4, 5})
    assert len(affinity) <= 6
