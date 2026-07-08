import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psutil
import torch
import asr_service  # 触发头部亲和性和线程设置

def test_cpu_affinity_and_threads():
    affinity = psutil.Process().cpu_affinity()
    assert len(affinity) == 6 or set(affinity).issubset({0, 1, 2, 3, 4, 5})
    assert torch.get_num_threads() == 6
