import os
import sys
import time
import numpy as np
import onnxruntime as ort

# 项目根目录
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 添加 SenseVoiceSmall 目录到 PYTHONPATH
sensevoice_dir = os.path.join(BASE, "model", "models", "iic", "SenseVoiceSmall")
if os.path.isdir(sensevoice_dir):
    sys.path.append(sensevoice_dir)

from utils.model_bin import SenseVoiceSmallONNX

def test_openvino_vs_ort():
    model_dir = os.path.join(BASE, "model", "models", "iic", "SenseVoiceSmall-ONNX")
    model_file = os.path.join(model_dir, "model_quant.onnx")
    
    # 1. 模拟 ASR 的标准输入形状
    batch_size = 4
    time_len = 100
    feat_dim = 560
    
    feats = np.random.randn(batch_size, time_len, feat_dim).astype(np.float32)
    feats_len = np.array([time_len] * batch_size, dtype=np.int32)
    language = np.array([0] * batch_size, dtype=np.int32)
    textnorm = np.array([15] * batch_size, dtype=np.int32)
    
    print("=" * 60)
    print("【开始 CPU 推理引擎跑评对照 (ORT vs OpenVINO)】")
    print("=" * 60)
    
    # ---------------- 引擎一: ONNX Runtime (默认 CPU - 优化参数) ----------------
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 6
    opts.inter_op_num_threads = 1
    
    ort_sess = ort.InferenceSession(model_file, sess_options=opts, providers=['CPUExecutionProvider'])
    
    # 预热
    for _ in range(5):
        _ = ort_sess.run(None, {
            "speech": feats,
            "speech_lengths": feats_len,
            "language": language,
            "textnorm": textnorm
        })
        
    t_start = time.time()
    for _ in range(30):
        _ = ort_sess.run(None, {
            "speech": feats,
            "speech_lengths": feats_len,
            "language": language,
            "textnorm": textnorm
        })
    ort_time = (time.time() - t_start) / 30 * 1000
    print(f"ONNX Runtime 默认 CPU (6线程) 平均推理耗时: {ort_time:.2f} ms")
    
    # ---------------- 引擎二: OpenVINO Core API ----------------
    try:
        from openvino import Core
        ie = Core()
        # 读取 ONNX 模型并在 CPU 上即时编译
        ov_model = ie.read_model(model_file)
        compiled_model = ie.compile_model(ov_model, "CPU", config={
            "INFERENCE_NUM_THREADS": "6",
            "NUM_STREAMS": "1"
        })
        
        # 预热
        for _ in range(5):
            _ = compiled_model([feats, feats_len, language, textnorm])
            
        t_start = time.time()
        for _ in range(30):
            _ = compiled_model([feats, feats_len, language, textnorm])
        ov_time = (time.time() - t_start) / 30 * 1000
        print(f"OpenVINO CPU (6线程) 平均推理耗时: {ov_time:.2f} ms")
        
        speedup = ort_time / ov_time
        print("-" * 60)
        print(f"OpenVINO 对比 ONNX Runtime 纯推理提速: {speedup:.2f} 倍")
        
    except Exception as e:
        print(f"OpenVINO 测试加载/运行失败: {e}")
        print("请检查 openvino 库安装或 DLL 环境。")
        
    print("=" * 60)

if __name__ == "__main__":
    test_openvino_vs_ort()
