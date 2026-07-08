import os
# 1. 在最顶层设置环境变量，严格限制 CPU 线程池大小
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import sys
import json
import time
import pytest

# 2. 导入 torch 之后，立刻限制 PyTorch CPU 线程
import torch
torch.set_num_threads(4)

# 导入 ONNX 运行时，检测 GPU 是否可用
import onnxruntime as ort

# 添加 SenseVoiceSmall 模型目录到 PYTHONPATH，使得可以导入 utils.model_bin 等
sys.path.append(r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmall")
from utils.model_bin import SenseVoiceSmallONNX

class SenseVoiceSmall(SenseVoiceSmallONNX):
    """包装类，重写了初始化和调用，适配用户要求的接口"""
    def __init__(self, model_dir, batch_size=1, quantize=True, device_id="-1", intra_op_num_threads=4, **kwargs):
        super().__init__(model_dir, batch_size=batch_size, device_id=device_id, quantize=quantize, intra_op_num_threads=intra_op_num_threads, **kwargs)
        # 加载 tokens.json 以还原文本
        tokens_path = os.path.join(model_dir, "tokens.json")
        with open(tokens_path, "r", encoding="utf-8") as f:
            self.tokens = json.load(f)

    def __call__(self, wav_content, language=[0], textnorm=[15], tokenizer=None, **kwargs):
        if tokenizer is None:
            # 内部默认 Tokenizer
            class DefaultTokenizer:
                def __init__(self, tokens):
                    self.tokens = tokens
                def tokens2text(self, ids):
                    res = []
                    for i in ids:
                        t = self.tokens[i]
                        # 过滤掉特殊 tag 像 <|zh|> <|happy|> 等
                        if t.startswith("<|") and t.endswith("|>"):
                            continue
                        if t == "<space>":
                            res.append(" ")
                        elif t == "<unk>":
                            continue
                        else:
                            res.append(t)
                    return "".join(res)
            tokenizer = DefaultTokenizer(self.tokens)
        return super().__call__(wav_content, language=language, textnorm=textnorm, tokenizer=tokenizer, **kwargs)

def test_onnx_performance_comparison():
    audio_path = r"E:\下载\下载\李雪花2.wav"
    model_dir = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmall-ONNX"
    
    print("\n" + "="*50)
    print("开始 ASR ONNX 推理性能评测")
    print("="*50)
    
    # ---------------- CPU 推理评估 ----------------
    print("\n[CPU 环境评测]")
    t_start = time.time()
    # 显式限制线程数为 4
    model_cpu = SenseVoiceSmall(model_dir, batch_size=1, quantize=True, device_id="-1", intra_op_num_threads=4)
    cpu_load_time = time.time() - t_start
    print(f"CPU 模型加载时间: {cpu_load_time:.4f} 秒")
    
    # CPU 冷启动转写
    t_start = time.time()
    res_cpu_cold = model_cpu([audio_path])
    cpu_cold_time = time.time() - t_start
    text_cpu_cold = res_cpu_cold[0] if res_cpu_cold else ""
    print(f"CPU 首次推理(冷启动)时间: {cpu_cold_time:.4f} 秒")
    print(f"CPU 首次转写文本: {text_cpu_cold}")
    
    # CPU 热启动转写
    t_start = time.time()
    res_cpu_hot = model_cpu([audio_path])
    cpu_hot_time = time.time() - t_start
    text_cpu_hot = res_cpu_hot[0] if res_cpu_hot else ""
    print(f"CPU 二次推理(热启动)时间: {cpu_hot_time:.4f} 秒")
    print(f"CPU 二次转写文本: {text_cpu_hot}")
    
    # ---------------- GPU 推理评估 ----------------
    print("\n[GPU 环境评测]")
    
    # 初始化一个临时变量用来检查 GPU 的可用性
    gpu_available = False
    try:
        t_start = time.time()
        # 尝试加载 GPU 模型
        model_gpu = SenseVoiceSmall(model_dir, batch_size=1, quantize=True, device_id="0", intra_op_num_threads=4)
        gpu_load_time = time.time() - t_start
        
        # 核心检查：虽然声明了 GPU，但 runtime 可能会静默退回至 CPU。我们必须检查它的 actual provider
        actual_providers = model_gpu.ort_infer.session.get_providers()
        print(f"ONNX Session 实际支持的 Providers: {actual_providers}")
        
        if "CUDAExecutionProvider" in actual_providers:
            gpu_available = True
            print(f"GPU 模型加载时间: {gpu_load_time:.4f} 秒")
            
            # GPU 冷启动转写
            t_start = time.time()
            res_gpu_cold = model_gpu([audio_path])
            gpu_cold_time = time.time() - t_start
            text_gpu_cold = res_gpu_cold[0] if res_gpu_cold else ""
            print(f"GPU 首次推理(冷启动)时间: {gpu_cold_time:.4f} 秒")
            print(f"GPU 首次转写文本: {text_gpu_cold}")
            
            # GPU 热启动转写
            t_start = time.time()
            res_gpu_hot = model_gpu([audio_path])
            gpu_hot_time = time.time() - t_start
            text_gpu_hot = res_gpu_hot[0] if res_gpu_hot else ""
            print(f"GPU 二次推理(热启动)时间: {gpu_hot_time:.4f} 秒")
            print(f"GPU 二次转写文本: {text_gpu_hot}")
        else:
            print("⚠️ 警告: CUDAExecutionProvider 未被 onnxruntime 实际加载 (可能由于 cudnn64_9.dll 丢失等驱动/库不兼容原因)，为了防范 CPU 重复满载计算，将跳过真实 GPU 推理测试。")
    except Exception as e:
        print(f"GPU 初始化或加载失败: {e}，跳过 GPU 测试。")
        
    print("\n" + "="*50)
    print("性能评测总结:")
    print(f"CPU 加载时间: {cpu_load_time:.4f}s | 冷启动时间: {cpu_cold_time:.4f}s | 热启动时间: {cpu_hot_time:.4f}s")
    if gpu_available:
        print(f"GPU 加载时间: {gpu_load_time:.4f}s | 冷启动时间: {gpu_cold_time:.4f}s | 热启动时间: {gpu_hot_time:.4f}s")
        if gpu_hot_time > 0:
            speedup = cpu_hot_time / gpu_hot_time
            print(f"GPU 对比 CPU 热启动提速: {speedup:.2f} 倍")
    else:
        print("GPU 推理未被成功点亮，仅使用 CPU 完成基准评估。")
    print("="*50)

if __name__ == "__main__":
    # 强制命令行输出 UTF-8 字符防止乱码
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    test_onnx_performance_comparison()
