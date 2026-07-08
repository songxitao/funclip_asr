import os
# 1. 在脚本最顶部（在导入任何其他AI库前）添加环境变量设置
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

# 2. 动态添加 DLL 搜索目录以点亮 onnxruntime GPU 推理
ctranslate2_dll_path = r"E:\conda\envs\asr_ui_env\Lib\site-packages\ctranslate2"
if os.path.exists(ctranslate2_dll_path):
    try:
        os.add_dll_directory(ctranslate2_dll_path)
    except Exception:
        os.environ["PATH"] += os.pathsep + ctranslate2_dll_path

# 强力点亮 onnxruntime GPU 推理：将 nvidia\cudnn\bin、onnxruntime\capi 和 torch\lib 优先加入 PATH 与 DLL 目录
cudnn_bin = r"E:\conda\envs\asr_ui_env\Lib\site-packages\nvidia\cudnn\bin"
capi_path = r"E:\conda\envs\asr_ui_env\Lib\site-packages\onnxruntime\capi"
torch_lib = r"E:\conda\envs\asr_ui_env\Lib\site-packages\torch\lib"

extra_paths = [cudnn_bin, capi_path, torch_lib]
for path in extra_paths:
    if os.path.exists(path):
        os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
        try:
            os.add_dll_directory(path)
        except Exception:
            pass

import sys
import json
import time

# 3. 限制 PyTorch VAD 推理线程数
import torch
torch.set_num_threads(4)

from funasr import AutoModel

# 添加 SenseVoiceSmall 目录到 PYTHONPATH
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

def run_pipeline(model, vad_model, audio_path):
    """FSMN-VAD (PyTorch版在CPU运行) + SenseVoiceSmall-ONNX (在GPU 0运行) 推理"""
    import librosa
    audio, _ = librosa.load(audio_path, sr=16000)
    
    # 确保 VAD 跑在 CPU
    vad_model.model.to("cpu")
    vad_model.kwargs["device"] = "cpu"
    
    vad_out = vad_model.generate(input=audio_path, batch_size_s=5000, max_single_segment_time=60000)
    raw_segs = vad_out[0]['value'] if vad_out and len(vad_out) > 0 and 'value' in vad_out[0] else [[0, len(audio)/16*1000]]
    
    # 合并段
    def _merge_vad_segments(segments, max_gap_ms=300, max_duration_ms=8000):
        if not segments: return []
        merged = []
        curr_start, curr_end = segments[0]
        for next_start, next_end in segments[1:]:
            gap = next_start - curr_end
            duration = (curr_end - curr_start) + (next_end - next_start)
            if gap < max_gap_ms and duration < max_duration_ms:
                curr_end = next_end 
            else:
                merged.append([curr_start, curr_end]) 
                curr_start, curr_end = next_start, next_end
        merged.append([curr_start, curr_end])
        return merged
        
    opt_segs = _merge_vad_segments(raw_segs)
    
    # ASR 推理
    texts = []
    for start_ms, end_ms in opt_segs:
        s_idx = int(start_ms * 16)
        e_idx = int(end_ms * 16)
        chunk = audio[max(0, s_idx-800):min(len(audio), e_idx+800)]
        if len(chunk) < 1600: continue
        
        res = model(chunk)
        if res and res[0]:
            texts.append(res[0])
            
    return "\n".join(texts)

def test_onnx_gpu_pipeline():
    """用于 pytest 运行的测试用例"""
    audio_path = r"E:\下载\下载\李雪花2.wav"
    vad_path = r"E:\project\funclip-pro\model\models\damo\speech_fsmn_vad_zh-cn-16k-common-pytorch"
    model_dir = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmall-ONNX"
    
    # 加载 VAD 模型
    vad_model = AutoModel(model=vad_path, trust_remote_code=True, device="cpu", disable_update=True, disable_pbar=True)
    # 实例化 SenseVoiceSmall 传入 device_id="0", quantize=True
    model_gpu = SenseVoiceSmall(model_dir, batch_size=1, quantize=True, device_id="0", intra_op_num_threads=4)
    
    ort_session = model_gpu.ort_infer.session
    actual_providers = ort_session.get_providers()
    
    # 强校验真正跑在 GPU 上
    assert "CUDAExecutionProvider" in actual_providers
    
    text = run_pipeline(model_gpu, vad_model, audio_path)
    assert len(text) > 0

def test_onnx_api_endpoint():
    """自动化验证 ASR ONNX API 服务的 HTTP 接口"""
    import httpx
    import time
    audio_path = r"E:\下载\下载\李雪花2.wav"
    url = "http://127.0.0.1:8002/transcribe"
    
    assert os.path.exists(audio_path), f"音频文件不存在: {audio_path}"
    
    with open(audio_path, "rb") as f:
        files = {"file": ("李雪花2.wav", f, "audio/wav")}
        data = {"vad_split": "true"}
        
        t0 = time.time()
        response = httpx.post(url, files=files, data=data, timeout=60.0)
        duration = time.time() - t0
        
    assert response.status_code == 200, f"接口请求失败: {response.status_code}, {response.text}"
    result = response.json()
    assert "text" in result
    assert "latency_ms" in result
    
    print(f"\n[API 接口测试] API 响应耗时: {duration:.4f} 秒")
    print(f"[API 接口测试] 服务端报告推理耗时: {result['latency_ms']:.2f} ms")
    print(f"[API 接口测试] 转写结果文本:\n{result['text']}\n")

def main():
    audio_path = r"E:\下载\下载\李雪花2.wav"
    vad_path = r"E:\project\funclip-pro\model\models\damo\speech_fsmn_vad_zh-cn-16k-common-pytorch"
    model_dir = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmall-ONNX"
    
    print("="*50)
    print("开始 FSMN-VAD (CPU) + SenseVoiceSmall-ONNX (GPU 0) 性能评测与测试")
    print("="*50)
    
    # 1. 加载 VAD
    t_start = time.time()
    vad_model = AutoModel(model=vad_path, trust_remote_code=True, device="cpu", disable_update=True, disable_pbar=True)
    vad_load_time = time.time() - t_start
    print(f"FSMN-VAD 模型加载时间: {vad_load_time:.4f} 秒")
    
    # 2. 实例化 SenseVoiceSmall 传入 device_id="0", quantize=True
    t_start = time.time()
    model_gpu = SenseVoiceSmall(model_dir, batch_size=1, quantize=True, device_id="0", intra_op_num_threads=4)
    model_load_time = time.time() - t_start
    print(f"SenseVoiceSmall-ONNX 模型加载时间: {model_load_time:.4f} 秒")
    
    # 3. 打印 ort session 实际加载的 providers
    ort_session = model_gpu.ort_infer.session
    actual_providers = ort_session.get_providers()
    print(f"ONNX Session 实际加载的 Providers: {actual_providers}")
    
    # 4. 强校验真正跑在 GPU 上
    assert "CUDAExecutionProvider" in actual_providers, "错误：CUDAExecutionProvider 没有被加载，未能运行在 GPU 上！"
    print("成功验证：SenseVoiceSmall-ONNX 正在 GPU 上运行！")
    
    # 5. 测量冷启动耗时
    print("\n--- 运行冷启动推理 ---")
    t0 = time.time()
    text_cold = run_pipeline(model_gpu, vad_model, audio_path)
    cold_time = time.time() - t0
    print(f"GPU Pipeline 冷启动耗时: {cold_time:.4f} 秒")
    print(f"冷启动转写结果片段: {text_cold[:100]}")
    
    # 6. 测量热启动耗时
    print("\n--- 运行热启动推理 ---")
    t1 = time.time()
    text_hot = run_pipeline(model_gpu, vad_model, audio_path)
    hot_time = time.time() - t1
    print(f"GPU Pipeline 热启动耗时: {hot_time:.4f} 秒")
    print(f"热启动转写结果：\n{text_hot}")
    
    print("\n" + "="*50)
    print("测试成功完成")
    print(f"冷启动时间: {cold_time:.4f}s")
    print(f"热启动时间: {hot_time:.4f}s")
    print("="*50)

if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
