import os
import sys
import json
import time
import numpy as np
import pytest
import torch
torch.set_num_threads(6)

# 限制线程数环境变量
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

# 项目根目录
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 添加项目路径和 SenseVoiceSmall 模型目录到 PYTHONPATH
sys.path.append(BASE)
sensevoice_dir = os.path.join(BASE, "model", "models", "iic", "SenseVoiceSmall")
if os.path.isdir(sensevoice_dir):
    sys.path.append(sensevoice_dir)

# 直接从 funclip_pro.core 导入我们已经重构好的 SenseVoiceSmall 类
from funclip_pro.core import SenseVoiceSmall
from funasr import AutoModel


def run_pipeline(model, vad_model, audio_path, limit_segments=10):
    """VAD + ASR ONNX 双模 Pipeline 推理函数"""
    import librosa
    audio, _ = librosa.load(audio_path, sr=16000)

    vad_model.model.to("cpu")
    vad_model.kwargs["device"] = "cpu"

    vad_out = vad_model.generate(input=audio_path, batch_size_s=5000, max_single_segment_time=60000)
    raw_segs = vad_out[0]['value'] if vad_out and len(vad_out) > 0 and 'value' in vad_out[0] else [[0, len(audio)/16*1000]]

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

    # 限制切片数量以大幅加速测试
    if limit_segments:
        print(f"限制仅测试前 {limit_segments} 个音频切片")
        opt_segs = opt_segs[:limit_segments]

    # 收集音频切片进行批量 ASR 推理
    chunks = []
    for start_ms, end_ms in opt_segs:
        s_idx = int(start_ms * 16)
        e_idx = int(end_ms * 16)
        chunk = audio[max(0, s_idx-800):min(len(audio), e_idx+800)]
        if len(chunk) < 1600: continue
        chunks.append(chunk)

    # 直接使用 asr_onnx_service 的批量推理机制进行调用
    texts = model(chunks)
    return "\n".join(texts)


def test_onnx_performance_comparison():
    model_dir = os.path.join(BASE, "model", "models", "iic", "SenseVoiceSmall-ONNX")
    vad_path = os.path.join(BASE, "model", "models", "damo", "speech_fsmn_vad_zh-cn-16k-common-pytorch")

    # 如果模型目录不存在则跳过
    if not os.path.isdir(model_dir):
        pytest.skip(f"模型目录不存在: {model_dir}")

    # 生成临时测试音频
    import tempfile, wave, struct, math
    tmp_wav = os.path.join(tempfile.gettempdir(), "perf_test.wav")
    n_samples = int(16000 * 5)
    with wave.open(tmp_wav, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        for i in range(n_samples):
            val = int(16000 * 0.3 * math.sin(2 * math.pi * 440 * i / 16000))
            w.writeframesraw(struct.pack('<h', val))

    print("\n" + "="*50)
    print("开始 ASR ONNX + FSMN VAD 性能评测与集成测试 (加速版)")
    print("="*50)

    # ---------------- CPU 环境评测 ----------------
    print("\n[CPU 环境评测]")
    t_start = time.time()
    # 使用 6 核心物理线程配置
    model_cpu = SenseVoiceSmall(model_dir, batch_size=16, quantize=True, device_id="-1", intra_op_num_threads=6)
    cpu_load_time = time.time() - t_start
    print(f"CPU ASR 模型加载时间: {cpu_load_time:.4f} 秒")

    t_start = time.time()
    vad_cpu = AutoModel(model=vad_path, trust_remote_code=True, device="cpu", disable_update=True, disable_pbar=True)
    cpu_vad_load_time = time.time() - t_start
    print(f"CPU VAD 模型加载时间: {cpu_vad_load_time:.4f} 秒")

    # CPU 极速单句模式 (冷/热启动)，使用 5 秒 dummy 音频以极速完成
    print("--- 轨道一: 极速单句模式 (5秒 Dummy 音频) ---")
    dummy_wav = np.zeros(16000 * 5, dtype=np.float32)

    t_start = time.time()
    res_cpu_cold = model_cpu([dummy_wav])
    cpu_cold_time = time.time() - t_start
    print(f"CPU 单句冷启动时间: {cpu_cold_time:.4f} 秒")

    t_start = time.time()
    res_cpu_hot = model_cpu([dummy_wav])
    cpu_hot_time = time.time() - t_start
    print(f"CPU 单句热启动时间: {cpu_hot_time:.4f} 秒")

    # CPU VAD+ASR 双模 Pipeline 模式 (冷/热启动)
    print("--- 轨道二: 双模 Pipeline 模式 (VAD+ASR) ---")
    try:
        t_start = time.time()
        text_cpu_pipe_cold = run_pipeline(model_cpu, vad_cpu, tmp_wav, limit_segments=10)
        cpu_pipe_cold_time = time.time() - t_start
        print(f"CPU Pipeline 冷启动时间: {cpu_pipe_cold_time:.4f} 秒")

        t_start = time.time()
        text_cpu_pipe_hot = run_pipeline(model_cpu, vad_cpu, tmp_wav, limit_segments=10)
        cpu_pipe_hot_time = time.time() - t_start
        print(f"CPU Pipeline 热启动时间: {cpu_pipe_hot_time:.4f} 秒")
        print(f"CPU Pipeline 转写出的文本片段 (前100字):\n{text_cpu_pipe_hot[:100]}...")
    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)

    print("\n" + "="*50)
    print("性能评测总结:")
    print(f"CPU 单句 -> 冷启动: {cpu_cold_time:.4f}s | 热启动: {cpu_hot_time:.4f}s")
    print(f"CPU Pipeline -> 冷启动: {cpu_pipe_cold_time:.4f}s | 热启动: {cpu_pipe_hot_time:.4f}s")
    print("="*50)

if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    test_onnx_performance_comparison()
