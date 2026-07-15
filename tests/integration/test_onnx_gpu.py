import os
import sys
import json
import time
import pytest

# --- 项目根目录（用于模型路径） ---
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 限制线程数环境变量（不依赖硬编码路径）
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

# 动态添加 SenseVoiceSmall 模型目录到 PYTHONPATH
sensevoice_dir = os.path.join(BASE, "model", "models", "iic", "SenseVoiceSmall")
if os.path.isdir(sensevoice_dir):
    sys.path.append(sensevoice_dir)

import torch
torch.set_num_threads(4)

from funasr import AutoModel
from utils.model_bin import SenseVoiceSmallONNX

pytestmark = pytest.mark.slow


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
    """FSMN-VAD (PyTorch版在CPU运行) + SenseVoiceSmall-ONNX 推理"""
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


def _generate_test_audio(path, duration_sec=1.0):
    """生成 1 秒正弦波测试音频"""
    import struct
    import wave
    import math
    n_samples = int(16000 * duration_sec)
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        for i in range(n_samples):
            val = int(16000 * 0.3 * math.sin(2 * math.pi * 440 * i / 16000))
            w.writeframesraw(struct.pack('<h', val))


def _model_paths():
    """返回项目相对路径描述的模型路径"""
    return {
        "model_dir": os.path.join(BASE, "model", "models", "iic", "SenseVoiceSmall-ONNX"),
        "vad_path": os.path.join(BASE, "model", "models", "damo", "speech_fsmn_vad_zh-cn-16k-common-pytorch"),
    }


@pytest.mark.skip(reason="需要 GPU + 真实 ONNX 模型文件才能运行")
def test_onnx_gpu_pipeline():
    """用于 pytest 运行的测试用例"""
    import tempfile
    paths = _model_paths()
    tmp_wav = os.path.join(tempfile.gettempdir(), "onnx_gpu_test.wav")
    _generate_test_audio(tmp_wav)
    try:
        vad_model = AutoModel(model=paths["vad_path"], trust_remote_code=True, device="cpu", disable_update=True, disable_pbar=True)
        model_gpu = SenseVoiceSmall(paths["model_dir"], batch_size=1, quantize=True, device_id="0", intra_op_num_threads=4)

        ort_session = model_gpu.ort_infer.session
        actual_providers = ort_session.get_providers()
        assert "CUDAExecutionProvider" in actual_providers

        text = run_pipeline(model_gpu, vad_model, tmp_wav)
        assert len(text) > 0
    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)


@pytest.mark.skip(reason="需要启动外部 ASR ONNX 服务")
def test_onnx_api_endpoint():
    """自动化验证 ASR ONNX API 服务的 HTTP 接口"""
    import httpx
    import tempfile
    url = "http://127.0.0.1:8002/transcribe"

    tmp_wav = os.path.join(tempfile.gettempdir(), "onnx_api_test.wav")
    _generate_test_audio(tmp_wav)
    try:
        with open(tmp_wav, "rb") as f:
            files = {"file": ("test.wav", f, "audio/wav")}
            data = {"vad_split": "true"}

            t0 = time.time()
            response = httpx.post(url, files=files, data=data, timeout=60.0)
            duration = time.time() - t0

        assert response.status_code == 200, f"接口请求失败: {response.status_code}, {response.text}"
        result = response.json()
        assert "text" in result
        assert "latency_ms" in result
    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)


def main():
    paths = _model_paths()
    import tempfile
    tmp_wav = os.path.join(tempfile.gettempdir(), "onnx_gpu_main.wav")
    _generate_test_audio(tmp_wav)
    try:
        audio_path = tmp_wav
        vad_path = paths["vad_path"]
        model_dir = paths["model_dir"]

        print("="*50)
        print("开始 FSMN-VAD (CPU) + SenseVoiceSmall-ONNX (GPU 0) 性能评测与测试")
        print("="*50)

        # 1. 加载 VAD
        t_start = time.time()
        vad_model = AutoModel(model=vad_path, trust_remote_code=True, device="cpu", disable_update=True, disable_pbar=True)
        vad_load_time = time.time() - t_start
        print(f"FSMN-VAD 模型加载时间: {vad_load_time:.4f} 秒")

        # 2. 实例化 SenseVoiceSmall
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
    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)


if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
