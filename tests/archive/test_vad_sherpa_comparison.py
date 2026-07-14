import os
import io
import re
import sys
import time
import gc
import librosa
import numpy as np

# 限制 CPU 线程数限制，必须在最上方，在引入 torch 和其他多线程库之前
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

import torch
torch.set_num_threads(6)

from funasr import AutoModel
import sherpa_onnx

# 强制使用 UTF-8 编码输出以防 Windows 控制台/重定向乱码
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 字符对齐辅助工具
def get_display_width(s):
    width = 0
    for c in str(s):
        if '\u4e00' <= c <= '\u9fa5':
            width += 2
        else:
            width += 1
    return width

def pad_right(s, width):
    cur_w = get_display_width(s)
    if cur_w >= width:
        return str(s)
    return str(s) + ' ' * (width - cur_w)

def print_row(col1, col2, col3):
    w1, w2, w3 = 30, 22, 22
    r1 = pad_right(col1, w1)
    r2 = pad_right(col2, w2)
    r3 = pad_right(col3, w3)
    print(f"{r1} | {r2} | {r3}")

# 文本清理：只保留核心文字（中文字符、英文字母、数字），过滤掉所有标点、空白和标签
def clean_text(text):
    # 过滤情绪/事件富文本标签，如 <|HEARTBEAT|>, <|laughter|> 等
    text = re.sub(r"<\|.*?\|>", "", text)
    # 过滤非中英文数字字符
    return re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", text)

# Levenshtein 动态规划编辑距离算法
def get_edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,    # 删除
                    dp[i][j-1] + 1,    # 插入
                    dp[i-1][j-1] + 1   # 替换
                )
    return dp[m][n]

def calculate_cer(text_p, text_o):
    clean_p = clean_text(text_p)
    clean_o = clean_text(text_o)
    max_len = max(len(clean_p), len(clean_o))
    if max_len == 0:
        return 0, clean_p, clean_o, 0.0
    dist = get_edit_distance(clean_p, clean_o)
    cer = dist / max_len
    return dist, clean_p, clean_o, cer

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

def test_vad_sherpa_comparison():
    # 路径配置
    asr_pytorch_dir = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmall"
    vad_pytorch_dir = r"E:\project\funclip-pro\model\models\damo\speech_fsmn_vad_zh-cn-16k-common-pytorch"
    sherpa_model_path = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmallOnnx\model.int8.onnx"
    sherpa_tokens_path = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmallOnnx\tokens.txt"
    audio_path = r"E:\下载\下载\李雪花2.wav"

    assert os.path.exists(audio_path), f"音频文件不存在: {audio_path}"
    assert os.path.exists(asr_pytorch_dir), f"PyTorch ASR 模型不存在: {asr_pytorch_dir}"
    assert os.path.exists(vad_pytorch_dir), f"VAD 模型不存在: {vad_pytorch_dir}"
    assert os.path.exists(sherpa_model_path), f"Sherpa ONNX 模型不存在: {sherpa_model_path}"
    assert os.path.exists(sherpa_tokens_path), f"Sherpa 词表不存在: {sherpa_tokens_path}"

    print(f"\n[1/5] 载入测试音频: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000)
    audio_duration = len(audio) / sr
    print(f"音频加载成功，时长: {audio_duration:.2f} 秒，采样率: {sr}")

    print(f"\n[2/5] 载入 VAD 模型并对音频进行分段...")
    vad_model = AutoModel(
        model=vad_pytorch_dir,
        trust_remote_code=True,
        device="cpu",
        disable_update=True,
        disable_pbar=True
    )
    
    # 运行 VAD 切分
    vad_out = vad_model.generate(input=audio_path, batch_size_s=5000, max_single_segment_time=60000)
    raw_segs = vad_out[0]['value'] if vad_out and len(vad_out) > 0 and 'value' in vad_out[0] else [[0, len(audio)/16*1000]]
    
    # 合并小切片（最大8秒）
    opt_segs = _merge_vad_segments(raw_segs)
    print(f"原始 VAD 片段数: {len(raw_segs)}, 合并后片段数: {len(opt_segs)}")

    # 切分出所有 chunks 音频片段
    chunks = []
    for start_ms, end_ms in opt_segs:
        s_idx = int(start_ms * 16)
        e_idx = int(end_ms * 16)
        chunk = audio[max(0, s_idx-800):min(len(audio), e_idx+800)]
        if len(chunk) < 1600: continue
        chunks.append(chunk)

    print(f"切分出有效 chunks 数量: {len(chunks)}")
    assert len(chunks) > 0, "没有切分出有效的 chunks"

    # --- 分支一：PyTorch FP32 ASR 推理 ---
    print(f"\n[3/5] 分支一：PyTorch FP32 ASR 推理")
    
    # 1. 测量冷启动（加载模型 + 首轮推理）
    t_py_cold_start = time.time()
    pytorch_model = AutoModel(
        model=asr_pytorch_dir,
        trust_remote_code=True,
        device="cpu",
        disable_update=True
    )
    
    py_res_cold = pytorch_model.generate(input=chunks, batch_size_s=0, language="auto", use_itn=True)
    py_cold_time = time.time() - t_py_cold_start
    print(f"PyTorch 冷启动耗时 (加载+推理): {py_cold_time:.4f} 秒")

    # 2. 测量热启动（仅推理）
    t_py_hot_start = time.time()
    py_res_hot = pytorch_model.generate(input=chunks, batch_size_s=0, language="auto", use_itn=True)
    py_hot_time = time.time() - t_py_hot_start
    print(f"PyTorch 热启动耗时 (仅推理): {py_hot_time:.4f} 秒")

    # 合并 PyTorch 转写文本
    py_texts = []
    for item in py_res_hot:
        raw = item.get('text', '').strip()
        py_texts.append(raw)
    py_full_text = "".join(py_texts)
    
    # 释放内存
    del pytorch_model
    gc.collect()

    # --- 分支二：Sherpa-ONNX INT8 ASR 推理 ---
    print(f"\n[4/5] 分支二：Sherpa-ONNX INT8 ASR 推理")
    
    # 1. 测量冷启动（加载 + 单次批量 decode_streams）
    t_sherpa_cold_start = time.time()
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=sherpa_model_path,
        tokens=sherpa_tokens_path,
        num_threads=6,
        use_itn=True
    )
    
    # 将 chunks 批量封装为 OfflineStream 列表
    cold_streams = []
    for chunk in chunks:
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, chunk)
        cold_streams.append(stream)
        
    recognizer.decode_streams(cold_streams)
    sherpa_cold_time = time.time() - t_sherpa_cold_start
    print(f"Sherpa-ONNX 冷启动耗时 (加载+推理): {sherpa_cold_time:.4f} 秒")

    # 2. 测量热启动（仅批量 decode_streams 推理时间）
    # 在热启动的测量中，仅对 decode_streams 推理时间进行计时
    # 为了公平，先在计时范围外创建新的 streams 列表并填充数据
    hot_streams = []
    for chunk in chunks:
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, chunk)
        hot_streams.append(stream)
        
    t_sherpa_hot_start = time.time()
    recognizer.decode_streams(hot_streams)
    sherpa_hot_time = time.time() - t_sherpa_hot_start
    print(f"Sherpa-ONNX 热启动耗时 (仅推理): {sherpa_hot_time:.4f} 秒")

    # 合并 Sherpa-ONNX 转写文本
    sherpa_texts = [s.result.text.strip() for s in hot_streams]
    sherpa_full_text = "".join(sherpa_texts)

    # --- 统计与对比 ---
    print(f"\n[5/5] 统计与对比结果分析")
    
    # 计算 CER
    dist, clean_p, clean_s, cer = calculate_cer(py_full_text, sherpa_full_text)
    match_rate = 1.0 - cer
    
    # 计算 RTF (热启动)
    py_rtf = py_hot_time / audio_duration
    sherpa_rtf = sherpa_hot_time / audio_duration
    
    # 计算加速比
    cold_speedup = py_cold_time / sherpa_cold_time if sherpa_cold_time > 0 else 0.0
    hot_speedup = py_hot_time / sherpa_hot_time if sherpa_hot_time > 0 else 0.0

    print("\n" + "=" * 80)
    print_row("CPU 评估指标 (6线程锁定)", "PyTorch FP32", "Sherpa-ONNX INT8")
    print("-" * 80)
    print_row("冷启动总耗时 (秒)", f"{py_cold_time:.4f}", f"{sherpa_cold_time:.4f}")
    print_row("冷启动性能比 (Py/Sherpa)", "1.00x", f"{cold_speedup:.2f}x")
    print_row("热启动推理耗时 (秒)", f"{py_hot_time:.4f}", f"{sherpa_hot_time:.4f}")
    print_row("热启动性能加速比", "1.00x", f"{hot_speedup:.2f}x")
    print_row("热启动实时率 (RTF)", f"{py_rtf:.4f}", f"{sherpa_rtf:.4f}")
    print_row("音频总时长 (秒)", f"{audio_duration:.2f}", f"{audio_duration:.2f}")
    print_row("原始识别总字数", f"{len(py_full_text)}", f"{len(sherpa_full_text)}")
    print_row("核心字数 (去标点/空白)", f"{len(clean_p)}", f"{len(clean_s)}")
    print_row("两端字符编辑距离", "-", f"{dist}")
    print_row("字符错误率 (CER)", "-", f"{cer * 100:.2f}%")
    print_row("字符吻合度 (1 - CER)", "-", f"{match_rate * 100:.2f}%")
    print("=" * 80)

    print("\nPyTorch FP32 原始文本 (前 150 字):")
    print(py_full_text[:150] + ("..." if len(py_full_text) > 150 else ""))
    print("-" * 80)
    print("Sherpa-ONNX INT8 原始文本 (前 150 字):")
    print(sherpa_full_text[:150] + ("..." if len(sherpa_full_text) > 150 else ""))
    print("=" * 80 + "\n")
    
    # 做一些基本的 assertion 确保代码没有崩
    assert len(py_full_text) > 0, "PyTorch 转写文本为空"
    assert len(sherpa_full_text) > 0, "Sherpa-ONNX 转写文本为空"
