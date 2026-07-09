import os
import sys
import time
import librosa
import numpy as np
import sherpa_onnx

# 限制线程数环境变量
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

def test_performance():
    tokens_path = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmallOnnx\tokens.txt"
    model_path = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmallOnnx\model.int8.onnx"
    audio_path = r"E:\下载\下载\李雪花2.wav"

    print("\n" + "="*50)
    print("开始 Sherpa-ONNX SenseVoiceSmall 性能评测与基准测试")
    print("="*50)

    # 1. 引擎加载时间
    t_start = time.time()
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=model_path,
        tokens=tokens_path,
        num_threads=6,
        use_itn=True
    )
    engine_load_time = time.time() - t_start
    print(f"引擎加载时间: {engine_load_time:.4f} 秒")

    # 2. 加载音频并计算时长
    if not os.path.exists(audio_path):
        print(f"错误: 测试音频文件不存在 -> {audio_path}")
        sys.exit(1)
        
    t_audio_start = time.time()
    audio, sr = librosa.load(audio_path, sr=16000)
    audio_duration = len(audio) / sr
    audio_load_time = time.time() - t_audio_start
    print(f"音频加载成功 (时长: {audio_duration:.2f} 秒, 加载耗时: {audio_load_time:.4f} 秒)")

    # 3. 测量冷启动推理时间（首轮推理）
    t_cold_start = time.time()
    stream = recognizer.create_stream()
    stream.accept_waveform(sr, audio)
    recognizer.decode_stream(stream)
    text_cold = stream.result.text
    cold_infer_time = time.time() - t_cold_start
    print(f"冷启动推理时间: {cold_infer_time:.4f} 秒")

    # 4. 测量热启动多轮（3轮）推理时间与 RTF
    hot_times = []
    text_hot = ""
    for idx in range(3):
        t_hot_start = time.time()
        stream = recognizer.create_stream()
        stream.accept_waveform(sr, audio)
        recognizer.decode_stream(stream)
        text_hot = stream.result.text
        duration = time.time() - t_hot_start
        hot_times.append(duration)
        print(f"热启动第 {idx + 1} 轮推理时间: {duration:.4f} 秒")

    avg_hot_time = sum(hot_times) / len(hot_times)
    rtf = avg_hot_time / audio_duration

    print("\n" + "="*50)
    print("测试结果数据汇总:")
    print(f"引擎加载时间: {engine_load_time:.4f} 秒")
    print(f"冷启动时间: {cold_infer_time:.4f} 秒")
    print(f"热启动平均时间: {avg_hot_time:.4f} 秒")
    print(f"音频时长: {audio_duration:.2f} 秒")
    print(f"RTF (实时率): {rtf:.4f}")
    
    # 打印转写出的首 100 字文本
    print(f"\n转写出的首 100 字文本:\n{text_hot[:100]}")
    print("="*50 + "\n")

if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    test_performance()
