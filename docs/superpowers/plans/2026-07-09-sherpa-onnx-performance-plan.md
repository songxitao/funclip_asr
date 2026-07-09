# 2026-07-09-sherpa-onnx-performance-plan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 评估新下载的 SenseVoiceSmall INT8 模型在 `sherpa-onnx` 推理引擎下的 CPU 推理速度与精度表现。

**Architecture:** 在独立测试脚本中使用 `sherpa-onnx.OfflineRecognizer` 加载新模型与 tokens 文件，加载真实音频 `李雪花2.wav`，进行纯推理跑评，获取冷启动时延、热启动时延以及 RTF，并验证识别文字正确率。

**Tech Stack:** Python 3.10, sherpa-onnx, numpy, librosa, pytest.

---

### Task 1: 准备物理环境与依赖安装

**Files:**
- N/A

- [ ] **Step 1: 安装 sherpa-onnx 依赖**

Run: `E:\conda\envs\asr_ui_env\python.exe -m pip install sherpa-onnx`
Expected: 成功安装 `sherpa-onnx` 且无版本冲突。

- [ ] **Step 2: 验证安装是否成功**

Run: `E:\conda\envs\asr_ui_env\python.exe -c "import sherpa_onnx; print(sherpa_onnx.__version__)"`
Expected: 正常输出 `sherpa-onnx` 的版本号，无 DLL 丢失等报错。

---

### Task 2: 编写独立测试脚本 `test_sherpa_performance.py`

**Files:**
- Create: [tests/test_sherpa_performance.py](file:///E:/project/funclip-pro/tests/test_sherpa_performance.py)

- [ ] **Step 1: 创建测试脚本**
在 `tests` 目录下新建测试脚本 `tests/test_sherpa_performance.py`，完整实现模型载入、预热、长音频推理及 RTF、精度统计。

```python
import os
import sys
import time
import numpy as np
import librosa
import sherpa_onnx

def test_sherpa_performance():
    tokens_path = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmallOnnx\tokens.txt"
    model_path = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmallOnnx\model.int8.onnx"
    audio_path = r"E:\下载\下载\李雪花2.wav"

    assert os.path.exists(tokens_path), f"Tokens file not found: {tokens_path}"
    assert os.path.exists(model_path), f"Model file not found: {model_path}"
    assert os.path.exists(audio_path), f"Audio file not found: {audio_path}"

    print("=" * 60)
    print("【开始 sherpa-onnx CPU 性能跑评测试】")
    print("=" * 60)

    # 1. 测量引擎加载时间（冷启动第一部分）
    t_load_start = time.time()
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        tokens=tokens_path,
        sense_voice_model=model_path,
        num_threads=6,
        use_itn=True,
    )
    t_load = (time.time() - t_load_start) * 1000
    print(f"1. 引擎与模型加载耗时: {t_load:.2f} ms")

    # 2. 加载音频数据
    print("正在使用 librosa 加载音频数据...")
    samples, sr = librosa.load(audio_path, sr=16000)
    duration = len(samples) / sr
    print(f"音频加载完成，时长: {duration:.2f} 秒")

    # 3. 测量冷启动推理时间（首次推理）
    t_first_start = time.time()
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, samples)
    recognizer.decode(stream)
    text_first = stream.result.text
    t_first = (time.time() - t_first_start) * 1000
    print(f"2. 冷启动首轮推理耗时: {t_first:.2f} ms (RTF: {(t_first/1000)/duration:.4f})")
    print(f"首轮转写结果: {text_first[:100]}...")

    # 4. 测量热启动多轮推理时间
    print("进行 3 轮热启动循环测试...")
    latencies = []
    for i in range(3):
        t_start = time.time()
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, samples)
        recognizer.decode(stream)
        _ = stream.result.text
        latencies.append((time.time() - t_start) * 1000)
    
    avg_latency = np.mean(latencies)
    rtf = (avg_latency / 1000) / duration
    print(f"3. 热启动推理平均耗时: {avg_latency:.2f} ms")
    print(f"4. 最终 RTF (实时率): {rtf:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    test_sherpa_performance()
```

- [ ] **Step 2: 运行测试并收集数据**

Run: `E:\conda\envs\asr_ui_env\python.exe tests/test_sherpa_performance.py`
Expected: 运行成功，完整打印加载耗时、冷启动耗时、热启动平均耗时、最终 RTF 以及转写出来的文字内容。

- [ ] **Step 3: 提交测试脚本**

Run:
```bash
git add tests/test_sherpa_performance.py
git commit -m "test: add sherpa-onnx performance benchmark test script"
```
Expected: 成功提交测试代码。
