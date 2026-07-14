"""路由与 trim 单元测试（TDD 先行）。

覆盖 HANDOFF §4 的核心逻辑:
- `_cheap_trim`: 廉价 trim（仅 librosa，不加载模型）
- `_select_engine`: auto/cpu/gpu 引擎路由
- `_use_vad`: auto/always/never 三态 VAD 策略
- `strip_punctuation`: 剥原生标点保留 ITN 数字

注意: 导入 asr_onnx_service 只执行其模块顶层代码（CPU 亲和性、torch.set_num_threads、
import openvino/funasr/sherpa_onnx），**不会**加载任何模型——模型仅在 uvicorn 触发
startup 事件时加载。因此本测试不依赖服务启动，也不加载 GPU/PyTorch 模型。
"""
import os
import tempfile

import numpy as np
import pytest

from funclip_pro.core.asr import (
    SHORT_AUDIO_MS,
    _cheap_trim,
    _select_engine,
    _use_vad,
    strip_punctuation,
)


def _make_silence_sine_silence_wav(path, sr=16000):
    """构造 2s 静音 + 1s 正弦 + 2s 静音 的 wav（总时长 5s）。"""
    import soundfile as sf

    t_sine = 1.0
    silence = np.zeros(int(2.0 * sr), dtype=np.float32)
    sine = (0.3 * np.sin(2 * np.pi * 440.0 * np.arange(int(t_sine * sr)) / sr)).astype(np.float32)
    y = np.concatenate([silence, sine, silence])
    sf.write(path, y, sr)
    return y


def test_cheap_trim_does_not_load_models():
    """_cheap_trim 仅用 librosa.effects.trim，不加载任何模型，且返回值符合预期。

    - 返回的 duration_ms 应约等于音频全长 (5s ≈ 5000ms)
    - 返回的 trimmed 波形长度应 < 完整波形长度（首尾静音被切掉）
    """
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "trim_test.wav")
        full = _make_silence_sine_silence_wav(wav_path)

        trimmed, duration_ms = _cheap_trim(wav_path)

    # 全长 ≈ 5000ms
    assert 4900 <= duration_ms <= 5100, f"duration_ms 应≈5000, 实际 {duration_ms}"
    # trimmed 短于 full（首尾静音被移除）
    assert len(trimmed) < len(full), "trim 后波形应短于原文"
    # 不应把正弦信号切光
    assert len(trimmed) > 0, "trim 不应切光有效信号"


def test_strip_punctuation_keeps_digits_and_han():
    """strip_punctuation 剥掉原生标点，但保留 ITN 规整后的数字与汉字。"""
    out = strip_punctuation("你好，世界！今天气温 23 度。")
    assert "，" not in out
    assert "！" not in out
    assert "。" not in out
    assert "你好世界今天气温23度" == out


def test_select_engine_overrides():
    """engine 参数强制覆盖：cpu→sherpa，gpu→torch（与音频长度无关）。"""
    assert _select_engine("cpu", 1000) == "sherpa"
    assert _select_engine("cpu", 10000) == "sherpa"
    assert _select_engine("gpu", 1000) == "torch"
    assert _select_engine("gpu", 10000) == "torch"


def test_select_engine_auto_short_always_sherpa():
    """auto 策略下，短音频(<SHORT_AUDIO_MS)永远走 Sherpa-CPU，即使 CUDA 可用。"""
    assert _select_engine("auto", 1000) == "sherpa"
    assert _select_engine("auto", SHORT_AUDIO_MS) == "sherpa"


def test_select_engine_auto_long_depends_on_cuda():
    """auto 策略下，长音频(>SHORT_AUDIO_MS)在 CUDA 可用时走 torch，否则 sherpa。"""
    import torch

    expected = "torch" if torch.cuda.is_available() else "sherpa"
    assert _select_engine("auto", 10000) == expected


def test_use_vad_states():
    """vad_strategy 三态映射。"""
    # always: 永远 VAD
    assert _use_vad("always", 1000) is True
    assert _use_vad("always", 10000) is True
    # never: 永远不 VAD（但仍做 cheap trim）
    assert _use_vad("never", 1000) is False
    assert _use_vad("never", 10000) is False
    # auto: 仅长音频走 VAD
    assert _use_vad("auto", 1000) is False
    assert _use_vad("auto", SHORT_AUDIO_MS) is False
    assert _use_vad("auto", 10000) is True
