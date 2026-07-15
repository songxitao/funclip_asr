"""P3.2 — core.audio 音频采集层下沉单测。

Seam 1: process_audio_frame() 纯函数
Seam 2: MixedStream.read() 混音逻辑
Seam 3: iter_frames() generator 协议
Seam 4: 硬件集成测试（跳过）
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from funclip_pro.core.audio import (          # noqa: E402
    process_audio_frame,
    MixedStream,
    MicStream,
    LoopbackStream,
    BaseStream,
)

# ==============================
# Seam 1: process_audio_frame()
# ==============================


def test_process_audio_frame_int16_to_float32():
    """int16 PCM→float32 [-1, 1] 转换。"""
    raw = np.array([0, 32767, -32768, 16384, -16384], dtype=np.int16)
    out = process_audio_frame(raw, src_channels=1, src_rate=16000, volume_boost=1.0)
    assert out.dtype == np.float32
    expected = np.array([0.0, 1.0, -1.0, 0.5, -0.5], dtype=np.float32)
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_process_audio_frame_stereo_to_mono():
    """立体声→单声道降混。"""
    raw = np.array([[1000, 2000], [3000, 4000], [5000, 6000]], dtype=np.int16).tobytes()
    arr = np.frombuffer(raw, dtype=np.int16)
    out = process_audio_frame(arr, src_channels=2, src_rate=16000, volume_boost=1.0)
    assert out.dtype == np.float32
    assert out.ndim == 1
    # 手动计算：先 reshape 成 (3,2), mean(axis=1), 然后 /32768
    expected = np.array([1500, 3500, 5500], dtype=np.float32) / 32768.0
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_process_audio_frame_resample_48k_to_16k():
    """48kHz→16kHz 重采样（3 倍压缩）。"""
    # 48kHz 下 480 个采样 = 10ms, 16kHz 下应为 160 个采样
    raw = np.arange(480, dtype=np.int16)
    out = process_audio_frame(raw, src_channels=1, src_rate=48000, volume_boost=1.0)
    assert len(out) == 160
    assert out.dtype == np.float32


def test_process_audio_frame_volume_boost():
    """VOLUME_BOOST 增益。"""
    raw = np.array([10000, -5000, 0], dtype=np.int16)
    out = process_audio_frame(raw, src_channels=1, src_rate=16000, volume_boost=3.0)
    # 10000/32768*3 ≈ 0.9155, -5000/32768*3 ≈ -0.4578, 0/32768*3 = 0
    expected = np.array([10000, -5000, 0], dtype=np.float32) / 32768.0 * 3.0
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_process_audio_frame_clip():
    """增益后截断 [-1, 1]。"""
    raw = np.array([20000, -20000], dtype=np.int16)
    out = process_audio_frame(raw, src_channels=1, src_rate=16000, volume_boost=3.0)
    # 20000/32768*3 ≈ 1.831, 应截断为 1.0
    # -20000/32768*3 ≈ -1.831, 应截断为 -1.0
    expected = np.array([1.0, -1.0], dtype=np.float32)
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_process_audio_frame_default_volume_boost():
    """默认 volume_boost=3.0。"""
    raw = np.array([10000], dtype=np.int16)
    out = process_audio_frame(raw, src_channels=1, src_rate=16000)
    expected = np.float32(10000.0 / 32768.0 * 3.0)
    assert abs(out[0] - expected) < 1e-5


def test_process_audio_frame_mono_passthrough():
    """1 声道输入不应该 reshape。"""
    raw = np.array([1000, 2000, 3000], dtype=np.int16)
    out = process_audio_frame(raw, src_channels=1, src_rate=16000, volume_boost=1.0)
    assert out.ndim == 1
    assert len(out) == 3


def test_process_audio_frame_same_rate_no_resample():
    """src_rate=16000 时不执行重采样。"""
    raw = np.array([1000, 2000, 3000], dtype=np.int16)
    out = process_audio_frame(raw, src_channels=1, src_rate=16000, volume_boost=1.0)
    assert len(out) == 3


# ===========================
# Seam 2: MixedStream.read()
# ===========================


def test_mixed_stream_both_sources_average():
    """双源都有数据时：平均混音。"""
    ms = MixedStream()
    # 直接往内部队列塞原始 int16 bytes
    # 注意：MicStream/LoopbackStream.__init__ 会创建 PyAudio (但 pyaudio 库已安装)
    # 只要不调用 start(), 就不会真的访问硬件
    arr_mic = np.array([10000, 20000, 30000], dtype=np.int16)
    arr_loop = np.array([4000, 8000, 12000], dtype=np.int16)
    ms.mic.q.put(arr_mic.tobytes())
    ms.loop.q.put(arr_loop.tobytes())
    out = ms.read()
    # 预期: (mic_val + loop_val) / 2
    expected = np.clip(
        (arr_mic.astype(np.float32) / 32768.0 + arr_loop.astype(np.float32) / 32768.0) / 2 * 3.0,
        -1.0, 1.0,
    )
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_mixed_stream_only_mic():
    """只有 Mic 时有 mic 数据（含 clip）。"""
    ms = MixedStream()
    arr = np.array([10000, 20000, 30000], dtype=np.int16)
    ms.mic.q.put(arr.tobytes())
    # loopback 队列空
    out = ms.read()
    expected = np.clip(arr.astype(np.float32) / 32768.0 * 3.0, -1.0, 1.0)
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_mixed_stream_only_loopback():
    """只有 Loopback 时有 loopback 数据（含 clip）。"""
    ms = MixedStream()
    arr = np.array([5000, 10000, 15000], dtype=np.int16)
    ms.loop.q.put(arr.tobytes())
    out = ms.read()
    expected = np.clip(arr.astype(np.float32) / 32768.0 * 3.0, -1.0, 1.0)
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_mixed_stream_both_empty_zeros():
    """都空时返回全零帧 (512 长度)。"""
    ms = MixedStream()
    out = ms.read()
    assert out.dtype == np.float32
    assert len(out) == 512
    assert np.all(out == 0.0)


# ===========================
# Seam 3: iter_frames()
# ===========================


def test_iter_frames_yields_float32():
    """iter_frames() 生成 float32 ndarray。"""
    ms = MixedStream()
    arr = np.array([10000, 20000], dtype=np.int16)
    ms.mic.q.put(arr.tobytes())
    frames = list(ms.iter_frames(max_frames=1))
    assert len(frames) == 1
    assert isinstance(frames[0], np.ndarray)
    assert frames[0].dtype == np.float32


def test_iter_frames_multiple():
    """多个数据帧 yield。"""
    ms = MixedStream()
    for _ in range(3):
        arr = np.array([10000, 20000], dtype=np.int16)
        ms.mic.q.put(arr.tobytes())
    frames = list(ms.iter_frames(max_frames=3))
    assert len(frames) == 3
    for f in frames:
        assert f.dtype == np.float32


def test_iter_frames_max_frames_limit():
    """max_frames 限制迭代次数。"""
    ms = MixedStream()
    for _ in range(10):
        arr = np.array([10000, 20000], dtype=np.int16)
        ms.mic.q.put(arr.tobytes())
    frames = list(ms.iter_frames(max_frames=4))
    assert len(frames) == 4


def test_iter_frames_no_max_frames():
    """不设 max_frames 时用空队列会卡住，用超时机制测试。"""
    ms = MixedStream()
    arr = np.array([10000, 20000], dtype=np.int16)
    ms.mic.q.put(arr.tobytes())
    gen = ms.iter_frames()
    f = next(gen)
    assert f.dtype == np.float32


# ===========================
# Seam 4: 硬件集成测试（跳过）
# ===========================


@pytest.mark.skip(reason="需要真实音频硬件，不在 CI 中运行")
def test_loopback_start_stop():
    pass


@pytest.mark.skip(reason="需要真实音频硬件，不在 CI 中运行")
def test_mic_start_stop():
    pass
