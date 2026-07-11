import numpy as np
from speaker_engine import segment_sliding_window


def test_sliding_window_basic():
    audio = np.zeros(16000 * 3, dtype=np.float32)  # 3s 静音
    wins = segment_sliding_window(audio, 16000, win_sec=1.5, step_sec=0.5)
    # 3s: 完整窗 0-1.5/0.5-2.0/1.0-2.5/1.5-3.0（4 个，各 1.5s）
    #     + 尾部窗 2.0-3.0（1.0s，重叠保留）→ 共 5 窗
    assert len(wins) == 5
    assert abs(wins[0][0] - 0.0) < 0.01
    assert abs(wins[0][1] - 1.5) < 0.01
    assert abs(wins[-1][1] - 3.0) < 0.01


def test_sliding_window_samples_length():
    audio = np.zeros(16000 * 2, dtype=np.float32)
    wins = segment_sliding_window(audio, 16000, win_sec=1.5, step_sec=0.5)
    # 除最后一窗（尾部短窗）外，其余完整窗均 1.5s = 24000 采样
    full = wins[:-1]
    tail = wins[-1]
    for st, en, samp in full:
        assert len(samp) == int(16000 * 1.5)
    # 尾部窗长度 ∈ [0.5s, 1.5s]
    assert int(16000 * 0.5) <= len(tail[2]) <= int(16000 * 1.5)


def test_sliding_window_tail_partial():
    # 2.2s 音频：完整窗 0-1.5/0.5-2.0，尾部 1.5-2.2 不足一窗但应保留
    audio = np.zeros(int(16000 * 2.2), dtype=np.float32)
    wins = segment_sliding_window(audio, 16000, win_sec=1.5, step_sec=0.5)
    assert wins[-1][1] > 2.0  # 尾部包含到 2.2
