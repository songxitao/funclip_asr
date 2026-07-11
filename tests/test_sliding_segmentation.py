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


# ---- cluster_sliding 单测（mock extract_embedding，不加载 Cam++ 模型）----
from unittest.mock import patch
from speaker_engine import CampPlusSpeaker


def test_cluster_sliding_merges_same_speaker():
    """前若干窗返回向量A、其余返回向量B，验证合并后得到 2 段（A 段 + B 段）。"""
    spk = CampPlusSpeaker.__new__(CampPlusSpeaker)  # 不加载模型
    emb_a = np.array([1.0, 0.0, 0.0])
    emb_b = np.array([0.0, 1.0, 0.0])
    state = {"i": 0}

    def fake_extract(samp):
        i = state["i"]
        state["i"] += 1
        return emb_a if i < 3 else emb_b

    with patch.object(spk, "extract_embedding", side_effect=fake_extract):
        audio = np.zeros(16000 * 6, dtype=np.float32)  # 6s -> 多个窗
        merged = spk.cluster_sliding(audio, sr=16000, n_speakers=2,
                                     win_sec=1.5, step_sec=0.5)
    # 应合并成 2 段（A 连续 + B 连续）
    assert len(merged) == 2
    assert merged[0][2] != merged[1][2]  # 两段不同人


def test_cluster_sliding_single_speaker():
    """所有窗同一人，合并成 1 段。"""
    spk = CampPlusSpeaker.__new__(CampPlusSpeaker)
    emb = np.array([1.0, 0.0])

    def fake_extract(samp):
        return emb

    with patch.object(spk, "extract_embedding", side_effect=fake_extract):
        audio = np.zeros(16000 * 5, dtype=np.float32)
        merged = spk.cluster_sliding(audio, sr=16000, n_speakers=1,
                                     win_sec=1.5, step_sec=0.5)
    assert len(merged) == 1


def test_cluster_sliding_none_embedding_filled():
    """某窗 extract_embedding 返回 None，应用前后窗标签填充，不崩。"""
    spk = CampPlusSpeaker.__new__(CampPlusSpeaker)
    emb = np.array([1.0, 0.0])
    state = {"i": 0}

    def fake_extract(samp):
        i = state["i"]
        state["i"] += 1
        return None if i == 1 else emb  # 第 2 窗失败，其余正常

    with patch.object(spk, "extract_embedding", side_effect=fake_extract):
        audio = np.zeros(16000 * 4, dtype=np.float32)
        merged = spk.cluster_sliding(audio, sr=16000, n_speakers=1,
                                     win_sec=1.5, step_sec=0.5)
    assert len(merged) >= 1  # 不崩即可

