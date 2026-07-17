"""单元测试：_split_timestamps_to_segments 智能分句函数。

覆盖场景：
1. 标点切分（逗号句号）
2. 静音 gap > 0.8s 切分
3. 5s 超时提前结句 + 长段切半
4. 连续流畅语音不切
5. 单段
6. 空输入
7. offset_ms 偏移
"""

import pytest
from funclip_pro.core.asr import _split_timestamps_to_segments


def test_punctuation_split():
    """标点切分：遇句号应拆成两个 segment"""
    ts = [
        {"text": "今天", "start": 0.1, "end": 0.3},
        {"text": "天气", "start": 0.3, "end": 0.6},
        {"text": "真", "start": 0.6, "end": 0.8},
        {"text": "好", "start": 0.8, "end": 1.0},
        {"text": "。", "start": 1.0, "end": 1.1},
        {"text": "我们", "start": 1.2, "end": 1.4},
        {"text": "去", "start": 1.4, "end": 1.6},
        {"text": "公园", "start": 1.6, "end": 1.9},
        {"text": "吧", "start": 1.9, "end": 2.1},
    ]
    full_text = "今天天气真好。我们去公园吧"
    segs = _split_timestamps_to_segments(ts, full_text)
    assert len(segs) == 2, f"应被切为2段，实际: {len(segs)}"
    assert "今天天气真好" in segs[0]["text"]
    assert "我们去公园吧" in segs[1]["text"]


def test_gap_split():
    """VAD 已切好，gap > 0.8s 不触发切分。"""
    ts = [
        {"text": "第一句", "start": 0.0, "end": 0.8},
        {"text": "第二句", "start": 2.0, "end": 2.8},
    ]
    full_text = "第一句第二句"
    segs = _split_timestamps_to_segments(ts, full_text)
    # gap 不切，无标点时合并为一句
    assert len(segs) == 1, f"gap不应切分: {len(segs)}"
    assert segs[0]["text"] == "第一句第二句"


def test_small_gap_no_split():
    """小 gap (< 0.8s) 且无标点 → 不切"""
    ts = [
        {"text": "今天", "start": 0.0, "end": 0.3},
        {"text": "天气", "start": 0.35, "end": 0.6},  # gap = 0.05s < 0.8
        {"text": "真好", "start": 0.65, "end": 0.9},
    ]
    full_text = "今天天气真好"
    segs = _split_timestamps_to_segments(ts, full_text)
    assert len(segs) == 1, f"不应切分: {len(segs)}"
    assert segs[0]["text"] == "今天天气真好"


def test_long_segment_split():
    """5s 超时 → 应提前结句"""
    # 模拟 6 秒长的无标点段
    ts = []
    for i in range(30):
        ts.append({"text": f"字", "start": i * 0.2, "end": i * 0.2 + 0.15})
    full_text = "字" * 30
    segs = _split_timestamps_to_segments(ts, full_text)
    # 6s 段应被切半
    assert len(segs) >= 2, f"长段应被切分: {len(segs)}"


def test_single_token():
    """只有一个 token → 一个 segment"""
    ts = [{"text": "好", "start": 0.0, "end": 0.5}]
    full_text = "好"
    segs = _split_timestamps_to_segments(ts, full_text)
    assert len(segs) == 1
    assert segs[0]["text"] == "好"
    assert segs[0]["start"] == 0
    assert segs[0]["end"] == 500


def test_empty_input():
    """空输入 → 空列表"""
    assert _split_timestamps_to_segments([], "") == []
    assert _split_timestamps_to_segments([], "abc") == []
    assert _split_timestamps_to_segments([{"text": "a", "start": 0, "end": 1}], "") == []


def test_offset_ms():
    """offset_ms 应正确累加到所有 segment 的时间戳上"""
    ts = [
        {"text": "你好", "start": 0.0, "end": 0.5},
    ]
    full_text = "你好"
    segs = _split_timestamps_to_segments(ts, full_text, offset_ms=5000)
    assert len(segs) == 1
    assert segs[0]["start"] == 5000  # 0 * 1000 + 5000
    assert segs[0]["end"] == 5500    # 500 + 5000


def test_comma_triggers_split():
    """逗号触发阶段1切分，但阶段2会合并（无硬标点+<5s）。"""
    ts = [
        {"text": "你好", "start": 0.0, "end": 0.3},
        {"text": "，", "start": 0.3, "end": 0.4},
        {"text": "今天", "start": 0.5, "end": 0.7},
        {"text": "很好", "start": 0.7, "end": 1.0},
    ]
    full_text = "你好，今天很好"
    segs = _split_timestamps_to_segments(ts, full_text)
    # 逗号是软标点，阶段2合并为一句（对齐FunClip行为）
    assert len(segs) == 1, f"逗号应合并: {len(segs)}"
    assert segs[0]["text"] == "你好，今天很好"


def test_hard_punc_flush():
    """硬标点（？！）强制结算"""
    ts = [
        {"text": "真的", "start": 0.0, "end": 0.3},
        {"text": "吗", "start": 0.3, "end": 0.5},
        {"text": "？", "start": 0.5, "end": 0.6},
        {"text": "当然", "start": 0.7, "end": 1.0},
        {"text": "！", "start": 1.0, "end": 1.1},
    ]
    full_text = "真的吗？当然！"
    segs = _split_timestamps_to_segments(ts, full_text)
    assert len(segs) == 2
    assert segs[0]["text"] == "真的吗？"
    assert segs[1]["text"] == "当然！"


def test_karaoke_char_ts_preserved():
    """_char_ts 在 segment 中被保留，供 ASS karaoke 使用"""
    ts = [
        {"text": "你好", "start": 0.0, "end": 0.3},
        {"text": "世界", "start": 0.3, "end": 0.6},
    ]
    full_text = "你好世界"
    segs = _split_timestamps_to_segments(ts, full_text)
    assert len(segs) == 1
    assert "_char_ts" in segs[0]
    assert len(segs[0]["_char_ts"]) == 2
    assert segs[0]["_char_ts"][0]["text"] == "你好"
    assert segs[0]["_char_ts"][1]["text"] == "世界"
