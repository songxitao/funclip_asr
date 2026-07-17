"""单元测试：core/models.py 数据模型。

验证 WordTimestamp / Segment / TranscriptionResult 的构造、字段访问、
from_qwen_ts 工厂方法、to_dict 兼容性。

直接从源文件加载，完全不经过包导入机制。
"""

from __future__ import annotations

import pathlib
import sys
import types


def _load_models_as_module():
    """将 models.py 作为独立模块加载，不触发包导入。"""
    models_path = pathlib.Path(__file__).resolve().parents[2] / "src" / "funclip_pro" / "core" / "models.py"
    source = models_path.read_text(encoding="utf-8")

    mod_name = "_test_funclip_models"
    mod = types.ModuleType(mod_name)
    mod.__file__ = str(models_path)
    # 必须注册到 sys.modules，否则 dataclass 装饰器会因 _is_type 查找失败
    sys.modules[mod_name] = mod

    exec(compile(source, str(models_path), "exec"), mod.__dict__)
    return mod


models = _load_models_as_module()
WordTimestamp = models.WordTimestamp
Segment = models.Segment
TranscriptionResult = models.TranscriptionResult


class TestWordTimestamp:
    """WordTimestamp 构造与工厂方法测试。"""

    def test_basic_construction(self):
        wt = WordTimestamp(text="你好", start_ms=100, end_ms=300)
        assert wt.text == "你好"
        assert wt.start_ms == 100
        assert wt.end_ms == 300
        assert wt.confidence is None

    def test_with_confidence(self):
        wt = WordTimestamp(text="test", start_ms=0, end_ms=100, confidence=0.95)
        assert wt.confidence == 0.95

    def test_from_qwen_ts_no_offset(self):
        qwen_ts = {"text": "hello", "start": 1.5, "end": 2.0}
        wt = WordTimestamp.from_qwen_ts(qwen_ts)
        assert wt.text == "hello"
        assert wt.start_ms == 1500
        assert wt.end_ms == 2000

    def test_from_qwen_ts_with_offset(self):
        qwen_ts = {"text": "world", "start": 0.5, "end": 1.0}
        wt = WordTimestamp.from_qwen_ts(qwen_ts, offset_ms=3000)
        assert wt.text == "world"
        assert wt.start_ms == 3500
        assert wt.end_ms == 4000

    def test_from_qwen_ts_empty_fields(self):
        qwen_ts = {}
        wt = WordTimestamp.from_qwen_ts(qwen_ts)
        assert wt.text == ""
        assert wt.start_ms == 0
        assert wt.end_ms == 0


class TestSegment:
    """Segment 构造与兼容性测试。"""

    def test_basic_construction(self):
        seg = Segment(start_ms=1000, end_ms=5000, text="今天天气不错")
        assert seg.start_ms == 1000
        assert seg.end_ms == 5000
        assert seg.text == "今天天气不错"
        assert seg.speaker == ""
        assert seg.words == []

    def test_with_speaker_and_words(self):
        words = [
            WordTimestamp(text="今天", start_ms=1000, end_ms=1500),
            WordTimestamp(text="天气", start_ms=1500, end_ms=2000),
        ]
        seg = Segment(start_ms=1000, end_ms=5000, text="今天天气", speaker="1", words=words)
        assert seg.speaker == "1"
        assert len(seg.words) == 2
        assert seg.words[0].text == "今天"
        assert seg.words[1].end_ms == 2000

    def test_to_dict(self):
        seg = Segment(start_ms=100, end_ms=500, text="test", speaker="0")
        d = seg.to_dict()
        assert d["start"] == 100
        assert d["end"] == 500
        assert d["text"] == "test"
        assert d["speaker"] == "0"

    def test_to_dict_no_speaker(self):
        seg = Segment(start_ms=0, end_ms=100, text="hello")
        d = seg.to_dict()
        assert d["speaker"] == ""

    def test_field_types(self):
        """验证字段类型正确。"""
        seg = Segment(start_ms=0, end_ms=100, text="x")
        assert isinstance(seg.start_ms, int)
        assert isinstance(seg.end_ms, int)


class TestTranscriptionResult:
    """TranscriptionResult 构造与属性测试。"""

    def test_minimal_construction(self):
        result = TranscriptionResult(text="hello world", engine="qwen", segments=[])
        assert result.text == "hello world"
        assert result.engine == "qwen"
        assert result.segments == []
        assert result.language == "auto"
        assert result.duration_ms == 0
        assert result.diarized_text == ""

    def test_full_construction(self):
        segs = [
            Segment(start_ms=0, end_ms=1000, text="hello"),
            Segment(start_ms=1000, end_ms=2000, text="world", speaker="1"),
        ]
        result = TranscriptionResult(
            text="hello world",
            engine="seaco",
            segments=segs,
            language="zh",
            duration_ms=2000,
            diarized_text="[说话人1] world",
        )
        assert result.engine == "seaco"
        assert len(result.segments) == 2
        assert result.language == "zh"
        assert result.duration_ms == 2000
        assert "说话人" in result.diarized_text

    def test_segment_access(self):
        """验证 segments 列表按索引访问正确。"""
        segs = [
            Segment(start_ms=0, end_ms=100, text="a"),
            Segment(start_ms=100, end_ms=200, text="b"),
        ]
        result = TranscriptionResult(text="ab", engine="test", segments=segs)
        assert result.segments[0].text == "a"
        assert result.segments[1].end_ms == 200

    def test_empty_segments(self):
        result = TranscriptionResult(text="", engine="test", segments=[])
        assert len(result.segments) == 0
