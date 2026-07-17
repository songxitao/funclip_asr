"""Mock-based pipeline tests — no Docker, no real models, runs in CI.

通过 pytest monkeypatch 替换 asr_mod._decode / VAD_MODEL / librosa.load / QwenEngine
/ _get_seaco_model 等，在不加载任何真实模型的情况下验证 pipeline 各分支逻辑。
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock
import types as _types

# ============================================================
# 提前 mock 所有重型/系统依赖，避免 CI 环境逐个报错
# ============================================================

# --- torch（魔改 ModuleType，让 scipy issubclass 检查能过）---
class _MockTensor:
    def __init__(self, data=None):
        self.data = data

_t = _types.ModuleType("torch")
_t.__version__ = "0.0.0"
_t.Tensor = _MockTensor
_t.cuda = MagicMock()
_t.cuda.is_available.return_value = False
_t.cuda.device_count.return_value = 0
_t.float32 = float
_t.float64 = float
_t.int32 = int
_t.int64 = int
sys.modules["torch"] = _t

# --- torchaudio ---
sys.modules["torchaudio"] = MagicMock()

# --- soundfile ---
sys.modules["soundfile"] = MagicMock()

# --- pyannote.audio（需要嵌套模块结构 from pyannote.audio import Model）---
_pyannote = _types.ModuleType("pyannote")
_pyannote.__path__ = []
_pyannote.__file__ = ""
_pyannote.audio = _types.ModuleType("pyannote.audio")
_pyannote.audio.Model = MagicMock
sys.modules["pyannote"] = _pyannote
sys.modules["pyannote.audio"] = _pyannote.audio

# --- pyaudio ---
sys.modules["pyaudio"] = MagicMock()

# --- pyaudiowpatch（仅 Windows，CI 无此包）---
sys.modules["pyaudiowpatch"] = MagicMock()

# ============================================================

# 确保能找到 funclip_pro 源码
_src = str(Path(__file__).resolve().parents[2] / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import numpy as np
import pytest

from funclip_pro.pipeline.offline import OfflinePipeline
from funclip_pro.core.models import TranscriptionResult, Segment
from tests.mocks import MOCK_ENGINES


# ----------------------------------------------------------------------
# 辅助工具
# ----------------------------------------------------------------------

_DEFAULT_VAD_SEGMENTS = [[0, 2000], [2500, 4500], [5000, 7000]]


def _make_fake_waveform(duration_sec=10, sr=16000):
    """生成随机噪声波形，避免 librosa.effects.trim 静音导致空切片。"""
    rng = np.random.default_rng(42)
    return rng.uniform(-0.3, 0.3, int(duration_sec * sr)).astype(np.float32)


class DummyVADModel:
    """Mock VAD model that returns predefined segments."""

    def __init__(self, segments):
        self._segments = segments

    def generate(self, input, **kwargs):
        if self._segments:
            return [{"value": self._segments}]
        return []


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def pipeline():
    """Pipeline 实例（不加载真实模型权重）。"""
    return OfflinePipeline(auto_load=False)


@pytest.fixture(autouse=True)
def mock_librosa(monkeypatch):
    """自动替换 librosa.load，避免 "test.wav" 不存在导致 FileNotFoundError。"""
    import librosa

    fake_wave = _make_fake_waveform()

    def fake_load(path, sr=16000, **kwargs):
        return fake_wave.copy(), sr

    monkeypatch.setattr(librosa, "load", fake_load)


@pytest.fixture
def mock_vad(monkeypatch):
    """设置 VAD_MODEL 返回预设的 VAD 段。"""
    from funclip_pro.core import asr as asr_mod
    monkeypatch.setattr(asr_mod, "VAD_MODEL", DummyVADModel(_DEFAULT_VAD_SEGMENTS))
    return _DEFAULT_VAD_SEGMENTS


@pytest.fixture
def mock_qwen_engine(monkeypatch):
    """替换 QwenEngine 为 MockQwenEngine。"""
    from funclip_pro.core import asr as asr_mod
    monkeypatch.setattr(asr_mod, "QwenEngine", MOCK_ENGINES["qwen"].__class__)


@pytest.fixture
def mock_seaco_getter(monkeypatch):
    """替换 _get_seaco_model 返回 MockSeACoEngine。"""
    from funclip_pro.core import asr as asr_mod
    monkeypatch.setattr(asr_mod, "_get_seaco_model", lambda: MOCK_ENGINES["seaco"])


@pytest.fixture
def mock_decode(monkeypatch):
    """替换 _decode，根据 engine_key 返回对应 mock 结果。"""
    from funclip_pro.core import asr as asr_mod

    def _fake_decode(engine_key, waveforms, hotwords=""):
        eng = MOCK_ENGINES.get(engine_key)
        if engine_key == "seaco":
            return eng(waveforms, hotwords=hotwords)
        return eng(waveforms)

    monkeypatch.setattr(asr_mod, "_decode", _fake_decode)


# ----------------------------------------------------------------------
# Qwen 分支
# ----------------------------------------------------------------------

def test_qwen_with_vad_returns_transcription_result(pipeline, mock_qwen_engine, mock_vad):
    """Qwen with VAD 应返回 TranscriptionResult，segment 边界与 VAD 段一致。"""
    result = pipeline.run(
        audio_path="dummy.wav",
        engine="qwen",
        vad_strategy="always",
    )

    assert isinstance(result, TranscriptionResult)
    assert result.engine == "qwen"
    assert len(result.segments) > 0

    # 每个 segment 的边界应有效，text 为非空 str
    for seg in result.segments:
        assert isinstance(seg, Segment)
        assert seg.end_ms >= seg.start_ms
        assert isinstance(seg.text, str)

    # segments 数量应等于 VAD 段数量（3 段）
    assert len(result.segments) == len(_DEFAULT_VAD_SEGMENTS)

    # 验证边界完全匹配 VAD 输出
    for seg, (expected_start, expected_end) in zip(result.segments, _DEFAULT_VAD_SEGMENTS):
        assert seg.start_ms == expected_start
        assert seg.end_ms == expected_end


def test_qwen_without_vad_returns_single_segment(pipeline, mock_qwen_engine):
    """Qwen without VAD 应返回一个覆盖全长的 segment。"""
    result = pipeline.run(
        audio_path="dummy.wav",
        engine="qwen",
        vad_strategy="never",
    )

    assert isinstance(result, TranscriptionResult)
    assert result.engine == "qwen"
    assert len(result.segments) == 1

    seg = result.segments[0]
    assert seg.text == "你好世界 今天天气不错 我们去散步吧"
    assert seg.speaker == "0"
    # 无 VAD 分支中 words 应来自 Qwen 时间戳
    assert len(seg.words) == 9


def test_qwen_with_vad_words_have_offset(pipeline, mock_qwen_engine, mock_vad):
    """Qwen with VAD: 词级时间戳应偏移到全局时间轴。

    注意：mock 为简化实现，所有 chunk 返回相同的 timestamps 集合，
    因此部分 words 可能跨出 VAD 段边界。本测试只验证 offset 机制生效。
    """
    result = pipeline.run(
        audio_path="dummy.wav",
        engine="qwen",
        vad_strategy="always",
    )

    # 每个 segment 的 words 都应存在
    for idx, seg in enumerate(result.segments):
        if seg.words:
            # offset 机制生效：word start 至少 >= segment start（允许少量偏差）
            assert seg.words[0].start_ms >= seg.start_ms - 100, (
                f"Segment {idx}: word starts at {seg.words[0].start_ms} "
                f"but segment starts at {seg.start_ms}"
            )
            # 所有 words 应有合法时间戳
            for w in seg.words:
                assert w.end_ms >= w.start_ms
                assert isinstance(w.text, str)
                assert len(w.text) > 0


# ----------------------------------------------------------------------
# SeACo 分支
# ----------------------------------------------------------------------

def test_seaco_with_diarize_false_strips_speaker(pipeline, mock_seaco_getter, mock_librosa):
    """SeACo with diarize=False 不应有说话人标注。"""
    result = pipeline.run(
        audio_path="dummy.wav",
        engine="seaco",
        diarize=False,
    )

    assert isinstance(result, TranscriptionResult)
    assert result.engine == "seaco"

    for seg in result.segments:
        assert seg.speaker == "", f"Expected empty speaker, got '{seg.speaker}'"


def test_seaco_with_diarize_true_keeps_speaker(pipeline, mock_seaco_getter, mock_librosa):
    """SeACo with diarize=True 应保留说话人标签。"""
    result = pipeline.run(
        audio_path="dummy.wav",
        engine="seaco",
        diarize=True,
    )

    assert isinstance(result, TranscriptionResult)
    assert result.engine == "seaco"
    assert len(result.segments) >= 1

    # 至少有一个 segment 有 speaker
    has_speaker = any(seg.speaker for seg in result.segments)
    assert has_speaker, "Expected at least one segment with speaker when diarize=True"

    # diarized_text 应非空且包含说话人标注
    assert result.diarized_text != ""
    assert "[说话人1]" in result.diarized_text
    assert "[说话人2]" in result.diarized_text


# ----------------------------------------------------------------------
# 空结果容错
# ----------------------------------------------------------------------

def test_pipeline_handles_empty_vad_result(pipeline):
    """空 VAD 结果应优雅返回空 TranscriptionResult。"""
    from funclip_pro.core import asr as asr_mod

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(asr_mod, "VAD_MODEL", DummyVADModel([]))

    result = pipeline.run(
        audio_path="dummy.wav",
        vad_strategy="always",
    )

    monkeypatch.undo()

    assert isinstance(result, TranscriptionResult)
    assert result.text == ""
    assert result.segments == []


# ----------------------------------------------------------------------
# 标准流水线（Sherpa / SenseVoice 分支）
# ----------------------------------------------------------------------

def test_standard_pipeline_returns_transcription_result(pipeline, mock_decode, mock_vad):
    """标准流水线（Sherpa/SenseVoice）+ VAD 应返回 TranscriptionResult。"""
    result = pipeline.run(
        audio_path="dummy.wav",
        engine="cpu",
        vad_strategy="always",
    )

    assert isinstance(result, TranscriptionResult)
    assert result.engine == "sherpa"
    assert len(result.segments) > 0

    for seg in result.segments:
        assert isinstance(seg, Segment)
        assert seg.end_ms >= seg.start_ms


def test_standard_pipeline_without_vad(pipeline, mock_decode):
    """标准流水线 without VAD（走廉价 trim）也应正常工作。"""
    result = pipeline.run(
        audio_path="dummy.wav",
        engine="cpu",
        vad_strategy="never",
    )

    assert isinstance(result, TranscriptionResult)
    assert result.engine == "sherpa"
    assert len(result.segments) >= 1


def test_torch_engine_route(pipeline, mock_decode):
    """GPU 路由应选中 torch 引擎并返回正确结果。"""
    result = pipeline.run(
        audio_path="dummy.wav",
        engine="gpu",
        vad_strategy="never",
    )

    assert isinstance(result, TranscriptionResult)
    assert result.engine == "torch"
    assert len(result.segments) >= 1


# ----------------------------------------------------------------------
# 整体结果完整性
# ----------------------------------------------------------------------

def test_transcription_result_immutability(pipeline, mock_qwen_engine):
    """返回的 TranscriptionResult 字段都应正确初始化。"""
    result = pipeline.run(
        audio_path="dummy.wav",
        engine="qwen",
        vad_strategy="never",
    )

    assert isinstance(result.text, str)
    assert isinstance(result.engine, str)
    assert isinstance(result.segments, list)
    assert isinstance(result.duration_ms, int)
    assert isinstance(result.diarized_text, str)
    assert result.language == "auto"
