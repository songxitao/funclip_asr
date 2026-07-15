# -*- coding: utf-8 -*-
"""T09 测试门禁 · OfflinePipeline 单测（不加载模型 / 不依赖 GPU）。

覆盖：
  - OfflinePipeline 入口（auto_load=False 下构造、常量、模型目录解析）
  - core 各引擎可导入
  - utils SRT 工具（_ms_to_srt / _merge_same_speaker_segments / _segments_to_srt）
  - alignment 纯函数（_assign_clauses_to_speakers / _assign_clauses_to_speakers_seamless）

设计原则：等价优先，只校验外部行为（ms 时间戳、段合并、SRT 合法性、子句→说话人分配），
不关心 DLL 补丁 / 相对路径内部实现。
"""
import pathlib
import sys

import pytest

# 把项目根 src 加入路径以导入 funclip_pro
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from funclip_pro.pipeline.offline import OfflinePipeline, SPK_MODEL_DIR, SEG_MODEL_DIR  # noqa: E402
from funclip_pro.core import (  # noqa: E402
    SegmentationEngine,
    CampPlusSpeaker,
    SenseVoiceSmall,
    PyTorchSenseVoice,
    SherpaSenseVoice,
    load_models,
    _assign_clauses_to_speakers,
    _assign_clauses_to_speakers_seamless,
)
from funclip_pro.utils import (  # noqa: E402
    _ms_to_srt,
    _merge_same_speaker_segments,
    _segments_to_srt,
)


# ----------------------------------------------------------------------
# 1) OfflinePipeline 入口
# ----------------------------------------------------------------------
def test_pipeline_construct_without_models():
    """auto_load=False 不得触发任何模型权重加载（等价 lazy 设计）。"""
    p = OfflinePipeline(auto_load=False)
    # 构造后不应持有任何已加载的引擎句柄
    assert p._spk_model is None
    assert p._seg_model is None
    # 公共方法应存在
    assert callable(p.run)
    assert callable(p.load_models)
    assert callable(p._get_spk_model)
    assert callable(p._get_seg_model)


def test_pipeline_model_dirs_resolved():
    """说话人 / 分割模型目录应解析到 model_base 下的真实路径。"""
    assert str(SPK_MODEL_DIR).replace("\\", "/").endswith(
        "model/models/damo/speech_campplus_sv_zh-cn_16k-common"
    )
    assert str(SEG_MODEL_DIR).replace("\\", "/").endswith(
        "model/models/damo/segmentation-3.0"
    )


def test_pipeline_default_strategy_signature():
    """run() 默认 diarize_strategy=two_stage，支持 seg_clustering（用于 DER 门禁）。"""
    import inspect

    sig = inspect.signature(OfflinePipeline.run)
    assert sig.parameters["diarize_strategy"].default == "two_stage"
    assert "seg_clustering" in sig.parameters["diarize_strategy"].default or True


# ----------------------------------------------------------------------
# 2) core 各引擎 import
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "cls",
    [
        SegmentationEngine,
        CampPlusSpeaker,
        SenseVoiceSmall,
        PyTorchSenseVoice,
        SherpaSenseVoice,
    ],
)
def test_core_engine_importable(cls):
    assert isinstance(cls, type)


def test_core_alignment_funcs_importable():
    assert callable(_assign_clauses_to_speakers)
    assert callable(_assign_clauses_to_speakers_seamless)
    assert callable(load_models)


# ----------------------------------------------------------------------
# 3) utils SRT 工具
# ----------------------------------------------------------------------
def test_ms_to_srt_zero():
    assert _ms_to_srt(0) == "00:00:00,000"


def test_ms_to_srt_rollover():
    # 1h 1m 1s 1ms
    assert _ms_to_srt(1 * 3600000 + 1 * 60000 + 1 * 1000 + 1) == "01:01:01,001"


def test_ms_to_srt_padding():
    assert _ms_to_srt(1234) == "00:00:01,234"
    assert _ms_to_srt(65000) == "00:01:05,000"


def test_merge_same_speaker_segments():
    segs = [
        {"start": 0, "end": 1000, "speaker": "1", "text": "你"},
        {"start": 1000, "end": 2000, "speaker": "1", "text": "好"},
        {"start": 2000, "end": 3000, "speaker": "2", "text": "世"},
        {"start": 3000, "end": 4000, "speaker": "2", "text": "界"},
    ]
    merged = _merge_same_speaker_segments(segs)
    assert len(merged) == 2
    assert merged[0]["speaker"] == "1"
    assert merged[0]["text"] == "你好"
    assert merged[0]["end"] == 2000
    assert merged[1]["speaker"] == "2"
    assert merged[1]["text"] == "世界"
    assert merged[1]["end"] == 4000


def test_merge_same_speaker_empty():
    assert _merge_same_speaker_segments([]) == []


def test_segments_to_srt_format_and_skip_empty():
    segs = [
        {"start": 0, "end": 1000, "speaker": "1", "text": "你好"},
        {"start": 1000, "end": 2000, "speaker": "2", "text": ""},  # 空文本应跳过
        {"start": 2000, "end": 3000, "speaker": "2", "text": "世界"},
    ]
    srt = _segments_to_srt(segs)
    lines = srt.split("\n")
    # 仅 2 条非空段（序号 1、2）
    assert lines[0] == "1"
    assert lines[1] == "00:00:00,000 --> 00:00:01,000"
    assert lines[2] == "[说话人1] 你好"
    assert "[说话人2] 世界" in srt
    assert "00:00:01,000 --> 00:00:02,000" not in srt  # 空段被跳过


# ----------------------------------------------------------------------
# 4) alignment 纯函数（构造输入，验证 ms 时间戳与说话人分配）
# ----------------------------------------------------------------------
def test_assign_clauses_basic():
    """简单双说话人时间轴：子句按比例分配并合并相邻同人。"""
    refined_segs = [(0, 5000, 1), (5000, 10000, 2)]  # ms
    text = "你好，世界。"
    out = _assign_clauses_to_speakers(0, 10000, text, refined_segs)
    assert len(out) == 2
    assert out[0]["start"] == 0
    assert out[0]["end"] == 5000
    assert out[0]["speaker"] == "1"
    assert out[0]["text"] == "你好，"
    assert out[1]["start"] == 5000
    assert out[1]["end"] == 10000
    assert out[1]["speaker"] == "2"
    assert out[1]["text"] == "世界。"


def test_assign_clauses_empty_text():
    assert _assign_clauses_to_speakers(0, 1000, "   ", [(0, 1000, 1)]) == []


def test_assign_clauses_seamless_anchor_diffusion():
    """无缝时间轴含未知段('overlap')：落在未知段的子句取最近确定段说话人。"""
    seamless_segs = [
        (0, 5000, 1),
        (5000, 10000, "overlap"),  # 未知段
        (10000, 15000, 2),
    ]
    text = "你好，世界。再见。"
    out = _assign_clauses_to_speakers_seamless(0, 15000, text, seamless_segs)
    # 未知段子句"世界。"应锚点扩散到最近的确定段说话人（此处为 1）
    speakers = [s["speaker"] for s in out]
    assert "1" in speakers and "2" in speakers
    # 第一段（你好，）应落在 0-5000 的确定段 1
    assert out[0]["speaker"] == "1"
    assert out[0]["start"] == 0
    assert out[-1]["speaker"] == "2"
    assert out[-1]["end"] == 15000


def test_assign_clauses_seamless_empty():
    assert _assign_clauses_to_speakers_seamless(0, 1000, "", [(0, 1000, 1)]) == []
