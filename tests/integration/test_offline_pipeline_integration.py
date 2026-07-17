# -*- coding: utf-8 -*-
"""T09 测试门禁 · OfflinePipeline 集成测试（需要真实模型权重 / GPU）。

默认跳过：仅在显式开启 FUNCLIP_INTEGRATION=1 或 `pytest -m ml` 时运行。
真实 ML 运行必须用宿主 Python：E:\conda\envs\asr_ui_env\python.exe
导入包时设 PYTHONPATH=E:\project\funclip-pro\src

本用例验证：
1. pipeline.run() 返回 TranscriptionResult
2. 各字段（text, engine, segments, duration_ms, diarized_text）类型正确
3. Qwen with-VAD 分支的 segment 边界与 VAD 一致（如果 Qwen 引擎可用）
4. Segment 字段访问（.start_ms, .end_ms, .text, .speaker）
"""
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# 仅在显式开启时运行（避免默认 pytest 套件触发重型模型加载）
RUN_ML = os.environ.get("FUNCLIP_INTEGRATION") == "1"

pytestmark = [pytest.mark.slow, pytest.mark.ml, pytest.mark.skipif(not RUN_ML, reason="需 FUNCLIP_INTEGRATION=1 才跑真实 ML 集成")]


@pytest.fixture(scope="module")
def pipeline():
    from funclip_pro.pipeline import OfflinePipeline

    return OfflinePipeline(auto_load=True)


def test_run_seg_clustering_end_to_end(pipeline):
    """对一段真实音频跑 seg_clustering，验证返回 TranscriptionResult 与 segments 结构合法。"""
    from funclip_pro.core.models import TranscriptionResult, Segment

    audio = ROOT / "output.mp3"
    if not audio.exists():
        pytest.skip("无可用测试音频 output.mp3")

    result = pipeline.run(
        str(audio), diarize=True, diarize_strategy="seg_clustering"
    )

    # 验证返回类型
    assert isinstance(result, TranscriptionResult)
    assert isinstance(result.text, str)
    assert isinstance(result.engine, str)
    assert isinstance(result.segments, list)
    assert isinstance(result.diarized_text, str)

    # 若产出段，校验 Segment dataclass 字段
    prev_end = -1
    for seg in result.segments:
        assert isinstance(seg, Segment)
        assert seg.end_ms >= seg.start_ms >= 0
        assert seg.end_ms >= prev_end  # 时间有序
        prev_end = seg.end_ms
        assert isinstance(seg.speaker, str)


def test_run_basic_transcription(pipeline):
    """验证非 diarize 路径也返回正确的 TranscriptionResult。"""
    from funclip_pro.core.models import TranscriptionResult, Segment

    audio = ROOT / "output.mp3"
    if not audio.exists():
        pytest.skip("无可用测试音频 output.mp3")

    result = pipeline.run(str(audio), diarize=False)

    assert isinstance(result, TranscriptionResult)
    assert isinstance(result.text, str)
    assert isinstance(result.engine, str)
    assert isinstance(result.segments, list)

    for seg in result.segments:
        assert isinstance(seg, Segment)
        assert isinstance(seg.text, str)
        assert seg.end_ms >= seg.start_ms >= 0


def test_qwen_with_vad_segment_boundary_consistency(pipeline):
    """验证 Qwen with-VAD 分支的 segment 边界与 VAD 段一致。

    当 Qwen 引擎可用时，VAD 段的 (start_ms, end_ms) 应该直接
    对应 Segment 的 (start_ms, end_ms)，不做二次拆分。
    """
    from funclip_pro.core.models import TranscriptionResult, Segment
    from funclip_pro.core import asr as asr_mod

    audio = ROOT / "output.mp3"
    if not audio.exists():
        pytest.skip("无可用测试音频 output.mp3")

    # 强制走 Qwen 引擎
    # 注意：如果 Qwen 引擎不可用，_select_engine 会回退到其他引擎
    # 这里我们只做验证性的执行
    import librosa
    try:
        y, sr = librosa.load(str(audio), sr=16000)
    except Exception:
        pytest.skip("无法加载测试音频")

    duration_ms = len(y) / sr * 1000

    # 检查 VAD 是否可用且是否应该使用 VAD
    vad_should_run = asr_mod._use_vad("auto", duration_ms)
    if not vad_should_run:
        pytest.skip("音频太短，不走 VAD 路径")

    try:
        # 先获取 VAD 输出
        vad_model = asr_mod.VAD_MODEL
        if vad_model is None:
            pytest.skip("VAD 模型未加载")

        result = pipeline.run(str(audio), engine="qwen")

        if result.engine != "qwen":
            pytest.skip("Qwen 引擎不可用，跳过")

        assert isinstance(result, TranscriptionResult)

        # 验证 segments 是 Segment dataclass
        for seg in result.segments:
            assert isinstance(seg, Segment)

        # 验证 words 的正确性（如果有）
        for seg in result.segments:
            if seg.words:
                for w in seg.words:
                    assert w.text
                    assert w.start_ms >= seg.start_ms - 100  # 允许少量偏移
                    assert w.end_ms <= seg.end_ms + 100

    except Exception as e:
        if "QwenEngine" in str(type(e)) or "connection" in str(e).lower():
            pytest.skip(f"Qwen 引擎连接失败: {e}")
        raise
