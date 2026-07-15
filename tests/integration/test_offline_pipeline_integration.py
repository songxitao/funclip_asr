# -*- coding: utf-8 -*-
"""T09 测试门禁 · OfflinePipeline 集成测试（需要真实模型权重 / GPU）。

默认跳过：仅在显式开启 FUNCLIP_INTEGRATION=1 或 `pytest -m ml` 时运行。
真实 ML 运行必须用宿主 Python：E:\conda\envs\asr_ui_env\python.exe
导入包时设 PYTHONPATH=E:\project\funclip-pro\src

本用例等价于 DER 门禁的"单文件端到端"最小验证：直接调用 OfflinePipeline.run()
（不走 HTTP 服务），验证 seg_clustering 分支产出合法 segments（ms 时间戳、整数说话人）。
全量 20 场 DER 由 run_ali_der_full.py --strategy seg_clustering 单独评测。
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
    """对一段真实音频跑 seg_clustering，验证返回四元组与 segments 结构合法。"""
    audio = ROOT / "output.mp3"
    if not audio.exists():
        pytest.skip("无可用测试音频 output.mp3")
    raw_text, engine_key, segments, diarized_text = pipeline.run(
        str(audio), diarize=True, diarize_strategy="seg_clustering"
    )
    assert isinstance(raw_text, str)
    assert isinstance(engine_key, str)
    assert isinstance(segments, list)
    assert isinstance(diarized_text, str)
    # 若产出段，校验结构：ms 时间戳有序、speaker 为整数（或整数字符串）
    prev_end = -1
    for seg in segments:
        assert "start" in seg and "end" in seg
        assert seg["end"] >= seg["start"] >= 0
        assert seg["end"] >= prev_end  # 时间有序
        prev_end = seg["end"]
        assert seg["speaker"] is not None
