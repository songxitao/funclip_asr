# -*- coding: utf-8 -*-
"""集成测试：diarize_strategy=sliding 时，服务返回带 speaker 的 segments。

不加载真实模型：mock VAD（提供一段）、ASR 解码（返回占位文本）、
Cam++（cluster_sliding 直接返回固定段）。直接驱动 _run_inference 的
diarize 分支，验证 sliding 路径走通并产出带 speaker 的 segments。
"""
import os
import wave
import array
import tempfile

import numpy as np
from unittest.mock import patch


def test_run_inference_sliding_returns_speaker_segments():
    import asr_onnx_service as svc

    # 假说话人引擎：cluster_sliding 直接返回固定段，extract_embedding 返回零向量兜底
    class FakeSpk:
        def extract_embedding(self, samp):
            return np.zeros(192, dtype=np.float32)

        def cluster_sliding(self, audio, sr=16000, **kw):
            return [(0.0, 1.5, 1), (1.5, 3.0, 2)]

    # 假 VAD：返回一段覆盖整段音频，使 chunks 非空（ASR 仍需 VAD 切分）
    class FakeVAD:
        def generate(self, **kw):
            return [{"value": [[0, 3000]]}]

    tmp_path = os.path.join(tempfile.gettempdir(), "sliding_integration_test.wav")
    with wave.open(tmp_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(array.array("h", [0] * 48000))  # 3s 静音
    try:
        with patch.object(svc, "_get_spk_model", return_value=FakeSpk()), \
             patch.object(svc, "_decode", return_value=["测试文本"]), \
             patch.object(svc, "VAD_MODEL", FakeVAD()):
            _, _, segments, _ = svc._run_inference(
                tmp_path, vad_strategy="never", diarize=True,
                diarize_strategy="sliding", num_speakers=2,
            )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    assert len(segments) >= 1
    assert all("speaker" in s for s in segments)
    # sliding 分支应来自 cluster_sliding 的固定返回（2 段，speaker 1 与 2）
    assert len(segments) == 2
    assert segments[0]["speaker"] == "1"
    assert segments[1]["speaker"] == "2"


def test_run_inference_vad_sliding_returns_speaker_text_segments():
    import asr_onnx_service as svc
    # FakeSpk，其中 cluster 返回聚类字典
    class FakeSpk:
        def cluster(self, chunks, strategy, **kw):
            return {0: 1} # 对应第 0 个 VAD 段属于 speaker 1
    class FakeVAD:
        def generate(self, **kw):
            return [{"value": [[0, 3000]]}]

    tmp_path = os.path.join(tempfile.gettempdir(), "vad_sliding_integration_test.wav")
    with wave.open(tmp_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(array.array("h", [0] * 48000))
    try:
        with patch.object(svc, "_get_spk_model", return_value=FakeSpk()), \
             patch.object(svc, "_decode", return_value=["测试文本"]), \
             patch.object(svc, "VAD_MODEL", FakeVAD()):
            _, _, segments, diarized_text = svc._run_inference(
                tmp_path, vad_strategy="never", diarize=True,
                diarize_strategy="vad_sliding", num_speakers=2,
            )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    assert len(segments) == 1
    assert segments[0]["speaker"] == "1"
    assert segments[0]["text"] == "测试文本"
    assert "[说话人1] 测试文本" in diarized_text

