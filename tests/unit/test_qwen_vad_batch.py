"""单元测试与集成测试：Qwen3-ASR 切句与 VAD 批量转写。

测试策略：
- 单元测试：用 mock 隔离 Qwen 后端，验证 _split_timestamps_to_segments 和 _build_srt
- 集成测试：用真实 Qwen 后端 + 真实音频，验证 VAD 批量路径的最终输出质量
"""

import os
import tempfile
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from funclip_pro.core.asr import QwenEngine, _split_timestamps_to_segments
from funclip_pro.pipeline.offline import OfflinePipeline


# ========= 单元测试：_split_timestamps_to_segments =========

class TestSplitTimestamps:
    """_split_timestamps_to_segments 纯函数测试——无需后端，纯内存计算。"""

    def test_punctuation_split(self):
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
        assert len(segs) == 2
        assert "今天天气真好" in segs[0]["text"]
        assert "我们去公园吧" in segs[1]["text"]

    def test_gap_split(self):
        """VAD 已切好，gap > 0.8s 不触发切分。"""
        ts = [
            {"text": "第一句", "start": 0.0, "end": 0.8},
            {"text": "第二句", "start": 2.0, "end": 2.8},
        ]
        full_text = "第一句第二句"
        segs = _split_timestamps_to_segments(ts, full_text)
        assert len(segs) == 1
        assert segs[0]["text"] == "第一句第二句"

    def test_single_segment_no_split(self):
        """连续流畅语音应合并为一句"""
        ts = [
            {"text": "今天", "start": 0.0, "end": 0.3},
            {"text": "天气", "start": 0.35, "end": 0.6},
            {"text": "真好", "start": 0.65, "end": 0.9},
        ]
        full_text = "今天天气真好"
        segs = _split_timestamps_to_segments(ts, full_text)
        assert len(segs) == 1
        assert segs[0]["text"] == "今天天气真好"

    def test_long_segment_cut_half(self):
        """超 5s 无标点段 → 按 token 数切半"""
        ts = [{"text": f"字", "start": i * 0.2, "end": i * 0.2 + 0.15} for i in range(30)]
        full_text = "字" * 30
        segs = _split_timestamps_to_segments(ts, full_text)
        assert len(segs) >= 2

    def test_empty_input(self):
        assert _split_timestamps_to_segments([], "") == []
        assert _split_timestamps_to_segments([], "abc") == []
        assert _split_timestamps_to_segments([{"text": "a", "start": 0, "end": 1}], "") == []

    def test_offset_ms(self):
        """offset_ms 累加到时间戳上"""
        ts = [{"text": "你好", "start": 0.0, "end": 0.5}]
        segs = _split_timestamps_to_segments(ts, "你好", offset_ms=5000)
        assert segs[0]["start"] == 5000
        assert segs[0]["end"] == 5500

    def test_hard_punc_flush(self):
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

    def test_char_ts_preserved(self):
        """_char_ts 被保留供 ASS karaoke 使用"""
        ts = [{"text": "你好", "start": 0.0, "end": 0.3}, {"text": "世界", "start": 0.3, "end": 0.6}]
        segs = _split_timestamps_to_segments(ts, "你好世界")
        assert "_char_ts" in segs[0]
        assert len(segs[0]["_char_ts"]) == 2


# ========= 单元测试：QwenEngine._build_srt =========

class TestBuildSrt:
    """验证 _build_srt 使用智能分句后输出句级 SRT。"""

    def test_build_srt_smart_split(self):
        engine = QwenEngine()
        ts = [
            {"text": "今天", "start": 0.1, "end": 0.3},
            {"text": "天气", "start": 0.3, "end": 0.6},
            {"text": "真", "start": 0.6, "end": 0.8},
            {"text": "好", "start": 0.8, "end": 1.0},
            {"text": "。", "start": 1.0, "end": 1.1},
            {"text": "我们", "start": 1.2, "end": 1.4},
            {"text": "去", "start": 1.4, "end": 1.6},
            {"text": "玩", "start": 1.6, "end": 1.8},
        ]
        full_text = "今天天气真好。我们去玩"
        srt = engine._build_srt(ts, full_text)
        lines = srt.strip().split("\n")
        block_count = sum(1 for l in lines if l.strip().isdigit())
        assert block_count == 2, f"应输出2条SRT字幕，实际: {block_count}"
        # 验证第一条时间轴
        assert "00:00:00,100" in srt
        assert "00:00:01,100" in srt
        assert "今天天气真好" in srt

    def test_build_srt_fallback_no_timestamps(self):
        """无时间戳时输出整段 fallback"""
        engine = QwenEngine()
        srt = engine._build_srt([], "今天天气真好")
        assert "00:00:00,000" in srt
        assert "今天天气真好" in srt


# ========= 单元测试：QwenEngine.transcribe_batch（mock 后端）========

@pytest.fixture
def mock_requests_post():
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "text": "测试一",
                    "timestamps": [
                        {"text": "测", "start": 0.1, "end": 0.2},
                        {"text": "试", "start": 0.2, "end": 0.3},
                        {"text": "一", "start": 0.3, "end": 0.4},
                    ]
                },
                {
                    "text": "测试二",
                    "timestamps": [
                        {"text": "测", "start": 0.1, "end": 0.2},
                        {"text": "试", "start": 0.2, "end": 0.3},
                        {"text": "二", "start": 0.3, "end": 0.4},
                    ]
                }
            ]
        }
        mock_post.return_value = mock_response
        yield mock_post


class TestQwenEngineBatch:
    """测试 QwenEngine 批量请求的编解码（mock 后端）。"""

    def test_batch_sends_correct_payload(self, mock_requests_post):
        engine = QwenEngine()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f1, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f2:
            path1, path2 = f1.name, f2.name
            f1.write(b"fake data 1")
            f2.write(b"fake data 2")

        try:
            engine.transcribe_batch([path1, path2], language="zh")
            _, kwargs = mock_requests_post.call_args
            payload = kwargs["json"]
            assert "audio_batch_base64" in payload
            assert payload["language"] == "Chinese"
        finally:
            for p in [path1, path2]:
                if os.path.exists(p): os.remove(p)

    def test_batch_numpy_path(self, mock_requests_post):
        engine = QwenEngine()
        chunk1 = np.zeros(16000, dtype=np.float32)
        chunk2 = np.ones(16000, dtype=np.float32)
        results = engine.transcribe_batch([chunk1, chunk2], language="zh")
        assert len(results) == 2
        assert results[0]["text"] == "测试一"


# ========= 单元测试：OfflinePipeline Qwen VAD 分支 =========

@pytest.mark.slow
class TestOfflinePipelineQwenVAD:
    """测试 OfflinePipeline 中 Qwen 引擎的 VAD 分割 + 分句（mock 后端）。"""

    @patch("funclip_pro.pipeline.offline.resolve_model_path", return_value="fake")
    @patch("funclip_pro.pipeline.offline.load_models")
    @patch("librosa.load")
    @patch("funclip_pro.core.asr._select_engine", return_value="qwen")
    @patch("funclip_pro.pipeline.offline.OfflinePipeline._get_spk_model")
    @patch("funclip_pro.pipeline.offline.OfflinePipeline._get_seg_model")
    def test_vad_branch_split_and_align(
        self, mock_seg, mock_spk, mock_select, mock_load_audio,
        mock_load_models, mock_resolve
    ):
        """VAD 分支：VAD 切分 → 批量发送 → 时间轴对齐 → 分句合并"""
        mock_load_audio.return_value = (np.zeros(160000), 16000)
        pipeline = OfflinePipeline(auto_load=False)

        mock_vad = MagicMock()
        mock_vad.generate.return_value = [{"value": [[0, 4000], [6000, 10000]]}]

        mock_qwen = MagicMock()
        mock_qwen.transcribe_batch.return_value = [
            {
                "text": "完成测试。继续运行",
                "timestamps": [
                    {"text": "完成", "start": 0.1, "end": 0.4},
                    {"text": "测试", "start": 0.4, "end": 0.7},
                    {"text": "。", "start": 0.7, "end": 0.8},
                    {"text": "继续", "start": 1.0, "end": 1.3},
                    {"text": "运行", "start": 1.3, "end": 1.6},
                ]
            },
            {
                "text": "第二段内容。结束",
                "timestamps": [
                    {"text": "第二段", "start": 0.1, "end": 0.5},
                    {"text": "内容", "start": 0.5, "end": 0.8},
                    {"text": "。", "start": 0.8, "end": 0.9},
                    {"text": "结束", "start": 1.0, "end": 1.3},
                ]
            }
        ]

        with patch("funclip_pro.pipeline.offline.asr_mod") as mock_asr_mod, \
             patch("funclip_pro.core.asr.QwenEngine", return_value=mock_qwen):
            mock_asr_mod._select_engine.return_value = "qwen"
            mock_asr_mod.VAD_MODEL = mock_vad
            mock_asr_mod._use_vad.return_value = True
            mock_asr_mod._merge_vad_segments.return_value = [(0, 4000), (6000, 10000)]

            raw_text, engine, segments, diarized = pipeline.run(
                "fake.wav", vad_strategy="always", engine="qwen", language=["zh"]
            )

            # 验证 VAD 被调用
            assert mock_vad.generate.called

            # 验证批量发送（2个numpy切片）
            call_args, _ = mock_qwen.transcribe_batch.call_args
            assert len(call_args[0]) == 2

            # 验证分句合并：第一段"完成测试。继续运行" → 2句
            # 第二段"第二段内容。结束" → 2句，共 4 句级 segment
            assert len(segments) == 4, f"应合并为4句，实际: {len(segments)}"

            # 验证时间轴对齐
            # 第一段偏移 0ms
            assert segments[0]["start"] == 100
            assert segments[0]["end"] == 800
            assert "完成测试。" in segments[0]["text"]

            # 第二句从第一段的第二半
            assert segments[1]["start"] == 1000
            assert segments[1]["end"] == 1600
            assert "继续运行" in segments[1]["text"]

            # 第二段偏移 6000ms
            assert segments[2]["start"] == 6100
            assert segments[2]["end"] == 6900
            assert "第二段内容。" in segments[2]["text"]

            assert segments[3]["start"] == 7000
            assert segments[3]["end"] == 7300
            assert "结束" in segments[3]["text"]

    @patch("funclip_pro.pipeline.offline.resolve_model_path", return_value="fake")
    @patch("funclip_pro.pipeline.offline.load_models")
    @patch("librosa.load")
    @patch("funclip_pro.core.asr._select_engine", return_value="qwen")
    @patch("funclip_pro.pipeline.offline.OfflinePipeline._get_spk_model")
    @patch("funclip_pro.pipeline.offline.OfflinePipeline._get_seg_model")
    def test_vad_branch_no_split_no_punc(
        self, mock_seg, mock_spk, mock_select, mock_load_audio,
        mock_load_models, mock_resolve
    ):
        """VAD 分支：连续无标点语音应合并为一句不切碎"""
        mock_load_audio.return_value = (np.zeros(160000), 16000)
        pipeline = OfflinePipeline(auto_load=False)

        mock_vad = MagicMock()
        mock_vad.generate.return_value = [{"value": [[0, 5000]]}]

        mock_qwen = MagicMock()
        mock_qwen.transcribe_batch.return_value = [
            {
                "text": "今天天气真好",
                "timestamps": [
                    {"text": "今天", "start": 0.0, "end": 0.3},
                    {"text": "天气", "start": 0.35, "end": 0.6},
                    {"text": "真好", "start": 0.65, "end": 0.9},
                ]
            },
        ]

        with patch("funclip_pro.pipeline.offline.asr_mod") as mock_asr_mod, \
             patch("funclip_pro.core.asr.QwenEngine", return_value=mock_qwen):
            mock_asr_mod._select_engine.return_value = "qwen"
            mock_asr_mod.VAD_MODEL = mock_vad
            mock_asr_mod._use_vad.return_value = True
            mock_asr_mod._merge_vad_segments.return_value = [(0, 5000)]

            _, _, segments, _ = pipeline.run(
                "fake.wav", vad_strategy="always", engine="qwen", language=["zh"]
            )

            # 连续无标点应保留为 1 句
            assert len(segments) == 1
            assert segments[0]["text"] == "今天天气真好"


@pytest.mark.slow
class TestOfflinePipelineQwenNoVAD:
    """测试 OfflinePipeline 中 Qwen 引擎的非 VAD 路径（直接 transcribe + 分句）。"""

    @patch("funclip_pro.pipeline.offline.resolve_model_path", return_value="fake")
    @patch("funclip_pro.pipeline.offline.load_models")
    @patch("librosa.load")
    @patch("funclip_pro.core.asr._select_engine", return_value="qwen")
    def test_no_vad_path_smart_segmentation(
        self, mock_select, mock_load_audio,
        mock_load_models, mock_resolve
    ):
        """非 VAD 路径：直接 transcribe，结果应经 _split_timestamps_to_segments 聚合成句"""
        mock_load_audio.return_value = (np.zeros(160000), 16000)
        pipeline = OfflinePipeline(auto_load=False)

        mock_qwen = MagicMock()
        mock_qwen.transcribe.return_value = {
            "text": "今天天气真好。我们一起去公园吧",
            "raw": {
                "timestamps": [
                    {"text": "今天", "start": 0.0, "end": 0.3},
                    {"text": "天气", "start": 0.3, "end": 0.6},
                    {"text": "真好", "start": 0.6, "end": 0.9},
                    {"text": "。", "start": 0.9, "end": 1.0},
                    {"text": "我们", "start": 1.2, "end": 1.5},
                    {"text": "一起", "start": 1.5, "end": 1.8},
                    {"text": "去", "start": 1.8, "end": 2.0},
                    {"text": "公园", "start": 2.0, "end": 2.3},
                    {"text": "吧", "start": 2.3, "end": 2.5},
                ]
            }
        }

        with patch("funclip_pro.pipeline.offline.asr_mod") as mock_asr_mod, \
             patch("funclip_pro.core.asr.QwenEngine", return_value=mock_qwen):
            mock_asr_mod._select_engine.return_value = "qwen"
            mock_asr_mod._use_vad.return_value = False

            raw_text, engine, segments, diarized = pipeline.run(
                "fake.wav", vad_strategy="never", engine="qwen", language=["zh"]
            )

            # 句号应切成 2 个句级 segment
            assert len(segments) == 2, f"应切为2句，实际: {len(segments)}"
            assert "今天天气真好" in segments[0]["text"]
            assert "我们一起去公园吧" in segments[1]["text"]


# ========= 集成测试（需要真实 Qwen Docker 后端）========

@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("TEST_QWEN_BACKEND"),
    reason="设置 TEST_QWEN_BACKEND=1 以启用真实后端集成测试"
)
class TestQwenBackendIntegration:
    """使用真实 Qwen3 Docker 后端的集成测试。
    
    运行方式：
        TEST_QWEN_BACKEND=1 python -m pytest tests/unit/test_qwen_vad_batch.py::TestQwenBackendIntegration -v
    """

    AUDIO_ZHS = r"E:\FunClip\FunClip\model\models\FunAudioLLM\Fun-ASR-Nano-2512\example\zh.mp3"
    AUDIO_EN = r"E:\FunClip\FunClip\model\models\FunAudioLLM\Fun-ASR-Nano-2512\example\en.mp3"
    AUDIO_WAV = r"E:\FunClip\FunClip\test_audio.wav"

    def _check_srt_quality(self, srt: str, min_blocks: int = 1):
        lines = srt.strip().split("\n")
        blocks = [l for l in lines if l.strip().isdigit()]
        assert len(blocks) >= min_blocks, f"SRT block count: {len(blocks)}"
        for line in lines:
            if "-->" in line:
                start, end = line.split(" --> ")
                # 每段至少 200ms，防止单字碎片
                s_parts = start.replace(",", ":").split(":")
                e_parts = end.replace(",", ":").split(":")
                s_ms = int(s_parts[0])*3600000 + int(s_parts[1])*60000 + int(s_parts[2])*1000 + int(s_parts[3])
                e_ms = int(e_parts[0])*3600000 + int(e_parts[1])*60000 + int(e_parts[2])*1000 + int(e_parts[3])
                dur = e_ms - s_ms
                assert dur >= 200, f"段时长 {dur}ms < 200ms，有单字碎片: {line}"

    def test_zh_mp3(self):
        engine = QwenEngine()
        result = engine.transcribe(self.AUDIO_ZHS, language="zh")
        self._check_srt_quality(result["srt"], min_blocks=1)
        print(f"\n[集成测试] zh.mp3 OK\n{result['srt']}")

    def test_en_mp3(self):
        engine = QwenEngine()
        result = engine.transcribe(self.AUDIO_EN, language="en")
        self._check_srt_quality(result["srt"], min_blocks=1)
        print(f"\n[集成测试] en.mp3 OK\n{result['srt']}")

    def test_audio_wav(self):
        engine = QwenEngine()
        result = engine.transcribe(self.AUDIO_WAV, language="zh")
        self._check_srt_quality(result["srt"], min_blocks=1)
        print(f"\n[集成测试] test_audio.wav OK\n{result['srt']}")
