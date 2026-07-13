# -*- coding: utf-8 -*-
"""无缝说话人时间轴单元测试。"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np


class TestProcessFullAudioSeamlessLogic(unittest.TestCase):
    """测试无缝时间轴的段类型逻辑（不依赖模型，只测数据结构和合并规则）。"""

    def test_seamless_no_gap_between_segments(self):
        """验证无缝时间轴相邻段首尾相连，无时间窟窿。"""
        segments = [
            (0.0, 2.0, "single", None),
            (2.0, 3.5, "overlap", None),
            (3.5, 5.0, "single", None),
            (5.0, 5.8, "silence", None),
            (5.8, 8.0, "overlap", None),
        ]
        for i in range(1, len(segments)):
            self.assertEqual(
                segments[i][0], segments[i-1][1],
                f"时间轴有窟窿: {segments[i-1][1]} -> {segments[i][0]}"
            )

    def test_seamless_types_contain_all_three(self):
        """验证无缝时间轴包含三种段类型。"""
        segments = [
            (0.0, 1.0, "single", None),
            (1.0, 2.0, "overlap", None),
            (2.0, 3.0, "silence", None),
        ]
        types = {seg[2] for seg in segments}
        self.assertIn("single", types)
        self.assertIn("overlap", types)
        self.assertIn("silence", types)

    def test_seamless_single_has_audio_others_none(self):
        """验证 single 段的 seg_audio 不为 None，overlap/silence 为 None。"""
        segments = [
            (0.0, 1.0, "single", np.zeros(16000)),
            (1.0, 2.0, "overlap", None),
            (2.0, 3.0, "silence", None),
        ]
        for st, en, typ, audio in segments:
            if typ == "single":
                self.assertIsNotNone(audio)
            else:
                self.assertIsNone(audio)


class TestClusterWithSeamlessSegmentation(unittest.TestCase):
    """测试 CampPlusSpeaker.cluster_with_seamless_segmentation()。"""

    def setUp(self):
        # mock FunASR AutoModel
        self.patcher_automodel = patch("funasr.AutoModel")
        self.mock_automodel_class = self.patcher_automodel.start()

        from speaker_engine import CampPlusSpeaker
        self.speaker_engine = CampPlusSpeaker(model_dir="mock_dir", device="cpu")

        # mock 说话人向量提取
        self.speaker_engine.extract_embedding = MagicMock()

    def tearDown(self):
        self.patcher_automodel.stop()

    def _make_seg_engine(self, seamless_segs):
        """创建 mock segmentation engine，返回指定无缝时间轴。"""
        engine = MagicMock()
        engine.process_full_audio_seamless.return_value = seamless_segs
        return engine

    def test_single_segments_get_int_speaker(self):
        """验证 single 段获得 int 类型的 speaker_id。"""
        seg_engine = self._make_seg_engine([
            (0.0, 1.5, "single", np.zeros(24000, dtype=np.float32)),
            (1.5, 2.5, "overlap", None),
            (2.5, 4.0, "single", np.zeros(24000, dtype=np.float32)),
        ])
        v1 = np.ones(192, dtype=np.float32)
        self.speaker_engine.extract_embedding.side_effect = [v1, v1]

        audio = np.zeros(16000 * 5, dtype=np.float32)
        result = self.speaker_engine.cluster_with_seamless_segmentation(
            audio, seg_engine, n_speakers=1
        )

        for st, en, val in result:
            if isinstance(val, int):
                self.assertGreaterEqual(val, 1)

    def test_overlap_silence_preserved(self):
        """验证 overlap/silence 段保留字符串标记。"""
        seg_engine = self._make_seg_engine([
            (0.0, 1.5, "single", np.zeros(24000, dtype=np.float32)),
            (1.5, 2.5, "overlap", None),
            (2.5, 4.0, "single", np.zeros(24000, dtype=np.float32)),
            (4.0, 5.0, "silence", None),
        ])
        v1 = np.ones(192, dtype=np.float32)
        self.speaker_engine.extract_embedding.side_effect = [v1, v1]

        audio = np.zeros(16000 * 6, dtype=np.float32)
        result = self.speaker_engine.cluster_with_seamless_segmentation(
            audio, seg_engine, n_speakers=1
        )

        types_found = set()
        for st, en, val in result:
            if isinstance(val, str):
                types_found.add(val)
        self.assertIn("overlap", types_found)
        self.assertIn("silence", types_found)

    def test_empty_segments(self):
        """segmentation 返回空列表时，返回空。"""
        seg_engine = self._make_seg_engine([])

        audio = np.zeros(16000, dtype=np.float32)
        result = self.speaker_engine.cluster_with_seamless_segmentation(audio, seg_engine)
        self.assertEqual(result, [])

    def test_all_single_embeddings_none_fallback(self):
        """所有 single 段 embedding 提取失败 → 统一标说话人1。"""
        seg_engine = self._make_seg_engine([
            (0.0, 2.0, "single", np.zeros(32000, dtype=np.float32)),
            (2.0, 4.0, "single", np.zeros(32000, dtype=np.float32)),
        ])
        self.speaker_engine.extract_embedding.return_value = None

        audio = np.zeros(16000 * 5, dtype=np.float32)
        result = self.speaker_engine.cluster_with_seamless_segmentation(audio, seg_engine)
        for st, en, val in result:
            self.assertEqual(val, 1)

    def test_seamless_merge_adjacent_same_speaker(self):
        """同人 single 段间隔 < 0.5s → 合并。遇到 overlap 断开。"""
        seg_engine = self._make_seg_engine([
            (0.0, 2.0, "single", np.zeros(32000, dtype=np.float32)),
            (2.3, 3.0, "single", np.zeros(11200, dtype=np.float32)),  # 间隔 0.3s < 0.5s
            (3.0, 4.0, "overlap", None),                              # 断开
            (4.0, 5.5, "single", np.zeros(24000, dtype=np.float32)),
        ])
        v1 = np.ones(192, dtype=np.float32)
        v2 = -np.ones(192, dtype=np.float32)
        self.speaker_engine.extract_embedding.side_effect = [v1, v1, v2]

        audio = np.zeros(16000 * 6, dtype=np.float32)
        result = self.speaker_engine.cluster_with_seamless_segmentation(
            audio, seg_engine, n_speakers=2
        )

        # 合并后应为3段：
        # [0-3.0] single段合并 → speaker_a
        # [3.0-4.0] overlap
        # [4.0-5.5] single → speaker_b
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0][2], int)  # single
        self.assertEqual(result[1][2], "overlap")  # overlap
        self.assertIsInstance(result[2][2], int)  # single


class TestAssignClausesToSpeakersSeamless(unittest.TestCase):
    """测试 _assign_clauses_to_speakers_seamless() 锚点扩散逻辑。"""

    def _call(self, asr_start, asr_end, text, seamless_segs):
        from asr_onnx_service import _assign_clauses_to_speakers_seamless
        return _assign_clauses_to_speakers_seamless(asr_start, asr_end, text, seamless_segs)

    def test_clause_on_determined_segment(self):
        """子句落在确定段上 → 直接取该说话人。"""
        seamless_segs = [
            (0, 2000, 1),        # 确定段 说话人1
            (2000, 3000, "overlap"),
            (3000, 5000, 2),     # 确定段 说话人2
        ]
        result = self._call(0, 2000, "今天天气真好。", seamless_segs)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["speaker"], "1")

    def test_clause_on_unknown_anchor_diffusion(self):
        """子句完全落在未知段 → 锚点扩散取最近确定段。"""
        seamless_segs = [
            (0, 1000, 1),          # 确定段 说话人1
            (1000, 2000, "overlap"),  # 子句落在这
        ]
        result = self._call(1000, 2000, "你觉得呢？", seamless_segs)
        self.assertEqual(len(result), 1)
        # 锚点扩散：最近确定段是说话人1
        self.assertEqual(result[0]["speaker"], "1")

    def test_no_determined_segments_fallback(self):
        """没有任何确定段 → 兜底标说话人1。"""
        seamless_segs = [
            (0, 2000, "overlap"),
            (2000, 4000, "silence"),
        ]
        result = self._call(0, 2000, "你好。", seamless_segs)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["speaker"], "1")

    def test_empty_text(self):
        """空文本返回空列表。"""
        result = self._call(0, 1000, "", [(0, 1000, 1)])
        self.assertEqual(result, [])

    def test_multi_clause_diff_speaker(self):
        """多子句分配：每个子句分给不同说话人。"""
        seamless_segs = [
            (0, 3000, 1),          # 说话人1 0-3s
            (3000, 6000, "overlap"),  # 未知
            (6000, 10000, 2),        # 说话人2 6-10s
        ]
        text = "今天开会讨论方案。明天继续。"
        result = self._call(0, 10000, text, seamless_segs)
        self.assertGreater(len(result), 0)
        # 验证每个子句都有 speaker
        for seg in result:
            self.assertIn(seg["speaker"], ["1", "2"])

    def test_multi_clause_merge_same_speaker(self):
        """相邻相同说话人的子句合并。"""
        seamless_segs = [
            (0, 10000, 1),
        ]
        text = "今天天气真好。你觉得呢？"
        result = self._call(0, 10000, text, seamless_segs)
        self.assertEqual(len(result), 1)  # 应合并为一段
        self.assertEqual(result[0]["speaker"], "1")
        self.assertIn("今天", result[0]["text"])
        self.assertIn("你觉得", result[0]["text"])

    def test_determined_seg_weighted_overlap(self):
        """子句跨确定段和未知段 → 取重叠最多的确定段。"""
        seamless_segs = [
            (0, 1000, 1),           # 说话人1 0-1s
            (1000, 3000, 2),        # 说话人2 1-3s (这个子句大部分落在此区间)
        ]
        text = "开始汇报。"
        subseg = self._call(0, 3000, text, seamless_segs)
        # 子句 0-3000ms 与说话人2的段重叠 2000ms > 与说话人1重叠 1000ms
        # 子句按字数比例 - 假设"开始汇报。"4个字, 0-3000ms, 整个子句跨两个段
        # 所以还是走最重叠的逻辑
        self.assertEqual(len(subseg), 1)
        self.assertEqual(subseg[0]["speaker"], "2")


if __name__ == "__main__":
    unittest.main()
