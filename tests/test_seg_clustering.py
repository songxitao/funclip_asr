# -*- coding: utf-8 -*-
"""CampPlusSpeaker 的 cluster_with_segmentation 方法的单元测试。"""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np


class TestSegClustering(unittest.TestCase):

    def setUp(self):
        # mock FunASR 的 AutoModel，防止 __init__ 真正加载庞大的模型
        self.patcher_automodel = patch("funasr.AutoModel")
        self.mock_automodel_class = self.patcher_automodel.start()

        # 实例化 CampPlusSpeaker
        from funclip_pro.core.speaker import CampPlusSpeaker
        self.speaker_engine = CampPlusSpeaker(model_dir="mock_dir", device="cpu")

        # mock 说话人向量提取
        self.speaker_engine.extract_embedding = MagicMock()

    def tearDown(self):
        self.patcher_automodel.stop()

    def test_basic_two_speakers(self):
        """测试基础的双说话人聚类且不合并（间隔 > 0.5s）。"""
        mock_segment_engine = MagicMock()
        mock_segment_engine.process_full_audio.return_value = [
            (0.0, 2.0, np.zeros(16000)),
            (3.0, 5.0, np.zeros(16000)),
            (6.0, 8.0, np.zeros(16000)),
        ]

        v1 = np.ones(192, dtype=np.float32)
        v2 = -np.ones(192, dtype=np.float32)
        # 1 和 3 同一个人，2 是另一个人
        self.speaker_engine.extract_embedding.side_effect = [v1, v2, v1]

        result = self.speaker_engine.cluster_with_segmentation(
            audio_16k=np.zeros(128000),
            segment_engine=mock_segment_engine,
            n_speakers=2
        )

        self.assertEqual(len(result), 3)
        # 检查 1 和 3 的 speaker_id 是否相同，2 的是否不同
        spk1, spk2, spk3 = result[0][2], result[1][2], result[2][2]
        self.assertEqual(spk1, spk3)
        self.assertNotEqual(spk1, spk2)
        # 检查时间戳
        self.assertEqual(result[0][0], 0.0)
        self.assertEqual(result[0][1], 2.0)
        self.assertEqual(result[1][0], 3.0)
        self.assertEqual(result[1][1], 5.0)
        self.assertEqual(result[2][0], 6.0)
        self.assertEqual(result[2][1], 8.0)

    def test_empty_segments(self):
        """测试无语音纯净段的情况，应直接返回空。"""
        mock_segment_engine = MagicMock()
        mock_segment_engine.process_full_audio.return_value = []

        result = self.speaker_engine.cluster_with_segmentation(
            audio_16k=np.zeros(32000),
            segment_engine=mock_segment_engine,
            n_speakers=2
        )
        self.assertEqual(result, [])

    def test_all_embeddings_none(self):
        """测试所有声纹提取均返回 None 的鲁棒退化情况（不合并，归为默认说话人 1）。"""
        mock_segment_engine = MagicMock()
        mock_segment_engine.process_full_audio.return_value = [
            (0.0, 2.0, np.zeros(16000)),
            (3.0, 5.0, np.zeros(16000)),
        ]
        self.speaker_engine.extract_embedding.return_value = None

        result = self.speaker_engine.cluster_with_segmentation(
            audio_16k=np.zeros(80000),
            segment_engine=mock_segment_engine,
            n_speakers=2
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][2], 1)
        self.assertEqual(result[1][2], 1)

    def test_merge_adjacent_same_speaker(self):
        """测试同人且间隔 < 0.5s 时合并，不同人或间隔 >= 0.5s 时不合并。"""
        mock_segment_engine = MagicMock()
        # 段 1 与段 2 间隔 0.3s (2.3 - 2.0) < 0.5s
        # 段 2 与段 3 间隔 0.1s (4.1 - 4.0) < 0.5s
        mock_segment_engine.process_full_audio.return_value = [
            (0.0, 2.0, np.zeros(16000)),
            (2.3, 4.0, np.zeros(16000)),
            (4.1, 6.0, np.zeros(16000)),
        ]

        v1 = np.ones(192, dtype=np.float32)
        v2 = -np.ones(192, dtype=np.float32)
        # 段 1 与段 2 相同，段 3 不同
        self.speaker_engine.extract_embedding.side_effect = [v1, v1, v2]

        result = self.speaker_engine.cluster_with_segmentation(
            audio_16k=np.zeros(96000),
            segment_engine=mock_segment_engine,
            n_speakers=2
        )

        # 最终应该合并为 2 段：
        # 段 1 + 段 2 合并为 (0.0, 4.0, spk_a)
        # 段 3 单独为 (4.1, 6.0, spk_b)
        self.assertEqual(len(result), 2)
        spk_a = result[0][2]
        spk_b = result[1][2]
        self.assertNotEqual(spk_a, spk_b)

        self.assertEqual(result[0][0], 0.0)
        self.assertEqual(result[0][1], 4.0)
        self.assertEqual(result[1][0], 4.1)
        self.assertEqual(result[1][1], 6.0)


if __name__ == "__main__":
    unittest.main()
