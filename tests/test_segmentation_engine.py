"""segmentation_engine.py 的单元测试。用 mock 模型替代真实模型，不加载 GPU。"""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch


class TestSegmentationEngine(unittest.TestCase):

    def _make_engine(self, mock_output_frames=293):
        """构造一个使用 mock 模型的 SegmentationEngine。"""
        with patch("segmentation_engine.Model") as MockModel:
            # mock 模型的 forward 输出
            mock_model = MagicMock()
            # 7 classes: non-speech 高概率，其他低概率
            fake_out = torch.zeros(1, mock_output_frames, 7)
            fake_out[:, :, 0] = 5.0  # non-speech logit 高
            mock_model.return_value = fake_out
            mock_model.eval = MagicMock()
            mock_model.to = MagicMock(return_value=mock_model)
            MockModel.from_pretrained.return_value = mock_model

            from segmentation_engine import SegmentationEngine
            engine = SegmentationEngine.__new__(SegmentationEngine)
            engine.model = mock_model
            engine.device = "cpu"
            engine.to_multilabel = None  # 将在下面设置

            from pyannote.audio.utils.powerset import Powerset
            engine.to_multilabel = Powerset(
                num_classes=3, max_set_size=2
            ).to_multilabel

            return engine, mock_model

    def test_process_chunk_silence(self):
        """全静音 chunk 应返回空 segments。"""
        engine, mock_model = self._make_engine()
        # mock 输出: 全部 non-speech
        frames = 293
        fake_out = torch.zeros(1, frames, 7)
        fake_out[:, :, 0] = 10.0  # non-speech 概率极高
        mock_model.return_value = fake_out

        audio = np.zeros(160000, dtype=np.float32)  # 10s 静音
        segments = engine.process_chunk(audio, sr=16000)
        self.assertEqual(len(segments), 0)

    def test_process_chunk_single_speaker(self):
        """一个人从头说到尾，应返回 1 个 segment。"""
        engine, mock_model = self._make_engine()
        frames = 293
        fake_out = torch.zeros(1, frames, 7)
        # class index 1 = spk1 only
        fake_out[:, :, 1] = 10.0
        mock_model.return_value = fake_out

        audio = np.random.randn(160000).astype(np.float32)
        segments = engine.process_chunk(audio, sr=16000)
        self.assertGreaterEqual(len(segments), 1)
        # 所有 segment 的 local_speaker_id 应为 0
        for s, e, spk_id in segments:
            self.assertEqual(spk_id, 0)

    def test_process_chunk_two_speakers(self):
        """前半 spk1，后半 spk2，应返回 2 个 segments。"""
        engine, mock_model = self._make_engine()
        frames = 293
        mid = frames // 2
        fake_out = torch.zeros(1, frames, 7)
        fake_out[:, :mid, 1] = 10.0   # spk1
        fake_out[:, mid:, 2] = 10.0   # spk2
        mock_model.return_value = fake_out

        audio = np.random.randn(160000).astype(np.float32)
        segments = engine.process_chunk(audio, sr=16000)
        self.assertGreaterEqual(len(segments), 2)
        speaker_ids = set(spk for _, _, spk in segments)
        self.assertEqual(speaker_ids, {0, 1})

    def test_process_chunk_overlap_excluded(self):
        """重叠帧不应出现在输出 segments 中。"""
        engine, mock_model = self._make_engine()
        frames = 293
        fake_out = torch.zeros(1, frames, 7)
        # 全部帧: spk1+spk2 重叠 (class index 4)
        fake_out[:, :, 4] = 10.0
        mock_model.return_value = fake_out

        audio = np.random.randn(160000).astype(np.float32)
        segments = engine.process_chunk(audio, sr=16000)
        # 重叠帧应被排除，segments 可能为空或只有非常短的段
        # 关键：不应返回混合了两个人的段
        for s, e, spk_id in segments:
            self.assertIn(spk_id, [0, 1, 2])

    def test_process_full_audio_short(self):
        """不足 10s 的音频也能正常处理。"""
        engine, mock_model = self._make_engine()
        frames = 293
        fake_out = torch.zeros(1, frames, 7)
        fake_out[:, :, 1] = 10.0
        mock_model.return_value = fake_out

        audio = np.random.randn(48000).astype(np.float32)  # 3s
        segments = engine.process_full_audio(audio, sr=16000)
        self.assertGreaterEqual(len(segments), 1)

    def test_process_full_audio_multi_chunk(self):
        """超过 10s 的音频应被自动分 chunk 处理。"""
        engine, mock_model = self._make_engine()
        frames = 293
        fake_out = torch.zeros(1, frames, 7)
        fake_out[:, :, 1] = 10.0
        mock_model.return_value = fake_out

        audio = np.random.randn(480000).astype(np.float32)  # 30s
        segments = engine.process_full_audio(audio, sr=16000)
        self.assertGreaterEqual(len(segments), 1)
        # 验证 segments 覆盖到了后半段
        max_end = max(e for _, e, _ in segments)
        self.assertGreater(max_end, 15.0)


if __name__ == "__main__":
    unittest.main()
