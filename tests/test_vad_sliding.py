import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from speaker_engine import CampPlusSpeaker

class MockTensor:
    """模拟带有 cpu().numpy() 接口的 Tensor 对象"""
    def __init__(self, data):
        self.data = np.asarray(data)
    def cpu(self):
        return self
    def numpy(self):
        return self.data

def test_extract_embedding_sliding_mean_robust():
    """测试滑动窗口平均及其 L2 归一化，采用抗滑窗数目变动的韧性 Mock 方式"""
    sr = 16000
    audio = np.random.randn(int(4.5 * sr))
    # 前半段设为 1.0，后半段设为 -1.0，以模拟部分滑窗有有效值，部分滑窗为 None
    audio[:int(2.0 * sr)] = 1.0
    audio[int(2.0 * sr):] = -1.0

    with patch('funasr.AutoModel') as mock_auto:
        speaker = CampPlusSpeaker(model_dir="mock_dir", device="cpu")
        
        # 模拟 extract_embedding：仅当子片均值大于 0.1 时返回有效 embedding
        # 且返回的 embedding 严格满足 y = 2x 的比例关系，从而平均后的方向也为 [1.0, 2.0]
        def mock_extract(samp):
            val = np.mean(samp)
            if val > 0.1:
                return np.array([val, val * 2.0])
            return None
            
        speaker.extract_embedding = MagicMock(side_effect=mock_extract)
        emb = speaker.extract_embedding_sliding_mean(audio, sr=sr)
        
        assert emb is not None
        expected = np.array([1.0, 2.0])
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_array_almost_equal(emb, expected, decimal=5)

def test_extract_embedding_sliding_mean_none():
    """边界测试：输入为 None 时安全返回 None"""
    with patch('funasr.AutoModel') as mock_auto:
        speaker = CampPlusSpeaker(model_dir="mock_dir", device="cpu")
        assert speaker.extract_embedding_sliding_mean(None) is None

def test_extract_embedding_sliding_mean_too_short():
    """边界测试：输入音频过短时正确兜底且不崩溃"""
    with patch('funasr.AutoModel') as mock_auto:
        speaker = CampPlusSpeaker(model_dir="mock_dir", device="cpu")
        audio = np.random.randn(1000)
        speaker.extract_embedding = MagicMock(return_value=None)
        
        emb = speaker.extract_embedding_sliding_mean(audio)
        assert emb is None
        speaker.extract_embedding.assert_called_once()

def test_extract_embedding_sliding_mean_all_none_fallback():
    """边界测试：所有子窗均为 None 时，能否退化到对整段再次提取"""
    sr = 16000
    audio = np.random.randn(int(4.5 * sr))
    
    with patch('funasr.AutoModel') as mock_auto:
        speaker = CampPlusSpeaker(model_dir="mock_dir", device="cpu")
        
        # 模拟 extract_embedding：
        # 如果是子窗（长度小于整段音频），返回 None
        # 如果是整段音频，返回有效 embedding
        def mock_extract(samp):
            if len(samp) < len(audio):
                return None
            return np.array([3.0, 4.0])
            
        speaker.extract_embedding = MagicMock(side_effect=mock_extract)
        emb = speaker.extract_embedding_sliding_mean(audio, sr=sr)
        
        assert emb is not None
        expected = np.array([3.0, 4.0])
        np.testing.assert_array_almost_equal(emb, expected, decimal=5)

def test_extract_embedding_sliding_mean_tensor_compat():
    """边界测试：PyTorch Tensor 及普通 Numpy 数组输入兼容性"""
    sr = 16000
    audio = np.random.randn(int(4.5 * sr))
    tensor_audio = MockTensor(audio)
    
    with patch('funasr.AutoModel') as mock_auto:
        speaker = CampPlusSpeaker(model_dir="mock_dir", device="cpu")
        speaker.extract_embedding = MagicMock(return_value=np.array([1.0, 0.0]))
        
        emb = speaker.extract_embedding_sliding_mean(tensor_audio, sr=sr)
        assert emb is not None
        np.testing.assert_array_almost_equal(emb, np.array([1.0, 0.0]), decimal=5)


def test_cluster_vad_sliding():
    with patch('funasr.AutoModel') as mock_auto:
        speaker = CampPlusSpeaker(model_dir="mock_dir", device="cpu")
        chunks = [np.random.randn(32000), np.random.randn(48000), np.random.randn(32000)]
        mock_mean_embs = [
            np.array([1.0, 0.0]),
            np.array([0.9, 0.1]),
            np.array([0.0, 1.0])
        ]
        speaker.extract_embedding_sliding_mean = MagicMock(side_effect=mock_mean_embs)
        result = speaker.cluster(chunks, strategy="vad_sliding", n_speakers=2)
        assert len(result) == 3
        assert result[0] == result[1]
        assert result[0] != result[2]
        assert result[0] in [1, 2]

