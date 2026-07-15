import os
import sys
import numpy as np
import pytest

# 项目根目录
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 添加 SenseVoiceSmall 目录到 PYTHONPATH
sensevoice_dir = os.path.join(BASE, "model", "models", "iic", "SenseVoiceSmall")
if os.path.isdir(sensevoice_dir):
    sys.path.append(sensevoice_dir)
from utils.model_bin import SenseVoiceSmallONNX

@pytest.mark.skip(reason="需要真实 ONNX 模型文件")
def test_extract_feat_equivalence():
    model_dir = os.path.join(BASE, "model", "models", "iic", "SenseVoiceSmall-ONNX")
    model = SenseVoiceSmallONNX(model_dir, batch_size=4, quantize=True, device_id="-1")
    
    # 模拟几个不同长度的音频信号 (16kHz采样率，分别为1秒，2秒，3秒)
    np.random.seed(42)
    waveforms = [np.random.randn(16000 * i).astype(np.float32) for i in range(1, 4)]
    
    # 获取当前多线程模式下的特征提取结果
    feats, feats_len = model.extract_feat(waveforms)
    
    # 临时手动用原串行逻辑进行特征提取以作对比
    original_feats = []
    original_lens = []
    for waveform in waveforms:
        speech, _ = model.frontend.fbank(waveform)
        feat, feat_len = model.frontend.lfr_cmvn(speech)
        original_feats.append(feat)
        original_lens.append(feat_len)
    
    original_feats = model.pad_feats(original_feats, np.max(original_lens))
    original_lens = np.array(original_lens).astype(np.int32)
    
    # 比较两者的形状和数值
    assert np.allclose(feats, original_feats, atol=1e-2)
    assert np.array_equal(feats_len, original_lens)
    print("并发与串行特征提取的数值完全等价！")

if __name__ == "__main__":
    test_extract_feat_equivalence()
