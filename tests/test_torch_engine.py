"""PyTorch-GPU 引擎接口测试（TDD 先行）。

验证 PyTorchSenseVoice 的接口契约：
- 接收 list[np.ndarray]（16k 波形）
- 返回 list[str]（已剥离 <|...|> 标签）

不要求文本有意义，仅需验证接口与标签剥离。模型目录不存在时整体跳过。
"""
import os

import numpy as np
import pytest

MODEL_DIR = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmall"

pytestmark = pytest.mark.skipif(
    not os.path.isdir(MODEL_DIR),
    reason="PyTorch SenseVoiceSmall 模型目录不存在，跳过接口测试",
)

from torch_engine import PyTorchSenseVoice


def _sine(seconds: float = 1.0, sr: int = 16000) -> np.ndarray:
    return (0.3 * np.sin(2 * np.pi * 440.0 * np.arange(int(seconds * sr)) / sr)).astype(np.float32)


def test_torch_engine_call_returns_list_of_str():
    """实例化引擎并用 1s 正弦 list[np.ndarray] 调用，断言返回 list[str] 且已剥离标签。"""
    engine = PyTorchSenseVoice(model_dir=MODEL_DIR, device="cpu")
    res = engine([_sine()])

    assert isinstance(res, list), "应返回 list"
    assert all(isinstance(x, str) for x in res), "每个元素应为 str"
    # 标签应已被剥离，不应出现 <|...|>
    assert "<|" not in "".join(res), "输出不应包含 <|...|> 标签"
