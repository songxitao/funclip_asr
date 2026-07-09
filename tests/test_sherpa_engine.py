import os
import numpy as np
import pytest

from sherpa_engine import SherpaSenseVoice

SHERPA_MODEL_DIR = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmallOnnx"

pytestmark = pytest.mark.skipif(
    not os.path.exists(SHERPA_MODEL_DIR),
    reason="需要 Sherpa INT8 模型目录",
)


def test_engine_loads():
    engine = SherpaSenseVoice(
        model_dir=SHERPA_MODEL_DIR,
        num_threads=6,
        use_itn=True,
    )
    assert engine.use_itn is True
    assert engine.recognizer is not None


def test_call_returns_list():
    engine = SherpaSenseVoice(
        model_dir=SHERPA_MODEL_DIR,
        num_threads=6,
        use_itn=True,
    )
    # 约 1 秒 16kHz 正弦波（无语音，输出应为空/很短文本，但必须是 list[str]）
    t = np.arange(16000) / 16000.0
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    result = engine([sine])
    assert isinstance(result, list)
    assert all(isinstance(x, str) for x in result)
