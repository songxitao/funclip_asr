"""funclip_pro.core — P1 算法 SDK 核心层。

统一导出下沉后的算法引擎，供 pipeline / 路由层以
`from funclip_pro.core import X` 形式绝对导入。
"""

from .segmentation import SegmentationEngine
from .speaker import CampPlusSpeaker, segment_sliding_window
from .asr import (
    SenseVoiceSmall,
    PyTorchSenseVoice,
    SherpaSenseVoice,
    load_models,
)
from .tokenization import CharTokenizer
from .alignment import (
    _assign_clauses_to_speakers,
    _assign_clauses_to_speakers_seamless,
)

__all__ = [
    "SegmentationEngine",
    "CampPlusSpeaker",
    "segment_sliding_window",
    "SenseVoiceSmall",
    "PyTorchSenseVoice",
    "SherpaSenseVoice",
    "load_models",
    "CharTokenizer",
    "_assign_clauses_to_speakers",
    "_assign_clauses_to_speakers_seamless",
]
