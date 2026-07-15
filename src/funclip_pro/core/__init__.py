"""funclip_pro.core — P1 算法 + P3.2 音频采集 + 流式 ASR SDK 核心层。

统一导出下沉后的算法引擎、音频流接口与流式 ASR 引擎，供 pipeline / 路由层以
`from funclip_pro.core import X` 形式绝对导入。
"""

from .segmentation import SegmentationEngine
from .speaker import CampPlusSpeaker, segment_sliding_window
from .asr import (
    SenseVoiceSmall,
    PyTorchSenseVoice,
    SherpaSenseVoice,
    QwenEngine,
    parse_qwen_timestamps,
    load_models,
)
from .tokenization import CharTokenizer
from .alignment import (
    _assign_clauses_to_speakers,
    _assign_clauses_to_speakers_seamless,
)
from .audio import (
    process_audio_frame,
    BaseStream,
    LoopbackStream,
    MicStream,
    MixedStream,
)
from .streaming_asr import (
    SileroVAD,
    FsmnVadStreaming,
    FunAsrStreamingEngine,
)

__all__ = [
    "SegmentationEngine",
    "CampPlusSpeaker",
    "segment_sliding_window",
    "SenseVoiceSmall",
    "PyTorchSenseVoice",
    "SherpaSenseVoice",
    "QwenEngine",
    "parse_qwen_timestamps",
    "load_models",
    "CharTokenizer",
    "_assign_clauses_to_speakers",
    "_assign_clauses_to_speakers_seamless",
    "process_audio_frame",
    "BaseStream",
    "LoopbackStream",
    "MicStream",
    "MixedStream",
    "SileroVAD",
    "FsmnVadStreaming",
    "FunAsrStreamingEngine",
]
