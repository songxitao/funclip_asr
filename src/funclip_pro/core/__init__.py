"""funclip_pro.core — P1 算法 + P3.2 音频采集 + 流式 ASR SDK 核心层。

重型 ML 依赖（torch / funasr / onnxruntime）使用惰性加载，不会在
`import funclip_pro.core` 时自动触发。子模块可直接导入：
  from funclip_pro.core.segmentation import SegmentationEngine
  from funclip_pro.core.audio import BaseStream

通过包名访问也会惰性加载（向后兼容）：
  from funclip_pro.core import SegmentationEngine
"""

import importlib as _importlib

from .models import WordTimestamp, Segment, TranscriptionResult  # noqa: F401 — re-exported public API
from .tokenization import CharTokenizer  # noqa: F401 — re-exported public API


__all__ = [
    "WordTimestamp",
    "Segment",
    "TranscriptionResult",
    "SegmentationEngine",
    "CampPlusSpeaker",
    "segment_sliding_window",
    "SenseVoiceSmall",
    "PyTorchSenseVoice",
    "SeACoParaformer",
    "SherpaSenseVoice",
    "QwenEngine",
    "parse_qwen_timestamps",
    "_split_timestamps_to_segments",
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

# ── 惰性加载映射 ──────────────────────────────────────────────────────────
# 以下子模块包含重型 ML 依赖，在首次访问对应名称时才实际加载。
# key: 对外暴露的名称，value: (子模块名, 内部属性名)
_lazy_attrs: dict[str, tuple[str, str]] = {
    "SegmentationEngine": ("segmentation", "SegmentationEngine"),
    "CampPlusSpeaker": ("speaker", "CampPlusSpeaker"),
    "segment_sliding_window": ("speaker", "segment_sliding_window"),
    "SenseVoiceSmall": ("asr", "SenseVoiceSmall"),
    "PyTorchSenseVoice": ("asr", "PyTorchSenseVoice"),
    "SeACoParaformer": ("asr", "SeACoParaformer"),
    "SherpaSenseVoice": ("asr", "SherpaSenseVoice"),
    "QwenEngine": ("asr", "QwenEngine"),
    "parse_qwen_timestamps": ("asr", "parse_qwen_timestamps"),
    "_split_timestamps_to_segments": ("asr", "_split_timestamps_to_segments"),
    "load_models": ("asr", "load_models"),
    "_assign_clauses_to_speakers": ("alignment", "_assign_clauses_to_speakers"),
    "_assign_clauses_to_speakers_seamless": (
        "alignment",
        "_assign_clauses_to_speakers_seamless",
    ),
    "process_audio_frame": ("audio", "process_audio_frame"),
    "BaseStream": ("audio", "BaseStream"),
    "LoopbackStream": ("audio", "LoopbackStream"),
    "MicStream": ("audio", "MicStream"),
    "MixedStream": ("audio", "MixedStream"),
    "SileroVAD": ("streaming_asr", "SileroVAD"),
    "FsmnVadStreaming": ("streaming_asr", "FsmnVadStreaming"),
    "FunAsrStreamingEngine": ("streaming_asr", "FunAsrStreamingEngine"),
    # 子模块本身（如 from funclip_pro.core import asr）
    "asr": ("asr", None),
    "segmentation": ("segmentation", None),
    "speaker": ("speaker", None),
    "audio": ("audio", None),
    "alignment": ("alignment", None),
    "streaming_asr": ("streaming_asr", None),
}


def __getattr__(name: str):
    """惰性加载重型子模块中的属性。

    Python 在以下场景会调用此函数：
    - `from funclip_pro.core import SegmentationEngine`
    - `funclip_pro.core.SegmentationEngine`
    """
    if name in _lazy_attrs:
        submod, attr = _lazy_attrs[name]
        mod = _importlib.import_module(f".{submod}", __package__)
        if attr is None:
            return mod
        return getattr(mod, attr)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    """列出所有公开的名称（含惰性加载的）。"""
    return sorted(set(__all__ + [k for k in _lazy_attrs if not k.startswith("_")]))
