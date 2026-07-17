"""funclip_pro.core.models — 统一数据模型（dataclass）。

定义三个核心 dataclass，消除各引擎 / 分支返回松散 dict 导致的 key 不一致问题。

- WordTimestamp：词级时间戳（Qwen ForcedAligner 输出映射）
- Segment：句级转写片段，内含词级时间戳列表
- TranscriptionResult：流水线单次转写的完整结果
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WordTimestamp:
    """词级时间戳，对应 Qwen ForcedAligner 返回的单个 token。

    Attributes:
        text: 词/字文本
        start_ms: 起始毫秒
        end_ms: 结束毫秒
        confidence: 置信度（引擎支持时可用）
    """
    text: str
    start_ms: int
    end_ms: int
    confidence: Optional[float] = None

    @classmethod
    def from_qwen_ts(cls, ts: dict, offset_ms: int = 0) -> "WordTimestamp":
        """从 Qwen 格式时间戳字典构造。

        Qwen 返回的 ts 格式为 {"text": str, "start": float(sec), "end": float(sec)}。
        """
        return cls(
            text=ts.get("text", ""),
            start_ms=int(ts.get("start", 0.0) * 1000 + offset_ms),
            end_ms=int(ts.get("end", 0.0) * 1000 + offset_ms),
        )


@dataclass
class Segment:
    """句级转写片段，对应一个 VAD 段或智能分句段。

    Attributes:
        start_ms: 起始毫秒
        end_ms: 结束毫秒
        text: 转写文本（含标点）
        speaker: 说话人标签，空字符串表示未标注
        words: 词级时间戳列表（用于卡拉 OK 等字级高亮需求）
    """
    start_ms: int
    end_ms: int
    text: str
    speaker: str = ""
    words: list[WordTimestamp] = field(default_factory=list)

    def to_dict(self) -> dict:
        """向后兼容：转为 dict（供仍使用 dict 格式的 SRT 工具使用）。"""
        return {
            "start": self.start_ms,
            "end": self.end_ms,
            "text": self.text,
            "speaker": self.speaker,
        }


@dataclass
class TranscriptionResult:
    """流水线单次转写的完整结果。

    Attributes:
        text: 全文转写文本（不含说话人标签，等价之前的 raw_text）
        engine: 实际使用的引擎标识
        segments: 句级转写片段列表
        language: 检测/指定的语言
        duration_ms: 音频总时长毫秒
        diarized_text: 带说话人标签的文本（可选，仅 diarize=True 时非空）
    """
    text: str
    engine: str
    segments: list[Segment]
    language: str = "auto"
    duration_ms: int = 0
    diarized_text: str = ""
