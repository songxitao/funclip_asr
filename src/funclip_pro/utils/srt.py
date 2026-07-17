"""SRT 转换与段内合并工具（下沉自根目录 asr_onnx_service.py）。

本模块提供纯函数，不加载任何模型权重。
- _ms_to_srt：毫秒时间戳 -> SRT 时间格式
- _merge_same_speaker_segments：合并相邻同说话人段（调用方按 VAD 段传入子列表，保证不跨段）
- _segments_to_srt：segments -> 合法 SRT 字符串
"""

from __future__ import annotations

from typing import Any


def _ms_to_srt(ms):
    """将毫秒转换为 SRT 时间格式 HH:MM:SS,mmm"""
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    mill = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{mill:03d}"


def _get_seg_field(seg: Any, key: str, default=None):
    """从 dict 或 dataclass 中安全获取字段。

    向后兼容：同时支持 dict (seg[key]) 和 dataclass (getattr(seg, key))。
    """
    if isinstance(seg, dict):
        return seg.get(key, default)
    return getattr(seg, key, default)


def _merge_same_speaker_segments(segments):
    """合并相邻同说话人的段。
    注意：本函数本身只做相邻同说话人合并；"VAD 段内不跨段"由调用方
    保证——调用方对单个 VAD 段内的子列表调用本函数，而非跨段传入。
    支持 dict 和 Segment dataclass 两种输入。
    """
    if not segments:
        return []

    def _copy_seg(seg):
        if isinstance(seg, dict):
            return dict(seg)
        return {
            "start": seg.start_ms,
            "end": seg.end_ms,
            "text": seg.text,
            "speaker": seg.speaker,
        }

    def _get_speaker(seg):
        if isinstance(seg, dict):
            return seg.get("speaker", "")
        return seg.speaker

    def _get_end(seg):
        if isinstance(seg, dict):
            return seg.get("end", 0)
        return seg.end_ms

    merged = []
    curr = _copy_seg(segments[0])
    for seg in segments[1:]:
        if _get_speaker(seg) == curr["speaker"]:
            curr["text"] += _get_seg_field(seg, "text", "")
            curr["end"] = _get_end(seg)
        else:
            merged.append(curr)
            curr = _copy_seg(seg)
    merged.append(curr)
    return merged


def _segments_to_srt(segments):
    """将 segments 列表转换为 SRT 字幕格式

    支持 Segment dataclass 和 dict 两种输入。
    """
    lines = []
    for idx, seg in enumerate(segments, start=1):
        if isinstance(seg, dict):
            start_ms = int(seg["start"])
            end_ms = int(seg["end"])
            speaker = seg.get("speaker", "")
            text = seg.get("text", "").strip()
        else:
            start_ms = seg.start_ms
            end_ms = seg.end_ms
            speaker = seg.speaker
            text = seg.text.strip()

        if not text:
            continue
        label = f"[说话人{speaker}] " if speaker else ""
        lines.append(str(idx))
        lines.append(f"{_ms_to_srt(start_ms)} --> {_ms_to_srt(end_ms)}")
        lines.append(f"{label}{text}")
        lines.append("")
    return "\n".join(lines)
