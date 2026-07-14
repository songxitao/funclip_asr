"""SRT 转换与段内合并工具（下沉自根目录 asr_onnx_service.py）。

本模块提供纯函数，不加载任何模型权重。
- _ms_to_srt：毫秒时间戳 -> SRT 时间格式
- _merge_same_speaker_segments：合并相邻同说话人段（调用方按 VAD 段传入子列表，保证不跨段）
- _segments_to_srt：segments -> 合法 SRT 字符串
"""


def _ms_to_srt(ms):
    """将毫秒转换为 SRT 时间格式 HH:MM:SS,mmm"""
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    mill = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{mill:03d}"


def _merge_same_speaker_segments(segments):
    """合并相邻同说话人的段。
    注意：本函数本身只做相邻同说话人合并；"VAD 段内不跨段"由调用方
    保证——调用方对单个 VAD 段内的子列表调用本函数，而非跨段传入。
    """
    if not segments:
        return []
    merged = []
    curr = dict(segments[0])
    for seg in segments[1:]:
        if seg["speaker"] == curr["speaker"]:
            curr["text"] += seg["text"]
            curr["end"] = seg["end"]
        else:
            merged.append(curr)
            curr = dict(seg)
    merged.append(curr)
    return merged


def _segments_to_srt(segments):
    """将 segments 列表转换为 SRT 字幕格式"""
    lines = []
    for idx, seg in enumerate(segments, start=1):
        start_ms = int(seg["start"])
        end_ms = int(seg["end"])
        speaker = seg.get("speaker", "")
        text = seg.get("text", "").strip()
        if not text:
            continue
        label = f"[说话人{speaker}] " if speaker else ""
        lines.append(str(idx))
        lines.append(f"{_ms_to_srt(start_ms)} --> {_ms_to_srt(end_ms)}")
        lines.append(f"{label}{text}")
        lines.append("")
    return "\n".join(lines)
