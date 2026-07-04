import librosa
import numpy as np

def merge_vad_segments(segments, max_gap_ms=300, max_duration_ms=12000):
    """
    Merge raw VAD segments into larger chunks.
    segments: [[start_ms, end_ms], ...]
    """
    if not segments: return []
    
    merged = []
    curr_start, curr_end = segments[0]
    
    for next_start, next_end in segments[1:]:
        gap = next_start - curr_end
        dur = curr_end - curr_start
        
        # Merge if gap is small AND merged segment won't be too long
        if gap < max_gap_ms and (dur + (next_end - next_start) + gap) < max_duration_ms:
            curr_end = next_end
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end
            
    merged.append((curr_start, curr_end))
    return merged

def format_timestamp(seconds):
    """0.0 -> 00:00:00,000"""
    whole_seconds = int(seconds)
    milliseconds = int((seconds - whole_seconds) * 1000)
    
    hours = whole_seconds // 3600
    minutes = (whole_seconds % 3600) // 60
    seconds = whole_seconds % 60
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def make_srt_block(idx, start_sec, end_sec, text):
    s_str = format_timestamp(start_sec)
    e_str = format_timestamp(end_sec)
    return f"{idx}\n{s_str} --> {e_str}\n{text}\n\n"
