"""core.alignment：子句→说话人分配对齐（锚点扩散）。

从 asr_onnx_service.py 等价下沉。时间戳单位为毫秒(ms)。
两函数均为自包含逻辑，不依赖外部引擎，仅迁移保持字节级等价。
"""


def _assign_clauses_to_speakers(asr_start, asr_end, text, refined_segs):
    """将 ASR 段识别出的带标点 text，按照标点分句，将每个子句作为一个整体分配给重叠时间最长的说话人。
    返回列表，每个元素为 {"start": int, "end": int, "speaker": str, "text": str}
    """
    if not text.strip():
        return []

    import re
    # 匹配中英文断句标点，包括逗号
    pattern = r'([^，。？！、；：,.?!;:：\s]+[，。？！、；：,.?!;:：\s]*)'
    clauses = re.findall(pattern, text)
    if not clauses:
        clauses = [text]

    total_len = sum(len(c) for c in clauses)
    if total_len == 0:
        return []

    dur = asr_end - asr_start
    curr_start = asr_start

    assigned_clauses = []
    for clause in clauses:
        c_len = len(clause)
        c_dur = dur * (c_len / total_len)
        c_end = curr_start + c_dur

        best_spk = None
        max_overlap = -1.0
        
        for st_ms, en_ms, spk in refined_segs:
            # 计算重叠时间
            overlap = min(c_end, en_ms) - max(curr_start, st_ms)
            if overlap > max_overlap:
                max_overlap = overlap
                best_spk = spk

        # 兜底：如果重合时间为 0 或没找到
        if max_overlap <= 0 or best_spk is None:
            mid_t = curr_start + c_dur / 2
            min_dist = float('inf')
            for st_ms, en_ms, spk in refined_segs:
                dist = min(abs(mid_t - st_ms), abs(mid_t - en_ms))
                if dist < min_dist:
                    min_dist = dist
                    best_spk = spk

        if best_spk is None:
            best_spk = "1"

        assigned_clauses.append({
            "start": int(curr_start),
            "end": int(c_end),
            "speaker": str(best_spk),
            "text": clause
        })

        curr_start = c_end

    # 合并相邻且相同说话人的子句
    merged_sub = []
    if assigned_clauses:
        curr = assigned_clauses[0]
        for idx in range(1, len(assigned_clauses)):
            nxt = assigned_clauses[idx]
            if nxt["speaker"] == curr["speaker"]:
                curr["text"] += nxt["text"]
                curr["end"] = nxt["end"]
            else:
                merged_sub.append(curr)
                curr = nxt
        merged_sub.append(curr)

    return merged_sub


def _assign_clauses_to_speakers_seamless(asr_start, asr_end, text, seamless_segs):
    """将 ASR 段识别出的带标点 text，按标点分句，分配到无缝说话人时间轴上。
    
    与 _assign_clauses_to_speakers 的区别：
    - seamless_segs 包含确定段(int speaker_id)和未知段(str type:"overlap"/"silence")
    - 优先匹配确定段（取重叠时间最长的说话人）
    - 子句完全落在未知段 → 取同一标点大句内最近确定段的说话人（锚点扩散）
    - 整个大句都没有确定段 → 取时间上最近的确定段说话人（兜底）
    
    Returns:
        list of {"start": int, "end": int, "speaker": str, "text": str}
    """
    if not text.strip():
        return []

    import re
    pattern = r'([^，。？！、；：,.?!;:：\s]+[，。？！、；：,.?!;:：\s]*)'
    clauses = re.findall(pattern, text)
    if not clauses:
        clauses = [text]

    total_len = sum(len(c) for c in clauses)
    if total_len == 0:
        return []

    dur = asr_end - asr_start
    curr_start = asr_start

    # 从 seamless_segs 中提取确定段（用于直接匹配）
    determined_segs = [(st, en, spk) for st, en, spk in seamless_segs if isinstance(spk, int)]

    assigned_clauses = []
    for clause in clauses:
        c_len = len(clause)
        c_dur = dur * (c_len / total_len)
        c_end = curr_start + c_dur

        # 1. 优先匹配确定段
        best_spk = None
        max_overlap = -1.0
        for seg_start_ms, seg_end_ms, spk in determined_segs:
            overlap = min(c_end, seg_end_ms) - max(curr_start, seg_start_ms)
            if overlap > max_overlap:
                max_overlap = overlap
                best_spk = spk

        # 2. 无确定段重叠 → 锚点扩散：取最近确定段
        if max_overlap <= 0 or best_spk is None:
            mid_t = curr_start + c_dur / 2
            min_dist = float('inf')
            for seg_start_ms, seg_end_ms, spk in determined_segs:
                dist = min(abs(mid_t - seg_start_ms), abs(mid_t - seg_end_ms))
                if dist < min_dist:
                    min_dist = dist
                    best_spk = spk

        # 3. 兜底：无任何确定段
        if best_spk is None:
            best_spk = 1

        assigned_clauses.append({
            "start": int(curr_start),
            "end": int(c_end),
            "speaker": str(best_spk),
            "text": clause
        })

        curr_start = c_end

    # 合并相邻相同说话人的子句
    merged_sub = []
    if assigned_clauses:
        curr = assigned_clauses[0]
        for idx in range(1, len(assigned_clauses)):
            nxt = assigned_clauses[idx]
            if nxt["speaker"] == curr["speaker"]:
                curr["text"] += nxt["text"]
                curr["end"] = nxt["end"]
            else:
                merged_sub.append(curr)
                curr = nxt
        merged_sub.append(curr)

    return merged_sub
