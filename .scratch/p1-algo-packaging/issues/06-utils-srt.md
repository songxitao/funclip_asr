# 06 — utils.srt：下沉 SRT 转换与段内合并

**What to build:** 将 SRT 转换与段内合并下沉到 `src/funclip_pro/utils/srt.py`：`_ms_to_srt`、`_merge_same_speaker_segments`（"段内合并"，限制在 VAD 段内合并、不跨段）、`_segments_to_srt`。

**Blocked by:** 05 (core.alignment) —— SRT 消费对齐后的 diarized segments

**Status:** ready-for-agent

- [ ] 输出 SRT 格式合法（可被播放器解析）
- [ ] 同说话人相邻段按 VAD 段内合并（不跨段，红线）
- [ ] 时间戳 ms

**Module interface contract:**
```python
def _ms_to_srt(ms): ...
def _merge_same_speaker_segments(segments) -> merged_segments: ...   # VAD 段内相邻同说话人合并，不跨段
def _segments_to_srt(segments) -> str: ...                            # 合法 SRT 字符串
```

**Notes:**
- "段内合并" = `_merge_same_speaker_segments`，必须限制在同 VAD 段内，禁止跨段合并（AGENTS.md / HANDOFF 红线，由前序会话验证）。
- 时间戳单位 ms。
- 最高指导：`.superpowers/spec/2026-07-14-refactor-p0-p1-spec.md` L53。
