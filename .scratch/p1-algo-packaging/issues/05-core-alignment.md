# 05 — core.alignment：下沉子句说话人分配对齐（锚点扩散）

**What to build:** 将子句级说话人分配对齐（锚点扩散）下沉到 `src/funclip_pro/core/alignment.py`：`_assign_clauses_to_speakers` 与 `_assign_clauses_to_speakers_seamless`。

**Blocked by:** 02 (core.segmentation), 03 (core.speaker), 04 (core.asr) —— 对齐消费 segmentation 的 refined_segs / speaker 的 labels / asr 的 segments

**Status:** ready-for-agent

- [ ] 函数签名与现 `asr_onnx_service.py` 一致
- [ ] 对齐后 segments 时间戳单位为 ms（回填口径）
- [ ] 锚点扩散逻辑不变

**Module interface contract:**
```python
def _assign_clauses_to_speakers(asr_start, asr_end, text, refined_segs) -> segments: ...
def _assign_clauses_to_speakers_seamless(asr_start, asr_end, text, seamless_segs) -> segments: ...
```
- `refined_segs` / `seamless_segs` 由 segmentation + speaker 产出；时间单位保持与现 `asr_onnx_service.py` 一致（ms）。
- 返回 segments 须带 speaker 标签，供 `utils.srt` 消费。

**Notes:**
- 红线：时间戳 API/评测用 ms；`cluster_with_segmentation` 返回秒，若经 speaker 层进入对齐需确保 ×1000。
- 算法逻辑不变（等价优先）。
- 最高指导：`.superpowers/spec/2026-07-14-refactor-p0-p1-spec.md` L52。
