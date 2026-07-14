# 09 — CLI 薄客户端：重写 cli_transcribe.py

**What to build:** `cli_transcribe.py` 改为 import `OfflinePipeline` 调用转写，删除内联重复逻辑；将 `asr_service.py` 的无 diarization `_run_inference` 收敛到 core（薄再导出或删除，由本票决定，倾向 `asr_service.py` 改为再导出 `core.asr` 能力）。

**Blocked by:** 07 (pipeline.offline)

**Status:** ready-for-agent

- [ ] CLI 转写输出与重构前一致
- [ ] 无重复推理代码

**Notes:**
- 收敛目标：消除 `asr_service.py` 与 `asr_onnx_service.py` 中重复的 `_run_inference` / `_merge_vad_segments`，统一由 `OfflinePipeline` 调度。
- 红线：等价优先，算法逻辑不变。
- 最高指导：`.superpowers/spec/2026-07-14-refactor-p0-p1-spec.md` L58。
