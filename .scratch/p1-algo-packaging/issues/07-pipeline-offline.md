# 07 — pipeline.offline：整合 OfflinePipeline 统一转写流水线

**What to build:** 整合统一转写流水线 `src/funclip_pro/pipeline/offline.py`：等价原 `asr_onnx_service._run_inference` 的编排（VAD 策略选择 + 引擎选择 + `seg_clustering` 分支 + 对齐 + SRT），对外暴露 `OfflinePipeline` 统一步骤管理器。

**Blocked by:** 02, 03, 04, 05, 06 (core.segmentation, core.speaker, core.asr, core.alignment, utils.srt)

**Status:** ready-for-agent

- [ ] 单测可跑通（无 GPU 部分用 mock / 单测隔离）
- [ ] `seg_clustering` 分支产出与重构前字节级等价
- [ ] 时间戳 ms；DER `seg_clustering` 口径对齐

**Module interface contract:**
```python
class OfflinePipeline:
    def __init__(self, ...): ...   # 配置走 config.loader（resolve_model_path / apply_dll_patch）
    def run(self, audio_path, vad_strategy="auto", diarize=False, engine=None,
            language=[0], textnorm=[15], ...) -> (raw_text, engine_key, segments, diarized_text): ...
```
- 返回四元组 `(raw_text, engine_key, segments, diarized_text)` 为红线，任何分支不得提前 return 列表。
- 编排等价原 `_run_inference`（含 `_select_engine` / `_use_vad` / `_cheap_trim` / `_clean` / `_decode` / `_post_punc` / seg_clustering 分支 / 对齐 / SRT）。

**Notes:**
- 算法逻辑与重构前完全等价；DLL 补丁保活（`apply_dll_patch()`）。
- 最高指导：`.superpowers/spec/2026-07-14-refactor-p0-p1-spec.md` L54。
