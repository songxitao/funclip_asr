# 02 — core.segmentation：下沉 SegmentationEngine

**What to build:** 将根 `segmentation_engine.py` 中的 `SegmentationEngine` 下沉到 `src/funclip_pro/core/segmentation.py`，作为核心分割引擎。保留 P0 已落地的动态路径（`DEFAULT_SEG_MODEL_DIR = resolve_model_path(...)`），不得回归硬编码盘符。DLL 补丁由 `config.loader` 在导入时执行，本模块 import 即触发。

**Blocked by:** 01 (包骨架搭建)

**Status:** ready-for-agent

- [ ] `from funclip_pro.core.segmentation import SegmentationEngine` 可用
- [ ] GPU 推理不崩；`powerset.cpu()` 必须在 `to_multilabel` 前调用（红线）
- [ ] 路径零硬编码（沿用 `resolve_model_path`）

**Module interface contract (签名与现 segmentation_engine.py 完全一致):**
```python
class SegmentationEngine:
    def __init__(self, model_dir: str = DEFAULT_SEG_MODEL_DIR, device: str = "cpu"): ...
    def process_chunk(self, ...): ...
    def process_full_audio(self, ...): ...
    def process_full_audio_seamless(self, ...): ...
    def _process_chunk_seamless(self, ...): ...
    def _type_label(self, ft): ...   # 私有辅助
```

**Notes:**
- 旧 `segmentation_engine.py` 暂保留（扩展-收缩的"旧形态"），T08/T09 收缩步再决定是否改为薄再导出。
- numpy 锁 1.26.4（AGENTS.md 红线）。
- 最高指导：`.superpowers/spec/2026-07-14-refactor-p0-p1-spec.md` L50。
