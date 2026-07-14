# 08 — FastAPI 薄路由：重写 asr_onnx_service.py

**What to build:** 重写 `asr_onnx_service.py` 为薄路由：仅保留 FastAPI app + `/transcribe` 路由 + 读取 config + 实例化 `OfflinePipeline` 调用；删除所有已从包迁走的推理类定义、对齐、`SRT`、`_run_inference` 等。保留 DLL 补丁调用（`apply_dll_patch`）。

**Blocked by:** 07 (pipeline.offline)

**Status:** ready-for-agent

- [ ] `E:\conda\envs\asr_ui_env\python.exe asr_onnx_service.py` 启动成功
- [ ] `/transcribe` 经 `OfflinePipeline` 返回与重构前等价结果
- [ ] DER `seg_clustering` 14.60% 口径无回归（独立 8003 端口验证，勿杀用户 8002）
- [ ] 顶部 DLL 补丁改为调用 `apply_dll_patch()`，路径零硬编码

**Notes:**
- 这是扩展-收缩策略的"收缩"步：重写后根文件不再含推理/对齐/SRT 定义。
- 旧根引擎文件（`segmentation_engine.py` / `speaker_engine.py` / `torch_engine.py` / `sherpa_engine.py`）在本票或 T09 中处理为薄再导出或删除，确保外部 import 不破。
- DER 评测必须显式 `seg_clustering`，否则默认口径得 49%~57% 错误高值。
- 最高指导：`.superpowers/spec/2026-07-14-refactor-p0-p1-spec.md` L57。
