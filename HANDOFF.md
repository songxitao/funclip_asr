# Handoff: 无缝说话人时间轴——seg 丢弃段回收与子句级说话人融合（实施 + 评测）

## Session Metadata
- Created: 2026-07-13 23:28
- Project: E:\project\funclip-pro
- Branch: main
- Session duration: 约 30 分钟
- Continues from: HANDOFF.md (2026-07-13 22:43:34, seg_clustering 全局聚类 + 子句投票分配)

### Recent Commits (for context)
  - 5cd2fdf feat: 新增 cli_transcribe.py 命令行转写客户端
  - 6ad166f docs: 更新 dify_openapi.yaml，增加 diarize 分人参数
  - 4899703 docs: 建立三驾马车文档系统 (README/ONBOARDING_MANUAL/AGENTS) 规范协同开发

## Current State Summary

本会话完成了"无缝说话人时间轴"设计方案的**全量代码实施、单元测试编写、实测 DER 评测**。核心改造是：将 segmentation-3.0 的输出从"只保留纯净段"改为"全量输出（含重叠/静音段）"，构建一条无窟窿的说话人时间轴，然后在子句分配阶段通过"锚点扩散"让未知段搭确定段的便车回收丢失语音。

已暂存 5 个文件（segmentation_engine.py / speaker_engine.py / asr_onnx_service.py / tests/test_seg_seamless.py / docs/superpowers/plans/...）。AliMeeting 近场单场评测结果：DER=14.54%（对比旧版 15.13%，MISS 从 11.3% 骤降至 0.7%）。

## Codebase Understanding

### Architecture Overview

新管线采用双轨并行 + 帧级提纯 + 锚点扩散方案：
1. VAD→ASR 路径不变（粗粒度段 + 800ms padding 防丢字）
2. seg-3.0→Cam++→谱聚类路径改为无缝输出（不再丢帧）
3. 汇合后 `_assign_clauses_to_speakers_seamless()` 做锚点扩散分配
4. 确定段取重叠最久的说话人，未知段取同一子句内最近确定段

### Critical Files

| File | Purpose | Relevance |
|------|---------|-----------|
| `segmentation_engine.py` | 新增 `process_full_audio_seamless()` + `_process_chunk_seamless()` | 输出所有段类型(single/overlap/silence)，无缝时间轴 |
| `speaker_engine.py` | 新增 `cluster_with_seamless_segmentation()` | 只对 single 段提 embedding+聚类，保留未知段标记 |
| `asr_onnx_service.py` | 新增 `_assign_clauses_to_speakers_seamless()` + 改造 seg_clustering 分支 | 锚点扩散逻辑；主分支改为调用无缝流程 |
| `tests/test_seg_seamless.py` | 15 条单元测试 | 覆盖无缝时间轴/聚类/锚点扩散/兜底等场景 |
| `docs/superpowers/plans/2026-07-13-seamless-speaker-timeline.md` | 实施计划 | 从 spec 到任务拆解的完整记录 |

### Key Patterns Discovered

- 无缝时间轴的 seg_type 有三种："single"(int speaker_id)、"overlap"(str)、"silence"(str)
- `cluster_with_seamless_segmentation()` 的合并逻辑遇到 overlap/silence 字符串标记会断开，不跨未知段合并同人段——这是有意设计（怕合并错）
- `_assign_clauses_to_speakers_seamless()` 本质上已经在做"重叠比例"匹配，即计算子句与每个确定段的重叠时间取最大——spec 里的典型例子已能正确推导

## Work Completed

### Tasks Finished

- [x] **Task 1**: segmentation_engine.py 新增 `process_full_audio_seamless()` 和 `_process_chunk_seamless()` — 保留所有帧段类型
- [x] **Task 2**: speaker_engine.py 新增 `cluster_with_seamless_segmentation()` — 只对 single 段提向量，保留未知段
- [x] **Task 3**: asr_onnx_service.py 新增 `_assign_clauses_to_speakers_seamless()` + 改造 seg_clustering 主分支
- [x] **Task 4**: 15 条单元测试全部通过
- [x] **Git 暂存**: segmentation_engine.py, speaker_engine.py, asr_onnx_service.py, tests/test_seg_seamless.py, docs/superpowers/plans/...
- [x] **实测 DER**: AliMeeting near R8002_M8002 跑通，DER=14.54%
- [x] **三种策略对比分析**: spectral vs old seg_clustering vs seamless，含图表

### Files Modified

| File | Changes | Rationale |
|------|---------|-----------|
| `segmentation_engine.py` | +`process_full_audio_seamless()` + `_process_chunk_seamless()` (138 行) | 构建无缝时间轴，保留所有帧类型 |
| `speaker_engine.py` | +`cluster_with_seamless_segmentation()` (114 行) + import Union | 只对确定段提 embedding |
| `asr_onnx_service.py` | +`_assign_clauses_to_speakers_seamless()` + 改造主分支 seg_clustering | 锚点扩散逻辑 |
| `tests/test_seg_seamless.py` | 新建 15 条测试 (252 行) | 覆盖无缝/聚类/分配/兜底 |
| `docs/superpowers/plans/2026-07-13-seamless-speaker-timeline.md` | 新建实施计划文档 | 记录任务拆解 |

### Decisions Made

| Decision | Options Considered | Rationale |
|----------|-------------------|-----------|
| 无缝时间轴改为主分支 | 另起新策略名 vs 替换现有 seg_clustering | seg_clustering 本身就是新策略，直接升级 |
| 锚点扩散硬分配 | 置信度加权 vs 软分配 | 现阶段先保召回率（MISS↓），CONF 后续优化 |
| 保留旧 seg_clustering 分支作为 fallback | 移除 vs 保留 | 无缝路径异常时兜底，但也引入了冗余 |
| 旧 HANDOFF.md 不纳入暂存 | 纳入 vs 不纳入 | 属于上个会话的修改，不混入本 feature |

## Immediate Next Steps

1. **P0 优化 — 置信度加权的锚点扩散**：当前子句如果落在未知段，直接全量归给最近确定段。改为看子句与多个确定段的重叠比例，低置信度时不硬分配。预期 CONF 从 13.6% 降至 8% 以下。
2. **P1 优化 — 同人段跨重叠段合并**：当前合并逻辑遇到 `overlap` 标记断开。研究能否让同人段跨未知段合并，减少 hyp 段数。
3. **P2 优化 — 移除第二个冗余 seg_clustering 分支**：`_run_inference` 里有两个 seg_clustering 分支（主分支 L501 + 旧分支 L622），可以精简去掉旧的。
4. **P2 优化 — SpectralClustering n_neighbors 剪枝**：当 single 段数很多时限制 n_neighbors，加速聚类。

## Pending Work

### Immediate Next Steps (detailed)

1. **P0 优化 — 置信度加权的锚点扩散**：当前子句如果落在未知段，直接全量归给最近确定段。改为看子句与多个确定段的重叠比例，低置信度时不硬分配。预期 CONF 从 13.6% 降至 8% 以下。
2. **P1 优化 — 同人段跨重叠段合并**：当前合并逻辑遇到 `overlap` 标记断开。研究能否让同人段跨未知段合并，减少 hyp 段数。
3. **P2 优化 — 移除第二个冗余 seg_clustering 分支**：`_run_inference` 里有两个 seg_clustering 分支（主分支 L501 + 旧分支 L622），可以精简去掉旧的。
4. **P2 优化 — SpectralClustering n_neighbors 剪枝**：当 single 段数很多时限制 n_neighbors，加速聚类。

### Blockers/Open Questions

- [ ] Question: 第二个 seg_clustering 分支（原始 `cluster_with_segmentation` 路径）是否确实可以作为 fallback 保留？如果确认，需要清理其中对 `cluster_with_segmentation` 的调用。
- [ ] Question: 后续是否要实现并发管线（ASR ∥ seg-3.0 并行跑）？架构改动较大，建议出设计案再动。

### Deferred Items

- 远场多通道 DER 优化（AISHELL-4 8ch → 选最佳通道融合）
- DER 全量 20 场评测（AliMeeting 全量，与现场确认后再跑）
- CER 全量 7176 条（用户明确暂不做）

## Important Context

- **当前代码有三处 seg_clustering 逻辑**：
  1. `_run_inference` 主分支（L587-646，调 `cluster_with_seamless_segmentation`）——**新无缝路径**
  2. `_run_inference` VAD fallback 分支（L708-732，调 `cluster_with_segmentation`）——**旧路径，保留做 fallback**
  3. HANDFF.md 里记录的旧逻辑（已过时，但仍作为历史参考）
- 无缝路径的 seg_type 标记传递链：`segmentation_engine.process_full_audio_seamless()` → `speaker_engine.cluster_with_seamless_segmentation()` → `asr_onnx_service._assign_clauses_to_speakers_seamless()`
- Cam++ embedding 只从 `seg_type == "single"` 的段提取，overlap/silence 段不参与聚类

## Context for Resuming Agent

### Important Context (detailed)

- **当前代码有三处 seg_clustering 逻辑**：
  1. `_run_inference` 主分支（L587-646，调 `cluster_with_seamless_segmentation`）——**新无缝路径**
  2. `_run_inference` VAD fallback 分支（L708-732，调 `cluster_with_segmentation`）——**旧路径，保留做 fallback**
  3. HANDFF.md 里记录的旧逻辑（已过时，但仍作为历史参考）
- 无缝路径的 seg_type 标记传递链：`segmentation_engine.process_full_audio_seamless()` → `speaker_engine.cluster_with_seamless_segmentation()` → `asr_onnx_service._assign_clauses_to_speakers_seamless()`
- Cam++ embedding 只从 `seg_type == "single"` 的段提取，overlap/silence 段不参与聚类

### Potential Gotchas

- `SpectralClustering` 需要 `n_clusters < n_samples`；当 single 段数量很少时（<2），会退化为全标 1
- `_assign_clauses_to_speakers_seamless` 的时间单位：seamless_segs 第三个元素如果是 int(speaker) 则是 ms，str(overlap/silence) 无单位概念
- 第二个 seg_clustering 分支（旧路径）运行后会返回结果但不执行 seamless 逻辑——如果主分支无缝路径失败，切到旧路径会获得旧版 DER 表现

## Environment State

### Tools/Services Used

- ASR 服务（asr_onnx_service.py）: 运行在 8002 端口（PID 23908），GPU 模式
- Python 环境: E:/conda/envs/asr_ui_env/python.exe
- 测试集路径: testset/ali_near_prep/ (R8002_M8002_mixed.wav)
- 上次 DER 评测终端依然处于有效状态

### Active Processes

- FastAPI 服务进程仍在运行（port 8002）
- 测试终端进程已完成评测

### Conda Environment

- 虚拟环境: E:\conda\envs\asr_ui_env
- 关键依赖: funasr, pyannote.audio==3.1.1, numpy==1.26.4, torch==2.3.1+cu121
- NumPy 版本锁定在 1.26.4（pyannote 的 np.NaN 兼容性）

## Related Resources

- `.superpowers/spec/2026-07-13-seamless-speaker-timeline.md` — 原设计方案
- `docs/superpowers/plans/2026-07-13-seamless-speaker-timeline.md` — 实施计划与任务拆解
- `testset/CER_DER_TEST_REPORT.md` — 上一轮评测报告
