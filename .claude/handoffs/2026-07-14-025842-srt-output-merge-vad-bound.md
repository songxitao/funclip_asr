# Handoff: SRT 输出 + 相邻同说话人合并 + VAD 段内合并限制

## Session Metadata
- Created: 2026-07-14 02:58
- Project: E:\project\funclip-pro
- Branch: main
- Session duration: 约 3 小时（含并行管线实验 + 回退 + SRT/合并功能）
- Continues from: HANDOFF.md (2026-07-13 23:28, 无缝说话人时间轴 seg 丢弃段回收与子句级说话人融合)

### Recent Commits (for context)
  - 4e3d1d1 fix: 限制同说话人合并在 VAD 段内，不再跨段合并
  - 3d46c15 Revert "experiment: seg_cut_asr"
  - 1181e0a feat: diarized_text 和 JSON segments 也应用相邻同说话人合并
  - 3b5cf43 feat: 新增 SRT 字幕输出格式 + 相邻同说话人段合并
  - c9dd659 Revert "perf: seg_clustering 分支并行管线"

## Handoff Chain
- **Continues from**: HANDOFF.md (project root, seamless speaker timeline handoff)
- **Supersedes**: HANDOFF.md（上次的 HANDOFF.md 已过期，此文档为最新会话记录）

## Current State Summary

本会话完成了三项主要工作：
1. **并行管线实验 & 回退**：实现 seg_clustering 的 ThreadPoolExecutor 并行化，实测因 GPU 争抢反而更慢，已 revert
2. **SRT 字幕输出 + 相邻同说话人合并**：新增 SRT 格式输出和跨段合并功能，实测通过
3. **VAD 段内合并限制**：将合并限制在 VAD 段内部，防止解说与角色跨段混淆。DER 评测 14.85%（对比旧 14.54%，无实质变化）

## Codebase Understanding

### Architecture Overview

当前 seg_clustering 主流线（无缝时间轴）：
1. seg-3.0 → 无缝时间轴（保留 single/overlap/silence 段）
2. Cam++ → 只对 single 段提 embedding
3. SpectralClustering 全局聚类
4. VAD → ASR 解码（串行）
5. _assign_clauses_to_speakers_seamless：子句级锚点扩散分配
6. 每个 VAD 段内合并相邻同说话人子句（限制跨段）

### Critical Files

| File | Purpose | Relevance |
|------|---------|-----------|
| `asr_onnx_service.py` | 主服务，_run_inference + /transcribe 端点 | 本会话所有改动 |
| `cli_transcribe.py` | 命令行客户端 | 新增 --format 参数 |
| `speaker_engine.py` | Cam++ 说话人模型 + 聚类 | 无改动 |

### Key Patterns Discovered

- **GPU 争抢问题**：ThreadPoolExecutor 并行 seg-3.0 + ASR 反而更慢，因为两者都用 GPU，CUDA 串行化执行
- **45454 交替根因**：seg-3.0 帧级输出碎段，子句按字数比例估算时间位置有偏差。改分配策略（seg_cut_asr）也无效，需要字级别时间戳或 seg-3.0 输出平滑
- **VAD 段内合并**：跨 VAD 段合并会把解说和角色内容拼在一起，限制段内合并可避免

## Work Completed

### Tasks Finished

- [x] 并行管线实施（ThreadPoolExecutor）→ 实测更慢，已 revert
- [x] SRT 字幕输出格式：_ms_to_srt / _segments_to_srt / 端点 response_format=srt
- [x] 相邻同说话人合并：_merge_same_speaker_segments，JSON/text/SRT 统一受益
- [x] VAD 段内合并限制：合并挪到循环内，防止跨段混淆
- [x] seg_cut_asr 实验：先分说话人再逐段 ASR → 效果不佳，已 revert
- [x] AliMeeting DER 评测：seg_clustering 14.85%（对比旧 14.54%）

### Files Modified

| File | Changes | Rationale |
|------|---------|-----------|
| `asr_onnx_service.py` | +SRT 工具函数 + merge + VAD 段内合并 + seg_cut_asr(已 revert) | 主功能改动 |
| `cli_transcribe.py` | +--format {json,text,srt} | CLI 支持新格式 |

### Decisions Made

| Decision | Options Considered | Rationale |
|----------|-------------------|-----------|
| 回退并行管线 | 保留 vs 回退 | 实测更慢（GPU 争抢），不适用 |
| 回退 seg_cut_asr | 保留 vs 回退 | 段太大含多个说话人，效果不如 seg_clustering |
| VAD 段内合并 | 全局合并 vs 段内合并 | 防止解说与角色跨段混淆 |
| 全文统一加标点 | 逐段加 vs 全文统一 | 全文加标点有完整上下文，标点更准 |

## Pending Work

### Immediate Next Steps

1. **P0 — 置信度加权的锚点扩散**：当前硬分配（子句全归重叠最多的确定段），改为按重叠比例加权，低置信度时不硬分配。预期 CONF 从 13.6% → 8%
2. **P0 — 字级别时间戳**：如果 SenseVoice 能输出每个字的起止时间，逐个字分配说话人可彻底解决 45454 交替
3. **P1 — seg-3.0 输出时序平滑**：短时间（<500ms）内交替归属不同人的段，强制合并或标未知

### Blockers/Open Questions

- [ ] 45454 交替问题的根本解决需要字级别时间戳，当前 seg-3.0 帧级碎段无法在分配层面完全解决
- [ ] 并行提速不可行（GPU 争抢），如有需要可考虑 ASR 切 CPU（sherpa）与 seg-3.0（GPU）分硬件事跑

### Deferred Items

- 远场多通道 DER 优化（AISHELL-4 8ch → 选最佳通道融合）
- DER 全量 20 场评测（AliMeeting 全量）

## Context for Resuming Agent

### Important Context

- **当前 seg_clustering 不走并行**，已 revert 串行（是好事——串行更快）
- **合并限制在 VAD 段内**：_merge_same_speaker_segments 在循环内每个 VAD 段处理完后独立调用，不再全局调用
- **_assign_clauses_to_speakers_seamless 内部也有一次 merge**（L553-567），外部 merge 是冗余的但无害
- **45454 交替**不是合并问题，是 seg-3.0 帧级输出碎 + 子句按字数估算位置有偏差导致的
- **SRT 输出**：`response_format=srt` 或 `--format srt`
- **Der 评测命令**：`E:/conda/envs/asr_ui_env/python.exe ali_der_eval.py R8002_M8002`（默认 spectral，需要改 strategy）

### Potential Gotchas

- `_post_punc` 需要完整句子上下文，逐段加标点效果差。seg_clustering 路径的标点在全文拼接后统一加
- `GPU_SEMAPHORE = asyncio.Semaphore(3)` 限制并发请求，但实际的推理线程受 GIL 影响
- DER 评测时 hyp 段数会因 SpectralClustering 的随机性波动（+/- 0.3% 正常范围）

## Environment State

### Tools/Services Used

- ASR 服务: asr_onnx_service.py, port 8002, GPU 模式
- Python: E:/conda/envs/asr_ui_env/python.exe
- 测试集: testset/ali_near_prep/ (R8002_M8002)
- CLI: cli_transcribe.py
- Conda 环境: asr_ui_env

## Related Resources

- `.superpowers/spec/2026-07-13-seamless-speaker-timeline.md` — 无缝时间轴原设计
- `.superpowers/spec/2026-07-13-parallel-pipeline.md` — 并行管线设计（已证实不可行）
- `testset/CER_DER_TEST_REPORT.md` — 历史评测报告
