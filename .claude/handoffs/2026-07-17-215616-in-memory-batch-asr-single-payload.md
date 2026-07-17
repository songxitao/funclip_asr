# Handoff: Qwen3-ASR 单次批量全内存去盘化重构 (v0.8.2 → v0.8.3)

## Session Metadata
- Created: 2026-07-17 21:56:16
- Project: E:\project\funclip-pro
- Branch: main
- Session duration: ~40 minutes (21:17 - 21:56)

### Recent Commits (for context)
  - 461cd22 feat: 修复集成客户端假死失效、集成宿主机 VAD 切割及 Qwen3-ASR vLLM 参数调优与共享卷优化
  - 42d4956 chore: v0.8.0 release preparation - impl CLI entrypoint, migrate eval, and fix README mermaid syntax
  - 450afce feat: CLI entrypoint + eval migration + test hardening
  - 6a156d1 chore: v0.8.0 release preparation — open-source infrastructure & test reorganization
  - 5316acd refactor(p1.5/p2/p3.1): app_control 离线归口、冗余文件清理、pyproject.toml 构建标准化

## Handoff Chain
- **Continues from**: [2026-07-17-204910-qwen-in-memory-stream-spec.md](./2026-07-17-204910-qwen-in-memory-stream-spec.md)
  - Previous title: Qwen3-ASR 内存字节流异步并发转写重构准备 (v0.8.1 -> v0.8.2)
- **Supersedes**: [HANDOFF-REVIEW.md](../../HANDOFF-REVIEW.md) — 旧异步流式方案已被新单次批量方案取代

---

## Current State Summary

本会话完成了 Qwen3-ASR 批量转写方案的**架构转变**：从"多路 aiohttp 异步并发流式"切换到"单次同步 Base64 批量 POST"。此前 grilling 会话达到了新共识（低耦合、无盘化批量重构），并落地了 `spec.md`（规格书）和 3 个垂直切片 Ticket。本会话派出子智能体实现了全部 3 个 Ticket 的代码变更，验证了语法正确性，并在用户启动 Docker 后端后运行了完整基准测试——**34min 长音频端到端 Pipeline 达到 71.9s / RTF 0.035 / 28.7 倍速**，超越旧基线（24.2x）15.8%，远优于旧异步方案（18.3x）。当前工作区有 5 个修改文件（含 HANDOFF.md），尚未 commit。

## Codebase Understanding

### Architecture Overview

Qwen3-ASR 是一个 Docker 容器化的 ASR 服务（`qwenllm/qwen3-asr:latest`），对外暴露 HTTP API 在 `127.0.0.1:28000`。客户端 SDK 在 `src/funclip_pro/core/asr.py` 的 `QwenEngine` 类中封装调用。

**新架构数据流：**
```
客户端 VAD 切片 (List[np.ndarray])
  ↓ io.BytesIO + sf.write → WAV bytes → base64.b64encode
  ↓ 单次 requests.post (sync)
/v1/audio/batch_transcriptions
  ↓ Data URL 直投（服务端已支持，零写盘）
vLLM 并行 Batch 推理
  ↓
返回带 timestamps 的转写结果
```

**关键改进：**
- 抛弃旧的异步流式（每个 chunk 一条 HTTP → `/v1/audio/transcribe_stream`，Semaphore=8、asyncio.gather）
- 改为 NumPy→WAV bytes→Base64 列表→单次 POST 到 `/v1/audio/batch_transcriptions`
- `return_timestamps=True` 硬编码，解决了旧方案时间戳为 0 的问题

### Critical Files

| File | Purpose | Relevance |
|------|---------|-----------|
| `src/funclip_pro/core/asr.py` | QwenEngine 批量接口 | **本次核心改动**：NumPy 分支从异步流式改为单次 Base64 批量 POST |
| `qwen_server/custom_server.py` | Docker 服务端批量端点 | 服务端 `batch_transcriptions` 的 Base64 分支已支持 Data URL 无盘化 |
| `src/funclip_pro/pipeline/offline.py` | 离线流水线编排 | 已采用纯内存路径，直接传 numpy 切片给 transcribe_batch |
| `tests/unit/test_qwen_vad_batch.py` | 单元测试 | 新增 `test_qwen_engine_transcribe_batch_numpy` 覆盖 Base64 批量路径 |
| `.scratch/in-memory-batch-asr/spec.md` | 规格书 | 本轮重构的设计文档和共识 |
| `.scratch/in-memory-batch-asr/issues/` | 3 个 Ticket 任务卡 | 垂直切片分解 |

## Work Completed

### Tasks Finished

- [x] **Ticket 01** (服务端)：`custom_server.py` 的 `batch_transcriptions` 已支持 Base64 内存直投（Data URL），无需修改
- [x] **Ticket 02** (客户端)：`QwenEngine.transcribe_batch` NumPy 路径从异步 aiohttp 流式替换为单次 sync Base64 批量 POST；删除 `_transcribe_single_chunk_async` 方法；移除 `aiohttp`/`asyncio` imports
- [x] **Ticket 03** (流水线+测试)：`offline.py` 无调试日志残留；新增 `test_qwen_engine_transcribe_batch_numpy` 测试
- [x] **基准测试**：34min 长音频端到端 **71.9s / RTF 0.035 / 28.7x** ✅、零写盘验证通过 ✅、GPU 显存稳定无 Swap ✅

### Benchmark Results

| 测试项 | 结果 |
|--------|------|
| 单文件 34min 直推 | 53.9s / RTF 0.026 (38.2x) |
| 端到端 Pipeline | **71.9s / RTF 0.035 (28.7x)** |
| 批量10 Base64 | 10.9s / RTF 0.293 |
| 批量10 共享卷 | 10.7s / RTF 0.285 |
| VAD 对齐片段 | **10945 段** ✅ |

**历史对比 (34min Pipeline)：**
| 版本 | 耗时 | RTF | 倍速 | vs 旧基线 |
|------|------|-----|------|-----------|
| v0.8.1 (共享卷直读) | 85.4s | 0.041 | 24.2x | — |
| v0.8.2 (异步流式) | 112.9s | 0.055 | 18.3x | ❌ -32% |
| **v0.8.3 (单次Batch)** | **71.9s** | **0.035** | **28.7x** | ✅ +15.8% |

### Files Modified

| File | Changes | Rationale |
|------|---------|-----------|
| `src/funclip_pro/core/asr.py` | NumPy 分支从异步 aiohttp 改为单次 sync Base64 批量 POST；删除 `_transcribe_single_chunk_async`；移除 `aiohttp`/`asyncio` imports | 消除多路 HTTP 开销，改用一次批量请求，解决 32% 性能倒退 |
| `tests/unit/test_qwen_vad_batch.py` | 新增 `test_qwen_engine_transcribe_batch_numpy` 测试 | 覆盖新 Base64 批量路径，验证 `return_timestamps=True`、WAV 头有效性 |
| `qwen_server/custom_server.py` | 此前已完成 Base64→Data URL 无盘化（非本次修改） | — |
| `HANDOFF.md` | 123 行变更 (前序 handoff，非本次) | — |

### Decisions Made

| Decision | Options Considered | Rationale |
|----------|-------------------|-----------|
| 用单次 sync `requests.post` 代替旧 `aiohttp` 异步并发 | ① 异步 aiohttp (旧) ② 单次 sync POST | ⑨ 服务端 `batch_transcriptions` 已支持批量推理；消除多路 HTTP 开销和事件循环阻塞问题；同步接口更简单可靠 |
| 删掉 `_transcribe_single_chunk_async` | ① 保留但废弃 ② 彻底删除 | 代码无引用，保留增加维护成本 |
| `return_timestamps=True` 硬编码在客户端 | ① 作为参数透传 ② 硬编码 | 离线批处理场景始终需要时间戳，简化接口 |
| NumPy→WAV→Base64 在客户端执行 | ① 服务端解码 ② 客户端编码 | 客户端已有 `soundfile` 依赖；`io.BytesIO` 零写盘 |

## Pending Work

## Immediate Next Steps

1. **提交代码 (commit)**：当前工作区有 5 个修改文件，建议做一次 commit，描述本次重构
2. **更新 README.md / CHANGELOG.md**：记录 v0.8.3 的架构变更和性能提升数据
3. **检查旧 `.scratch/in-memory-asr/` 目录**：旧异步流式方案的 spec 和 4 个 Ticket 已过时，考虑标记废弃或删除

### Blockers/Open Questions

- [ ] 单文件 `/v1/audio/transcriptions` 端点（`custom_server.py:402-460`）仍在使用临时文件写盘，若需全链路零写盘需要重构此端点
- [ ] 端到端 Pipeline 的 10945 段 VAD 对齐虽然工作，但未知是否有过分割问题

### Deferred Items

- 实时 WebSocket `/ws/asr` 端点重构（spec 明确 Out of Scope）
- Gradio UI 和界面状态修改（Out of Scope）
- WSL 2 物理内存内核机制调整（Out of Scope）

## Context for Resuming Agent

## Important Context

1. **服务端已支持 Base64 无盘化**：`custom_server.py` 的 `batch_transcriptions` 在接收 Base64 时直接走 `data:audio/wav;base64,...` 路径，`is_direct_path=True` 跳过文件清理，无需额外改动
2. **Docker 容器约束**：`qwen3-asr` 容器已映射端口 `28000:80`，`GPU_MEMORY_UTILIZATION=0.70`、`MAX_MODEL_LEN=4096`、`max_num_seqs=8` 不可随意调整
3. **客户端新路径**：`QwenEngine.transcribe_batch` 现在有三个分支：
   - `List[np.ndarray]` → 新 Base64 单次批量 POST ✅
   - `List[str]` 且在共享卷内 → 共享卷直读模式
   - `List[str]` 不在共享卷 → 文件 Base64 批量模式
4. **旧异步代码已彻底删除**：`_transcribe_single_chunk_async`、`_async_run()`、`aiohttp` import 全部移除

### Potential Gotchas

- **GPU 显存红线**：RTX 4080 Laptop 12GB，`GPU_MEMORY_UTILIZATION=0.70`、`max_num_seqs=8` 是压测后的最优参数，调大立刻触发 PCIe Swap 导致性能暴跌
- **Qwen 引擎与 VAD 底座模型都是独立的**：OfflinePipeline 加载 VAD 模型时可能会打印 deprecation warning，不影响运行
- **测试运行**：需要 Docker 后端运行；单测需要 `asr_ui_env` conda 环境（路径 `E:\conda\envs\asr_ui_env\python.exe`）

### Environment State

- Docker 容器 `qwen3-asr` 运行中，`127.0.0.1:28000` 映射正常
- GPU: ~11.6GB/12.3GB 显存使用，无 Swap
- 测试文件位于 `testset/` 目录：短音频 BAC*、长音频 `R8002_M8002_mixed.wav` (34min)

## Related Resources

- 规格书：[.scratch/in-memory-batch-asr/spec.md](../../.scratch/in-memory-batch-asr/spec.md)
- Ticket 01：[.scratch/in-memory-batch-asr/issues/01-server-in-memory-batch.md](../../.scratch/in-memory-batch-asr/issues/01-server-in-memory-batch.md)
- Ticket 02：[.scratch/in-memory-batch-asr/issues/02-client-in-memory-batch.md](../../.scratch/in-memory-batch-asr/issues/02-client-in-memory-batch.md)
- Ticket 03：[.scratch/in-memory-batch-asr/issues/03-pipeline-and-test-alignment.md](../../.scratch/in-memory-batch-asr/issues/03-pipeline-and-test-alignment.md)
- 旧设计（已废弃）：[.scratch/in-memory-asr/](../../.scratch/in-memory-asr/)
- 旧方案审查报告：[HANDOFF-REVIEW.md](../../HANDOFF-REVIEW.md)
- 基准脚本：`benchmark_qwen_asr.py`

---

**Security Reminder**: Before finalizing, run `validate_handoff.py` to check for accidental secret exposure.
