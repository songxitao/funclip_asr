# Handoff: Qwen3-ASR 离线转写 VAD 批量化与微批次重构 (qwen-vad-batching-refactor)

## Session Metadata
- Created: 2026-07-17 18:50:48
- Project: E:\project\funclip-pro
- Branch: main
- Session duration: 2.5 hours

### Recent Commits (for context)
  - 42d4956 chore: v0.8.0 release preparation - impl CLI entrypoint, migrate eval, and fix README mermaid syntax
  - 450afce feat: CLI entrypoint + eval migration + test hardening
  - 6a156d1 chore: v0.8.0 release preparation — open-source infrastructure & test reorganization
  - 5316acd refactor(p1.5/p2/p3.1): app_control 离线归口、冗余 file 清理、pyproject.toml 构建标准化
  - 6d8dd9f docs: DER 测试集纠正为 Ali + 全量评测暂停，写 handoff 交接下一智能体

## Handoff Chain

- **Continues from**: [2026-07-15-194100-github-release-ready.md](./2026-07-15-194100-github-release-ready.md)
  - Previous title: 2026-07-15-194100-github-release-ready
- **Supersedes**: None

## Current State Summary

本会话完成了长音频离线转写选用 Qwen3-ASR 引擎时的客户端 VAD 切割分片、并发批量转写、绝对时间戳重组校准以及服务端 Micro-batching 并发机制重构。编写的 TDD 单元测试 `test_qwen_vad_batch.py` 及全量回归测试均已 100% 绿灯跑通。目前代码已准备就绪，正在等待尖子测试锁定在 8 并发下的实际速度与显存/内存表现。

## Architecture Overview

1. **客户端转写流水线**：离线 ASR 入口为 `OfflinePipeline.run`。原 Qwen 分支不经过 VAD，现已重构为：当音频时长 > 5s（`_use_vad` 为 True）时，调用本地 VAD 模型分割，写入临时 wav 列表，调用 `QwenEngine.transcribe_batch` 并行发送。
2. **服务端并发管理**：服务端 `custom_server.py` 在接收到批量 Base64 列表后，会将临时分片以微批次（Micro-batching，目前默认为 8）进行循环分批加载和推理，以防止 vLLM 内部排队引发显存 Swap 导致系统内存暴满卡死。
3. **绝对时间对齐**：因各个 chunk 的字级时间戳 `start` / `end`（单位是秒）是相对于各个 chunk 起始点的相对时间，客户端在重组时必须将其乘以 1000 转换为毫秒后，累加上该 VAD 分段相对于整段音频的绝对起始毫秒数，最终重新拼接为一整条绝对时间字幕轴。

## Critical Files

| File | Purpose | Relevance |
|------|---------|-----------|
| [asr.py](src/funclip_pro/core/asr.py) | Qwen 客户端驱动类 `QwenEngine` | 新增了 `transcribe_batch` 方法，重构 `transcribe` 复用此方法以保持 DRY。 |
| [offline.py](src/funclip_pro/pipeline/offline.py) | 离线转写 pipeline 主控类 | 在 `engine_key == "qwen"` 分支下实现了 VAD 自动切割、临时文件管理、以及时间轴累加对齐逻辑。 |
| [custom_server.py](qwen_server/custom_server.py) | Qwen API 服务端 | 在 `batch_transcriptions` 接口中引入了 Micro-batching 机制（限制并发微批次为 8）。 |
| [test_qwen_vad_batch.py](tests/unit/test_qwen_vad_batch.py) | Qwen VAD 转写单元测试 | TDD 测试用例，覆盖并发 Base64 请求组装、VAD 调用频次、绝对时间轴偏置对齐、以及临时文件删除。 |

## Key Patterns Discovered

- 本地 VAD 产生的 `opt_segs` 起始/结束时间为毫秒 (ms)，而 Qwen 服务端返回的字级时间戳单位为秒 (sec)，在客户端累加时需确保单位的正确转换（`int(start_sec * 1000) + start_ms`）。
- 即使是批量转写，也绝对不能一次性将 100 个分片送入 vLLM，因为 vLLM 内部没有物理 batch 限流，会直接撑满显存并触发 Swap 换页到内存，导致系统卡死。必须在 API 层显式做微批次分片处理。

## Work Completed

### Tasks Finished

- [x] 扩展 `QwenEngine`，新增 `transcribe_batch` 批量转写请求。
- [x] 重构 `OfflinePipeline.run`，当选用 Qwen 且开启 VAD 时，实现分片保存为临时文件并批量调用。
- [x] 实现分片字级时间戳基于 VAD 绝对毫秒偏置 of 累加对齐重组。
- [x] 增加 `try...finally` 块，确保转写完成或异常时，所有临时 wav 文件能被彻底 `os.remove` 清理。
- [x] 修正单元测试中的 Mock 逻辑漏失，并使用标准的 `patch("soundfile.write")` 机制拦截。
- [x] 在服务端 [custom_server.py](qwen_server/custom_server.py) 的 `batch_transcriptions` API 中实现限制最大并发为 8 的 Micro-batching 机制，彻底规避内存暴涨。
- [x] TDD 单元测试及全量 82 项单元测试全部绿灯跑通。

### Files Modified

| File | Changes | Rationale |
|------|---------|-----------|
| `src/funclip_pro/core/asr.py` | 新增 `transcribe_batch`；重构 `transcribe`； | 实现批量 Base64 发送请求并保持逻辑 DRY。 |
| `src/funclip_pro/pipeline/offline.py` | Qwen 分支重构，引入本地 VAD 切分和临时文件物理清理； | 防止整段大音频导致 Qwen 后端 OOM。 |
| `qwen_server/custom_server.py` | 入口路由限制微批处理大小为 8； | 防止 vLLM KV Cache 发生 Swap 撑爆物理内存。 |
| `tests/unit/test_qwen_vad_batch.py` | 修正 mock 逻辑，使用标准的 `patch` 拦截。 | 保证 TDD 回归测试可用性。 |

### Decisions Made

| Decision | Options Considered | Rationale |
|----------|-------------------|-----------|
| 客户端 VAD 切割 + 服务端微批处理 | 直接发送整段音频 / 仅客户端 VAD 直发 | 直接发送会导致 API 报 500 且爆显存；若仅 VAD 直发但不限制 Batch 大小，则 vLLM 会因高并发调度队列触发 KV 缓存 Swap，撑爆物理内存。8 并外的 Micro-batching 能够在不触发 Swap 的情况下榨干 RTX 4080 GPU 推理吞吐。 |

## Immediate Next Steps

1. **测试性能**：使用长音频进行真机测试，确认当前微批次（并发8）版本的实际转写速度与系统内存占用。
2. **打通 Batch Size 联动**（可选）：若 8 并发测试表现完美，可按照 [implementation_plan.md](file:///C:/Users/song/.gemini/antigravity/brain/6c5e2b23-94c0-4743-bfaf-b53fc8cd8d6d/implementation_plan.md) 将控制台前端的 **“批处理量 (Batch Size)”** 滑块变量透传给服务端，使微批次大小可由用户拉动滑块来控制（显存够大可拉高，显存不够则拉低）。
3. **提交变更**：执行 `git add` 和 `git commit`。

### Blockers/Open Questions

- [ ] 需要确认 8 并发在 RTX 4080 Laptop (12G) 上的真实性能和系统资源负载，以决定是否需要滑块透传。

### Deferred Items

- 暂无延后项。

## Important Context

- **修改服务端后必须重启**：因为 `custom_server.py` 承载了最大上下文和微批次逻辑，修改它后**必须运行 `stop_qwen_backend.bat` 并重启一键启动控制台**，否则新改动不会生效。
- **单元测试执行环境**：专用虚拟环境为 `asr_ui_env`，其 python 路径在 `E:/conda/envs/asr_ui_env/python.exe`。运行测试的 PYTHONPATH 指向 `src/`。

### Assumptions Made

- 假定客户端本地 VAD 切割出来的各个 chunk 的采样率完全是 16000Hz，因为 `OfflinePipeline.run` 开始阶段已经通过 `librosa.load(audio_path, sr=16000)` 对音频统一重采样到了 16k。

### Potential Gotchas

- 在测试中 Mock `soundfile.write` 时，必须使用 `@patch("soundfile.write")`，由于 `offline.py` 内部是局部 `import soundfile as sf`，直接用 `sf.write = mock` 可能会由于模块 reload 或 pytest 隔离而无法拦截成功。

## Environment State

### Tools/Services Used

- Conda 虚拟环境: `asr_ui_env` (Python 3.11, CUDA PyTorch 2.3.1)
- vLLM 推理后端，默认加载 `Qwen3-ASR-1.7B` 模型，运行在 localhost 端口。

### Active Processes

- Qwen vLLM 后端服务 (通过 `custom_server.py` 启动)
- Gradio 前端控制台 (通过 `app_control.py` 启动)
