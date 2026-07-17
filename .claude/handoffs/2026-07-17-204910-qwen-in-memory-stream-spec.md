# Handoff: Qwen3-ASR 内存字节流异步并发转写重构准备 (v0.8.1 -> v0.8.2)

## Session Metadata
- Created: 2026-07-17 20:49:00
- Project: E:\project\funclip-pro
- Branch: main
- Session duration: 40 minutes

### Recent Commits (for context)
  - 461cd22 feat: 修复集成客户端假死失效、集成宿主机 VAD 切割及 Qwen3-ASR vLLM 参数调优与共享卷优化
  - 42d4956 chore: v0.8.0 release preparation - impl CLI entrypoint, migrate eval, and fix README mermaid syntax

## Handoff Chain
- **Continues from**: [2026-07-17-201557-qwen-batch-throughput-tuning.md](file:///E:/project/funclip-pro/.claude/handoffs/2026-07-17-201557-qwen-batch-throughput-tuning.md)
  - Previous title: Qwen3-ASR 离线批量推理吞吐量优化（实现66倍速）
- **Supersedes**: None

---

## Current State Summary

1. **已实现的里程碑调优 (v0.8.1)**:
   * 修复并解决了 Qwen3-ASR 在集成客户端下假死失效的问题。
   * 在宿主机 SDK 离线流水线（`OfflinePipeline`）中集成 VAD 音频切割。
   * 调优 vLLM 核心参数解决 Token 溢出报错（`MAX_MODEL_LEN=4096`），微调并发 `max_num_seqs=8` 配合 `GPU_MEMORY_UTILIZATION=0.70`，成功让出 `2.4GB` 显存红线，解决由于显存不足发生 Swap 换页卡死问题。
   * 容器启动硬限制为 `--cpus 8`，彻底降噪 150+ 线程上下文切换带来的 CPU 锯齿过载。
   * **压测成绩**：34.3分钟长音频（R8002_M8002_mixed.wav）端到端 VAD 批量对齐转写仅需 **85.373秒**，RTF 达 **0.041**（**24.2 倍速**）！共享卷极速直读相比 Base64 节省了 **51.5%** 的批量传输时间。
2. **已归档的技术设计与任务卡 (v0.8.2 准备阶段)**:
   * 编写并落盘了 [spec.md](file:///E:/project/funclip-pro/.scratch/in-memory-asr/spec.md)（纯内存字节流异步并发转写技术规格）。
   * 拆解并创建了 4 个 tracer-bullet 本地 Ticket 任务卡（在 [issues/ 目录](file:///E:/project/funclip-pro/.scratch/in-memory-asr/issues/)）。

---

## Guide for Resuming Agent (接手指南)

下一个接管的智能体必须围绕 `.scratch/in-memory-asr/issues/` 中的 4 个 Tickets 顺次执行重构开发，核心任务是**彻底消除音频切片存盘行为，改为内存字节流异步并发交互**。

### 1. 核心开发步骤 (Frontier)
- **第一步：主攻 Ticket 01**：
  * 修改 `qwen_server/custom_server.py`，添加 `POST /v1/audio/transcribe_stream`。
  * 接收表单参数和 `UploadFile` 二进制流，直接用 `soundfile.read(io.BytesIO(await file.read()))` 转换，不要写盘，推理后返回。
- **第二步：推进 Ticket 02 & 03**：
  * 在 SDK 端的 `asr.py` 的 `QwenEngine` 内实现私有异步方法 `_transcribe_single_chunk_async`。使用 `io.BytesIO` 将 numpy 片段格式化为标准 WAV bytes 并用 `aiohttp` 异步推给 `/transcribe_stream`。
  * 在 `QwenEngine.transcribe_batch` 内引入最大限制为 8 的并发信号量，通过 `asyncio.run` 同步等待并收集 list 结果。
- **第三步：收口 Ticket 04**：
  * 重构 `OfflinePipeline.run`，删掉 `tempfile.mkstemp`、`sf.write` 及文件清理的写盘代码，直接把 NumPy 的 `y[s_idx:e_idx]` slice 列表提给 `transcribe_batch` 即可。

### 2. 如何测试与验证

- **后端单接口验证 (Ticket 01 验证)**：
  在宿主机写一个 scratch 脚本，在内存中把一段 wav 二进制字节直接用 requests 发送给 `/v1/audio/transcribe_stream`，断言其正确返回 transcription 的 JSON 内容且容器 `/tmp` 下没有新增文件。
- **单元测试回归 (Ticket 04 验证)**：
  在宿主机运行本地单元测试，确保 VAD 的切割在 mock 状态下依然能够精准完成绝对时间戳对齐：
  `$env:PYTHONPATH="E:\project\funclip-pro\src"; E:\conda\envs\asr_ui_env\python.exe -m pytest tests/unit/test_qwen_vad_batch.py`
- **一键性能基准测试与零写盘验证 (Ticket 04 验证)**：
  在宿主机运行 `run_bench.bat`，观察端到端的 34 分钟长音频测试是否成功跑通。
  * **验证标准**：核对整个转写运行期间，宿主机临时目录中**没有产生过任何 `qwen_chunk_*.wav` 音频文件**，证明磁盘 I/O 写入次数已被彻底压缩为 0，完全基于 RAM 执行。

---

## Context & Potential Gotchas (避坑指南)

1. **显存红线控制**：
   RTX 4080 Laptop 卡只有 12GB 显存，且宿主机常驻着 PID 为 `20372` 的 Gradio 后台程序（`app_control.py`，占了约 3GB 显存）。
   * **切忌**：随意把 `GPU_MEMORY_UTILIZATION` 调回 `0.85` 或随意将 `max_num_seqs` 设过大，否则会瞬间触发跨 PCIe 显存 Swap 导致性能暴跌或触发 vLLM OOM 崩溃。
2. **宿主机 RAM 回收**：
   如果测试和开发期间发现宿主机 32GB RAM 几乎吃满（95% 以上），请通知人类或代劳执行 `docker stop qwen3-asr; docker rm qwen3-asr;`，随后在终端运行 `wsl --shutdown` 回收 30GB 的虚拟缓存，然后再拉起 Docker 容器即可。
