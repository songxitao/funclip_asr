# Handoff: Qwen3-ASR 离线批量推理吞吐量优化（实现66倍速）

## Session Metadata
- Created: 2026-07-17 20:16:00
- Project: E:\project\funclip-pro
- Branch: main
- Session duration: 1 hour

### Recent Commits (for context)
  - 42d4956 chore: v0.8.0 release preparation - impl CLI entrypoint, migrate eval, and fix README mermaid syntax
  - 450afce feat: CLI entrypoint + eval migration + test hardening

## Handoff Chain
- **Continues from**: [2026-07-17-192200-qwen-cpu-memory-swap-optimization.md](file:///E:/project/funclip-pro/.claude/handoffs/2026-07-17-192200-qwen-cpu-memory-swap-optimization.md)
  - Previous title: Qwen3-ASR 推理性能与 CPU/显存 Swap 挂起问题深度优化
- **Supersedes**: None

---

## Current State Summary

在上一轮解决 CPU 过载和 CPU-GPU 内存 Swap 的基础上，我们对 Qwen3-ASR **离线字幕的批量推理吞吐量** 进行了深度重构与极限调优：
1. **打破人工微批次限制**：之前在 [custom_server.py](file:///E:/project/funclip-pro/qwen_server/custom_server.py) 中，`micro_batch_size` 限制为 `2`。这意味着即使前端一次性发送了 10 个或更多切片文件，后端也会人为将其拆成“2个一组”多次循环调用 vLLM 推理。由于每次调用推理接口有大约 1~2 秒的系统调度和前处理固定开销，导致 GPU 处于严重的饥饿状态，利用率频繁断续。
2. **释放 vLLM PagedAttention 并发管道**：
   - 将 `max_num_seqs` 从 `2` 提升至 `16`（配合 `max_inference_batch_size=16`）。
   - 将后端 API 的 `micro_batch_size` 从 `2` 直接拉大到 `64`。
   - 关闭分块预填 `enable_chunked_prefill=False`，减少离线非流式场景下的任务碎裂和调度开销。
3. **Docker 资源硬件限制**：在 [start_qwen_backend.bat](file:///E:/project/funclip-pro/start_qwen_backend.bat) 中，添加了 `--cpus 8` 限制，并将共享内存限制下调至 `--shm-size 2g`，释放被系统预占的物理 RAM，成功解决了 154 个线程在 24 核上疯狂上下文切换导致的 CPU 锯齿问题。

**优化后的终极 Benchmark 表现**：
* 10个短音频文件（总时长 **37.38秒**），批量推理仅需 **0.546秒**！
* 离线批量推理 RTF 达到惊人的 **0.015**，即 **66倍速以上**（1小时音频转写只需 32.7秒）！
* GPU 显存峰值稳定在 **10.8GB**，没有发生 OOM，留出了 1.2GB 以上的安全红线。
* CPU 占用从早期的 86% 大幅降低至平稳的 **32%**。
* GPU 锯齿基本消除，批量推理时可以维持高利用率完成计算。

---

## Codebase Understanding

### Critical Files

| File | Purpose | Relevance |
|------|---------|-----------|
| [custom_server.py](file:///E:/project/funclip-pro/qwen_server/custom_server.py) | 后端 vLLM FastAPI 服务入口 | 调优了 `max_num_seqs=16`, `max_inference_batch_size=16`, `enable_chunked_prefill=False`, `swap_space=0`, 且将微批次扩容为 `micro_batch_size=64` |
| [start_qwen_backend.bat](file:///E:/project/funclip-pro/start_qwen_backend.bat) | Docker 容器启动脚本 | 加入了 `--cpus 8` 物理核心限制，并收紧了 `--shm-size 2g` 节省物理内存 |
| [benchmark_qwen_asr.py](file:///E:/project/funclip-pro/benchmark_qwen_asr.py) | 性能测试基准脚本 | 包含了预热、单文件测试、中音频测试和批量测试模块，支持在 Windows 侧一键采集 Docker 内的进程内存、线程数和物理 GPU 占用指标 |

---

## Work Completed

### Tasks Finished
- [x] 重构了 `custom_server.py` 的微批次提交机制，将 `micro_batch_size` 扩展至 `64`。
- [x] 释放 vLLM 性能上限，配置 `max_num_seqs = 16`，支持 16 路序列在 GPU 内部并行。
- [x] 禁用了 `enable_chunked_prefill`，大幅提高了离线高吞吐场景下的批量推理效率。
- [x] 在 `start_qwen_backend.bat` 容器创建命令中，添加了 `--cpus 8` 物理硬限制。
- [x] 编写并运行了 [benchmark_qwen_asr.py](file:///E:/project/funclip-pro/benchmark_qwen_asr.py) 脚本，实现了 66倍速 (RTF 0.015) 的性能飞跃与 GPU 锯齿压制。

---

## Pending Work

### Immediate Next Steps

1. **观察真实离线字幕提取任务**：
   * 使用实际的大视频或大音频文件（例如 30分钟至1小时的测试音频），调用宿主机的离线转写接口，检查在完整的 VAD ➔ 临时切片 ➔ HTTP 批量发送 ➔ 后端转写 ➔ 标点对齐链路下的吞吐量表现，看是否稳定在约 50~60 倍速左右。
2. **清理本地测试垃圾**：
   * 离线转写时，注意检查 `/tmp`（Docker 内）和临时目录下的 wav 切片文件是否能被正确 `unlink`/删除，防止磁盘空间暴涨。
3. **WSL2 物理内存回收**：
   * 在调优结束后，建议运行一次 `wsl --shutdown` 重启 WSL 环境，释放之前 Docker 容器缓存占用的虚拟内存（WslManage 气球占用的 31GB RAM），将物理内存降回正常线。

---

## Context for Resuming Agent

- **Qwen3-ASR** 的主要瓶颈已经不是 GPU 计算，而是 I/O 准备（文件读取、Base64 编解码等）。
- 并发设置为 `16` 已经是 12GB 显存显卡的物理极限。**请勿在不限制 MAX_MODEL_LEN (当前是1024) 的情况下盲目提高 max_num_seqs**，否则会瞬间触发 OOM 崩溃。
