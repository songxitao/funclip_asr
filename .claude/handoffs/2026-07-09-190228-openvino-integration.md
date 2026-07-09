# Handoff: SenseVoiceSmall OpenVINO 引擎集成与 CPU 推理优化

## Session Metadata
- Created: 2026-07-09 19:02:28
- Project: E:\project\funclip-pro
- Branch: main
- Session duration: 约 1.5 小时

### Recent Commits (for context)
  - e448dd9 perf: compile other performance comparison scripts and baseline configs into main
  - 8817c89 feat: replace ONNX Runtime with native OpenVINO Core engine
  - 11d0b27 fix: adjust mock feature dimension to 560 in openvino speed test
  - 5798fc3 feat: add psutil physical cpu affinity binding and thread limits
  - 3d8c087 docs: add OpenVINO integration and CPU optimization implementation plan

## Handoff Chain

- **Continues from**: [2026-07-04-204929-git-init-and-refactoring-roadmap.md](./2026-07-04-204929-git-init-and-refactoring-roadmap.md)
  - Previous title: 2026-07-04-204929-git-init-and-refactoring-roadmap
- **Supersedes**: None

## Current State Summary

本会话完成了在 `funclip-pro` ASR 微服务中集成 Intel OpenVINO 引擎以替换 ONNX Runtime 默认 CPU 执行层的优化工作。
1. 在 `asr_onnx_service.py` 头部通过 `psutil` 强制绑定了 CPU 的前 6 个大核心物理核，限制了多线程计算库的环境变量软防线（均为6），杜绝了超额调度开销。
2. 修复了跑评中由于低帧率拼接（LFR=7）导致 80/560 维度不匹配报错的问题。
3. 重构了 `SenseVoiceSmall` 包装类，使用 `openvino.Core` 原生 API 代替 ONNX Runtime 载入并即时编译模型。
4. 跑通了大 Batch（16）混合 VAD 的长音频（85切片）最终 CPU A/B 对照跑评，微服务端到端耗时由 36.85 秒缩短至 30.96 秒（性能提速近 20%），去标点文字吻合度达到 98.76%。目前工作区处于干净状态。

## Codebase Understanding

### Architecture Overview

该服务是一个拼装式的 FastAPI 接口。音频分段由 VAD-FSMN（PyTorch-CPU）决定，分段特征提取使用 WavFrontend，接着由 ASR 模型（OpenVINO 核心）推理，最后由 CT-Punc 标点模型（PyTorch-CPU）做标点后处理。

### Critical Files

| File | Purpose | Relevance |
|------|---------|-----------|
| `asr_onnx_service.py` | ONNX/OpenVINO ASR 微服务核心入口 | 绑核、限制计算线程、OpenVINO Core 重构及推理核心所在地。 |
| `tests/test_openvino_speed.py` | OpenVINO vs ORT 纯推理时延跑评 | 模拟数据下推理引擎前向耗时基准评估。 |
| `tests/test_asr_comparison_cpu.py` | PyTorch vs ONNX/OpenVINO 微服务端到端 A/B 测试 | 长音频实际批处理吞吐基准性能测试。 |
| `tests/test_onnx_decode_refactor.py` | NumPy 批量解码等价性校验 | 验证 numpy decode 替换原 unique_consecutive 解码的一致性。 |
| `tests/test_affinity.py` | 核心物理绑定状态校验 | 自动化测试进程 CPU 亲和性硬绑定。 |

### Key Patterns Discovered

*   `model_quant.onnx` 包含了大量显式 ONNX 量化节点（QuantizeLinear/DequantizeLinear）。通用的执行框架（OpenVINO/ONNX Runtime）由于无法对这类破碎的显式量化图进行 Attention 算子融合，使得前向推理时频繁产生反量化开销（Dequantize Overhead），导致在 CPU 推理上性能甚至不敌 PyTorch 原生 FP32 (oneDNN 汇编级 JIT 算子融合)。
*   但在大 Batch（16）高负载吞吐时，OpenVINO 原生 Core C++ 下多流调度与物理绑核的资源红利最终使得端到端性能反超了 ORT-CPU。

## Work Completed

### Tasks Finished

- [x] 物理核心硬绑定与 CPU 多线程环境变量限制 (Task 1)
- [x] 跑评时延维度修复与 OpenVINO 时延基准跑通 (Task 2)
- [x] ASR 模型加载与推理的 OpenVINO 原生 Core 重构 (Task 3)
- [x] 接口大 Batch 化标点集成验证测试 (Task 4)
- [x] 最终 A/B 性能测试对比跑评与基准数据收集 (Task 5)

### Files Modified

| File | Changes | Rationale |
|------|---------|-----------|
| `asr_onnx_service.py` | 引入 `openvino.Core` 推理；引入 `psutil` 硬绑定 CPU 前 6 个物理核心，设定计算库线程为 6 | 实现 CPU 下 OpenVINO 算子级并行推理，并防止线程超抢占引起的上下文切换开销。 |
| `tests/test_openvino_speed.py` | 修正模拟特征输入维度为 560 | 适配 SenseVoiceSmall 的 LFR 7倍拼接特征维度。 |
| `tests/test_affinity.py` | 新建单元测试用例 | 校验 psutil 的硬绑核状态是否生效。 |
| `docs/superpowers/specs/2026-07-09-openvino-integration-design.md` | 新建 Spec 规约 | 记录 OpenVINO 引擎编译与物理绑核的总体设计。 |
| `docs/superpowers/plans/2026-07-09-openvino-integration.md` | 新建执行计划 | 记录打勾跟进的实施任务。 |

### Decisions Made

| Decision | Options Considered | Rationale |
|----------|-------------------|-----------|
| 选用 OpenVINO 原生 Python API (Core) 方案 | 1. 原生 API (Core)<br>2. ORT-OpenVINO-EP<br>3. 离线转 OpenVINO IR 格式 | 原生 API 不受 ORT-EP 接口映射和动态 shape 频繁重编的限制，大 Batch 下多流调度能力更好；比 IR 离线格式更易于支持模型的快速热插拔更新。 |

## Pending Work

### Immediate Next Steps

1.  **直接集成 `sherpa-onnx` 引擎**：`model_quant.onnx` 实际上是由 `sherpa-onnx` 项目为嵌入式/CPU 定制量化并导出的。如果直接在 ASR 服务中集成 `sherpa-onnx` 引擎 Python 绑定（`pip install sherpa-onnx`），在其内置的全 C++ 高性能 ASR Pipeline（C++ 特征提取 + C++ CTC 解码 + C++ 逆文本正则）支持下，**CPU 推理速度预计能比当前的 PyTorch 原生模式再提速 2~3 倍**。
2.  **性能对照跑评开发**：编写接入 `sherpa-onnx` 引擎后的端到端微服务 A/B 跑评测试。

### Blockers/Open Questions

- 在我们手写的拼装服务（Python特征 + OV/ORT推理 + Python解码）中，INT8 量化模型即使优化后比 ORT 快 20%，但依旧比 PyTorch-CPU FP32 慢（30.96s vs 24.43s）。
- 这验证了轻量声学模型在通用推理框架下的量化瓶颈（动态反量化开销吃掉了乘法节省时间），以及 PyTorch-oneDNN JIT 算子融合的强大。
- 终极破局只有依靠 `sherpa-onnx` 的全 C++ 高性能融合 Pipeline 引擎。

### Deferred Items

- 离线导出并配置 OpenVINO 原生 IR 格式量化模型（由于有 `sherpa-onnx` 这一更直接、更强悍的路线，优先进行 sherpa-onnx 的集成尝试，暂缓原厂工具量化）。

## Context for Resuming Agent

### Important Context

*   **模型与推理器的强绑定性**：`model_quant.onnx` 是 k2-fsa/sherpa-onnx 为自己的 C++ 引擎编译导出的。在通用 session 下由于显式量化图严重干扰算子融合，时延表现会退化。下一任高级 AI 代理应着重在微服务中把 ASR 模块替换为 `sherpa-onnx` 的 python Recognizer。
*   **环境说明**：运行于 `E:\conda\envs\asr_ui_env`，所有 CPU 推理基准跑评均通过 `FORCE_CPU=1` 环境变量控制。

### Assumptions Made

- 假设运行于 6 逻辑核心及以上的 x86 CPU 系统（如果换为更少或更多核心机器，需要动态调整 `asr_onnx_service.py` 头部的 `cpu_affinity` 绑核列表）。

### Potential Gotchas

- 在不加载 VAD 和拼合时，`test_openvino_speed.py` 会因为缺少 560 拼接导致 input shape 崩溃，注意 mock 数据维度必须为 560。

## Environment State

### Tools/Services Used

- OpenVINO 2026.2.1-21919
- ONNX Runtime 1.26.0
- psutil 6.1.1
- pytest 9.1.1

### Active Processes

- None (跑评在 8001 和 8002 端口的后台 Python 进程已在脚本清理阶段优雅终止，无挂起进程)

### Environment Variables

- `FORCE_CPU` (用于控制 PyTorch 禁用 cuda 转移、ONNX Runtime 使用 cpu runtime 执行的纯 CPU 基准测试标记)

## Related Resources

- 设计规约: [2026-07-09-openvino-integration-design.md](file:///E:/project/funclip-pro/docs/superpowers/specs/2026-07-09-openvino-integration-design.md)
- 执行计划: [2026-07-09-openvino-integration.md](file:///E:/project/funclip-pro/docs/superpowers/plans/2026-07-09-openvino-integration.md)
- 工作总结: [walkthrough.md](file:///C:/Users/song/.gemini/antigravity/brain/70f1533e-b1f7-41bf-b86d-ccd0a3b29517/walkthrough.md)
