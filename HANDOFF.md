# Handoff: SenseVoiceSmall Sherpa-ONNX 联合评测与推理性能飞跃

## Session Metadata
- Created: 2026-07-09 19:54:00
- Project: E:\project\funclip-pro
- Branch: main
- Session duration: 约 1 小时

## Current State Summary

本会话验证并实施了使用 `sherpa-onnx` 高性能推理引擎搭配专为其定制量化的新 INT8 模型（`model.int8.onnx`）的优化工作。
1. **测试脚本开发**：先后编写了纯推理评测脚本 `tests/test_sherpa_performance.py` 与端到端 VAD 联合 A/B 测试脚本 `tests/test_vad_sherpa_comparison.py`，并提交至 git (提交ID: `d2e098b` 与 `3a9521a`)。
2. **多向性能结果对齐**：成功跑通了 CPU 与 GPU 在 PyTorch、ONNX Runtime、OpenVINO、Sherpa-ONNX 等多维度下的 7 分钟真实长音频跑评，形成了完整的 ASR 推理性能矩阵。
3. **性能飞跃验证**：在 CPU 6 线程硬性限制下，`Sherpa-ONNX` 以 **0.0248** 的超低实时率 (RTF) 跑出了 **10.52 秒** 的成绩，比 PyTorch FP32 CPU 快了 **54%**，比 ORT/OpenVINO 快了近 **200%**，去标点文本吻合度高达 **96.92%**。目前工作区处于干净状态。

## Codebase Understanding

### Architecture Overview
音频处理由 `FSMN-VAD`（PyTorch-CPU）进行切片分割，每一片切分后的 wav chunk 注入 `sherpa-onnx` 的 `OfflineStream` 中，最终利用它的批量 C++ 推理接口 `decode_streams` 进行全并理解码。

### Critical Files

| File | Purpose | Relevance |
|------|---------|-----------|
| `tests/test_vad_sherpa_comparison.py` | VAD + PyTorch 与 VAD + Sherpa-ONNX 端到端 A/B 联合测试 | 验证多切片并发解码、RTF 耗时对比以及 Levenshtein 吻合度计算的所在地。 |
| `tests/test_sherpa_performance.py` | 纯 ASR 模型单流推理时延跑评 | 测量冷启动、热启动与长音频流式解码耗时。 |
| `model/models/iic/SenseVoiceSmallOnnx` | 新下载的 sherpa-onnx 专属模型目录 | 包含 `model.int8.onnx` 模型与 `tokens.txt` 词表。 |
| `README_PERF.md` | 四轨性能与字错吻合度对比报表 | 汇总记录最新的跑评指标与量化慢的深层根因。 |

### Key Patterns Discovered
- **`decode_streams` 高并发接口**：使用 `sherpa-onnx` 时，如果对多个切句调用 `decode_stream` 循环会增加 Python 开销，而将所有 `OfflineStream` 放入列表传入 `decode_streams` 则能自动在底层用 C++ 并发跑满所有物理核。

## Work Completed

### Tasks Finished
- [x] 安装 `sherpa-onnx==1.13.4` (Task 1)
- [x] 独立测试脚本 `test_sherpa_performance.py` 编写、运行与提交 (Task 2)
- [x] 联合切片测试脚本 `test_vad_sherpa_comparison.py` 编写与 A/B 对比评测 (Task 3)
- [x] 完成 CPU 与 GPU 维度的全面性能数据分析与 README_PERF 更新 (Task 4)

### Files Modified

| File | Changes | Rationale |
|------|---------|-----------|
| `README_PERF.md` | 补充了完整的 GPU 和 CPU 联合对比数据，深度分析了量化在 CPU/GPU 上慢的物理本质。 | 方便让后续接管人清晰了然为什么之前方案慢、新方案如何避坑。 |
| `tests/test_sherpa_performance.py` | 新增纯 ASR 测试脚本 | 验证 Sherpa 引擎加载及长音频纯解码速度。 |
| `tests/test_vad_sherpa_comparison.py` | 新增 VAD 联合测试脚本 | 完成 FSMN-VAD 自动分段后的双轨批处理对照测试。 |

### Decisions Made
- **在 CPU 推理上完全选用 Sherpa-ONNX 方案**：由于 ONNX Runtime/OpenVINO 的动态 Shape 重新编译和零散量化算子在 GPU/CPU 下均存在严重时延退化，因此将 CPU 轨道的部署引擎选定为 `sherpa-onnx`。

## Pending Work

### Immediate Next Steps
1. **替换微服务引擎**：在 `asr_onnx_service.py` 内部引入 `sherpa_onnx.OfflineRecognizer`。
2. 将服务启动加载的 ASR 模型目录变更为 `model/models/iic/SenseVoiceSmallOnnx`。
3. 重构接口的 ASR 推理方法（使用 `recognizer.decode_streams`）。
4. 运行 `tests/test_onnx_service_integration.py` 确保微服务接口在合入后仍然能返回带有 ITN 与标点的正确 JSON 格式。

### Blockers/Open Questions
- 无。目前 `sherpa-onnx` 依赖环境、模型加载与精度对齐均已全部打通，无任何阻碍。

## Context for Resuming Agent

### Important Context
- **GPU 瓶颈警示**：千万不要尝试将 `model_quant.onnx` 或 `model.int8.onnx` 送入 ONNX Runtime GPU (CUDA) 执行。由于频繁的动态量化反量化转换，其速度 (32.33s) 比 PyTorch 原生 FP32 GPU (7.74s) 慢 4.17 倍。
- **环境说明**：运行于 `E:\conda\envs\asr_ui_env`，已成功安装了 `sherpa-onnx`（版本 `1.13.4`）。

### Potential Gotchas
- 在批量创建 `OfflineStream` 时，如果音频数据长度极短（少于 1600 个采样点，即 0.1 秒），调用 `accept_waveform` 会导致模型推理报错，在联合切片推理时已硬性过滤了这些无效短片。
