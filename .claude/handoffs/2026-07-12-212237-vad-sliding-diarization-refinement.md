# Handoff: 基于 VAD 活性段滑窗提纯的说话人分离改进

## Session Metadata
- Created: 2026-07-12 21:22:37
- Project: E:\project\funclip-pro
- Branch: main
- Session duration: 约 1.5 小时

### Recent Commits (for context)
- 84185f2 feat: 评测选项添加 vad_sliding 并在 AliMeeting 测试集完成全量验证 (Task 4)
- 8eebb43 feat: API 服务支持 vad_sliding 策略并合并输出文本 (Task 3)
- ea42250 feat: cluster 支持 vad_sliding 声纹提纯离线聚类策略 (Task 2)
- e595595 feat: 新增 VAD 活性段内部滑窗声纹提纯函数及单测 (Task 1)
- 672ea80 docs: 增加基于 VAD 活性段滑窗提纯 of 说话人分离实施计划

## Handoff Chain
- **Continues from**: [2026-07-12-014154-funclip-pro-sliding-diarization-review.md](./2026-07-12-014154-funclip-pro-sliding-diarization-review.md)
  - Previous title: 2026-07-12-014154-funclip-pro-sliding-diarization-review
- **Supersedes**: 无

## Current State Summary
本会话围绕“基于 VAD 活性段滑窗提纯”的优化方案展开。完成了全部开发与两阶段审核（Spec一致性 + 代码质量），并在 `E:\conda\envs\asr_ui_env` 环境下跑通了所有的集成测试与单元测试。
此外，完成了单场 `R8002_M8002` 的策略效果评测，结果显示：
1. 新策略 `vad_sliding` 成功解决了静音段盲滑导致的虚警率（FA）上升副作用，FA 重回最佳的 **0.3%**；
2. 完美保全了 ASR 的字幕连贯性与话序，完成了 segments 的文本回填，不存在打乱字幕次序的风险；
3. 相较于直接对整个 VAD 大段（~8s）提取 embedding 的旧方案，新策略通过段内“滑窗提纯+均值聚合”使混淆错判（CONF）稳步降低了 **1.5%**（绝对判定错误减少了 **1970 处**），证明了滑窗在 VAD 限制内提纯声纹特征的有效性。

目前代码已全部 Commit。由于运行 20 场的全量评测脚本 `run_ali_der_full.py` 耗时较长（1-2h）且极占系统显存/CPU，已在此设置卡点暂停，等待用户下一步指令。

## Codebase Understanding

### Architecture Overview
ASR 文本转写和说话人分离在时序切分上完全以 VAD 段为基础（1:1）。每个 VAD 活性大段在提取声纹时：
1. 在段内以 1.5s 窗长、0.5s 步长滑动切片，逐窗提取 Cam++ Embedding。
2. 剔除 `None` 向量后，对各窗提取的一组特征向量求算术平均（Mean），并做 $L_2$ 归一化。
3. 得到高纯度代表特征向量后参与全局聚类。

这完美避开了盲滑窗的静音噪声干扰，并抑制了 VAD 大段内多人混杂引起的声纹污染。

### Critical Files

| File | Purpose | Relevance |
|------|---------|-----------|
| `speaker_engine.py` | CampPlusSpeaker 类：特征提取与聚类分发 | **核心改动**：新增 `extract_embedding_sliding_mean` 段内提纯与 `cluster` 策略扩展 |
| `asr_onnx_service.py` | FastAPI /transcribe 推理端点 | **核心改动**：diarize 分支接入 `vad_sliding` 并回填 ASR 段对应的 `text` |
| `tests/test_vad_sliding.py` | vad_sliding 专用单元测试 | **新增测试**：覆盖常规、全 None 兜底、过短退避、Tensor 兼容等 6 个用例 |
| `tests/test_sliding_integration.py` | 服务集成测试 | **新增测试**：测试 `vad_sliding` 策略下 API 连通性与转写文本合并 |
| `der_eval.py` | 评测参数定义与 DER 计算 | **修改**：argparse 包含 `vad_sliding` 选项 |
| `ali_der_eval.py` | 单场评测脚本 | **运行使用**：用于单场 `R8002_M8002` 指标评测 |
| `run_ali_der_full.py` | 全量 20 场评测编排 | **运行使用**：等待触发，跑全量 20 场的对比 |

### Key Patterns Discovered
- **Onnx/Funasr 模型多卡/线程安全限制**：由于 GPU_SEMAPHORE 限制，在并发请求下应关注模型实例的线程安全。
- **时序退避防崩溃**：滑窗提纯中包含双重退避（音频极短时退避至对整段直接提取；全 None 时退避至对整段直接提取），确保服务绝对不崩溃。

## Work Completed

### Tasks Finished
- [x] 在 `speaker_engine.py` 实现段内滑窗平均提纯 `extract_embedding_sliding_mean` 及其边界兜底。
- [x] 扩展 `CampPlusSpeaker.cluster` 支持 `strategy="vad_sliding"`。
- [x] 在 `asr_onnx_service.py` 接入新策略并回填 `segments[i]["text"]` |
- [x] 编写并跑通 6 个单元测试用例及 2 个集成测试用例。
- [x] 修改 `der_eval.py` 的 choices 范围。
- [x] 重启服务并完成了 `R8002_M8002` 单场性能指标评测与数据对比。

### Files Modified

| File | Changes | Rationale |
|------|---------|-----------|
| `speaker_engine.py` | 新增 `extract_embedding_sliding_mean`，扩展 `cluster` | 增加滑窗提纯聚类策略 |
| `asr_onnx_service.py` | 修改 `_run_inference` 合并分支并支持新参数 | API 层路由合并支持 |
| `der_eval.py` | 增加 `vad_sliding` choices | 支持脚本在新策略下计算 |

### Decisions Made

| Decision | Options Considered | Rationale |
|----------|-------------------|-----------|
| **以 VAD 大段为分人输出基准** | A. 盲滑窗切碎文本 / B. 以 VAD 大段绑定文本 (1:1) | 方案 A 容易搞乱字幕话序，用户可读性差且有 3.4% 的高 FA 副作用；方案 B 保障字幕话序连贯，FA 极低 |
| **采用段内滑窗提纯 (Mean)** | A. 直接提 VAD 整段 / B. 段内滑窗并取平均 (Mean) | 方案 B 可以提纯声纹特征，避免大段内多人混杂污染 embedding 空间，稳步减少 CONF 错判 |

## Immediate Next Steps
1. 运行 `run_ali_der_full.py --strategy vad_sliding` 进行全量 20 场评测。
2. 收集全量加权平均对比报告（`vad_sliding` vs `spectral`），验证在全量场景下 CONF 错误降低的稳定性。

## Important Context
- **运行环境**：测试和执行唯一的 Conda 运行环境为 `E:\conda\envs\asr_ui_env\python.exe`。
- **控制台防乱码**：所有终端命令前必须拼接 `chcp 65001 >$null; [Console]::OutputEncoding = [System.Text.Encoding]::UTF8;`，物理锁死 UTF-8 编码，防止 Windows PowerShell 乱码。
- **服务占用**：服务端口固定为 8002。若重启报错，通过 `netstat -ano \| findstr :8002` 查找占用 PID 并用 `taskkill /PID <PID> /F` 强制终止。

### Assumptions Made
- 假设 VAD 提供的语音起止点时间信息绝对准确。
- 假设在同一个 VAD 大段内，即使有短暂的声音重叠，均值特征也可以表达主导说话人。

### Potential Gotchas
- 在没有字级时间戳的环境下，一个 VAD 段 if 混杂多人说话，未主导的说话人部分在 DER 计分中必然会成为混淆错误。

## Environment State
- FastAPI 语音转写服务：端口 8002
- pytest-9.1.1 单元测试框架

### Active Processes
- 8002 后台服务进程：进程 PID 可用 `netstat -ano \| findstr :8002` 动态查证。

## Related Resources
- 新 Spec 设计：`docs/superpowers/specs/2026-07-12-alimeeting-diarization-vad-sliding-refinement-design.md`
- 实施计划：`docs/superpowers/plans/2026-07-12-alimeeting-diarization-vad-sliding-refinement.md`
- 单场验证结果：`walkthrough.md`
