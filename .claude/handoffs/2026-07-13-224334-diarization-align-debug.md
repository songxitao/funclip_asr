# Handoff: 说话人分离全局聚类与语义对齐优化调试

## Session Metadata
- Created: 2026-07-13 22:43:34
- Project: E:\project\funclip-pro
- Branch: main
- Session duration: 约 1.5 小时

### Recent Commits (for context)
  - 5cd2fdf feat: 新增 cli_transcribe.py 命令行转写客户端
  - 6ad166f docs: 更新 dify_openapi.yaml，增加 diarize 分人参数
  - 4899703 docs: 建立三驾马车文档系统 (README/ONBOARDING_MANUAL/AGENTS) 规范协同开发

## Handoff Chain
- **Continues from**: [HANDOFF.md (2026-07-12 22:45:00)](./HANDOFF.md)
- **Supersedes**: 无

## Current State Summary
本会话主要解决了在开启 `seg_clustering` 策略时，新版本存在的“识别速度慢、无标点、无时间戳、字幕重复复读”等一系列痛点问题。我们发现了两个极具隐蔽性的底层 Bug，完成了高精度的“全局聚类 + 语义子句级投票分配”对齐重构，在 `output.mp3` 测试音频上通过验证，彻底消除了文本重复且找回了丢失的重叠段字幕，解说员亦被正确分配为全局一致的 `说话人3`。

针对重叠语音和无字级时间戳的对齐精度挑战，尖子提出了“基于断句语义分配重叠语音”的高阶多模态融合思路，我们已将其提炼并更新至任务 TODO 选项中，待尖子后续决策。

## Codebase Understanding

### Architecture Overview
在说话人分离转写中，ASR 解码与 Diarization（Seg）通常是并行提取的。由于 SenseVoice ONNX 解码层没有直接提取字级时间戳，只能在 ASR 解码后将文本与说话人时间轴进行对齐。

### Critical Files

| File | Purpose | Relevance |
|------|---------|-----------|
| `asr_onnx_service.py` | FastAPI 接口推理逻辑 | **修改**：在 `_run_inference` 函数中重构了 `seg_clustering` 逻辑，在顶层进行全局 Diarization，并引入了 `_assign_clauses_to_speakers` 语义分句投票算法。 |
| `tests/test_sliding_integration.py` | 集成测试 | **修改**：更新 mock 数据以适配新的对齐流程，目前全套测试 PASSED。 |

### Key Patterns Discovered
*   **局部聚类导致 ID 错乱**：不能在每个局部 VAD 段内部单独调用 `cluster_with_segmentation`。这会导致各段之间的同一个说话人 ID 无法对齐。必须对整段音频 `y` 统一跑全局聚类。
*   **物理裁剪导致丢字**：`segmentation-3.0` 会把所有的说话人重叠段（Overlap）标为 `-1` 并在输出中扔掉。如果直接以 Seg 子区间为 ASR 的物理切片块，会直接把重叠段的音频截断剪掉，导致丢字。

## Work Completed

### Tasks Finished
- [x] **Bug 1（解说员无法识别为说话人3）诊断与修复**：将说话人聚类提升至顶层全局运行，保障了全局发音人 ID 连续性。
- [x] **Bug 2（重叠段丢字）诊断与修复**：ASR 块重新放回粗粒度 VAD 活性块（带 800ms padding ），一字不漏地解码。
- [x] **语义子句级时间戳分配**：实现 `_assign_clauses_to_speakers` 算法。将文本按标点断句为子句单元，估算子句区间并与全局说话人时间轴比对，将整个子句一次性投票分配给重合度最长的说话人，彻底杜绝了复读。
- [x] **接口验证成功**：本地接口调用与 `cli_transcribe.py` 转写测试完全符合预期。

## Pending Work

### Immediate Next Steps
1.  **尖子进行方案决策**：
    *   **方案一（粗粒度 ASR + 字级时间戳映射）**：如果确认需要最高精度的对齐，且后续要导出标准的 SRT 格式，需调试出 `funasr` 在 ONNX 推理时输出字级/字符级 `timestamp` 字段，并在文本层就近向说话人段收拢（或按语义断句分配）。
    *   **方案二（无缝时间轴扩张 + 物理切片 ASR）**：如果无法轻易获取字级时间戳，使用此替代方案。通过把 Seg 时间段的所有空隙对半平分并扩张填充，构成无缝首尾相连的时间轴，再物理切割 ASR 解码。
    *   **方案三（尖子提议的转换点细切 + 未知段语义融入）**：面临 ASR 在极短音频（<1s）下漏字、幻觉和识别率急剧下降的隐患，在工程上通常不推荐物理切得过碎。
2.  **固化重构代码**：根据尖子的最终方案选择，微调 [asr_onnx_service.py](file:///E:/project/funclip-pro/asr_onnx_service.py) 并提交 Git。

## Context for Resuming Agent

### Important Context
*   目前已实现的“子句级投票分配”已经在 [asr_onnx_service.py](file:///E:/project/funclip-pro/asr_onnx_service.py) 中，对 `output.mp3` 进行测试时，端到端耗时大幅降至 **`12.9` 秒**（提速 >30%），没有漏字和重复，标点正常。
*   如果运行 `cli_transcribe.py`，请先通过端口检测 `netstat -ano \| findstr :8002` 确认服务状态，并用 `E:\conda\envs\asr_ui_env\python.exe asr_onnx_service.py` 开启服务。

### Potential Gotchas
*   `segmentation-3.0` 模型必须用 `pyannote.audio` 原生接口调用。环境内 numpy 需锁定在 `1.26.4` 以避免 `np.NaN` 兼容冲突。
*   在 Windows 终端输出中，如果打印非 ASCII 字符，必须先在终端切换代码页：`chcp 65001 >$null`。

## Environment State
*   Conda 专属虚拟环境：`E:\conda\envs\asr_ui_env\python.exe`
*   FastAPI 监听端口：`8002`
*   测试音频文件：`E:\project\funclip-pro\output.mp3`
