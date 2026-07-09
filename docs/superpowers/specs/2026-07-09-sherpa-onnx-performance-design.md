# 2026-07-09-sherpa-onnx-performance-design

本设计规约描述了如何使用 `sherpa-onnx` 离线识别引擎对新下载的 SenseVoiceSmall INT8 量化模型（`model.int8.onnx`）进行 CPU 推理性能跑评测试。

## 1. 目标与成功指标
* **功能验证**：验证 `model/models/iic/SenseVoiceSmallOnnx/model.int8.onnx` 配合 `tokens.txt` 在 `sherpa-onnx` 推理器中能正常解码中文音频。
* **时延跑评**：测试单个音频片段的推理延迟（冷启动/热启动），并计算 RTF（实时率）。
* **精度对齐**：确保转写出的中文字符串与 PyTorch 基准文字一致性在 95% 以上。

## 2. 架构设计
测试脚本 `tests/test_sherpa_performance.py` 将作为一个独立的测试工具运行，使用 `E:\conda\envs\asr_ui_env` 虚拟环境。

### 数据流向
```
[李雪花2.wav] -> [librosa/wave 加载 (float32, 16kHz)] -> [sherpa_onnx.OfflineRecognizer] -> [转写文本]
```

### 依赖库
* `sherpa-onnx`：用于离线非流式识别。
* `numpy`：波形采样点承载。

## 3. 测试规约
脚本将执行以下测试：
1. **冷启动测试**：首次加载模型并对一段 5秒的 dummy 信号/音频进行推理，记录总体耗时。
2. **多轮热启动测试**：连续对 5秒的 dummy 信号推理 5 次，取平均耗时。
3. **真实长音频测试**：加载 `E:\下载\下载\李雪花2.wav`（或其片段），由 `sherpa-onnx` 离线批量解码，记录总时长与 RTF。

---
*设计方案自检完毕，无占位符，作用域明确。*
