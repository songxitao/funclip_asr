# SenseVoice ASR 性能优化深度分析与高级模型提问指引 (README_PERF)

本文件整理了监听 8001 端口的原生 PyTorch 服务与 8002 端口的 ONNX 服务在 GPU/CPU 维度的四向跑评数据，并为尖子提炼了针对高级模型的深度提问 Prompt，以便您在新会话中直接开展后续研究。

---

## 1. 四向性能与字错吻合度 (CER) 报表

使用 4 分钟长音频 `李雪花2.wav`（VAD 划分 85 个音频片段）测试，数据结果如下：

| 推理引擎 | GPU (移入显存) | CPU (纯6线程限制) | 加速比 (GPU/CPU) | 核心字数 | 较 PyTorch 文字吻合度 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch 原生 (FP32)** | **7.78 秒** | **23.23 秒** | **2.98x** | 1692 字 | 100.00% (Baseline) |
| **ONNX (INT8 量化)** | **26.94 秒** | **33.49 秒** | **1.24x** | 1692 字 | 98.52% (CER 1.48%) |

*   **数据亮点**：
    1.  **PyTorch 优化巨大成功**：在 GPU 上仅耗时 **7.78 秒**，彻底解决了因 VAD 重复拷贝显存和 `empty_cache()` 导致的 CPU 锁死疯跑问题。
    2.  **性能反转现象**：
        - 在 GPU 上，PyTorch 速度是 ONNX 的 **3.46 倍**。
        - 在 CPU 上，未量化的 PyTorch FP32（23.23秒）依然比量化后的 ONNX INT8（33.49秒）快了 **1.44 倍**。

---

## 2. 核心瓶颈理论分析

1.  **解码后处理的 Python 性能蚕食**：
    ONNX 版本的 `SenseVoiceSmall` 的 `__call__` 函数采用纯 Python 编写了 Greedy Search 和 CTC 去重后处理：
    ```python
    for b in range(end_idx - beg_idx):
        x = ctc_logits[b, : encoder_out_lens[b].item(), :]
        yseq = x.argmax(dim=-1)
        yseq = torch.unique_consecutive(yseq, dim=-1)
        ...
        asr_res.append(tokenizer.tokens2text(token_int))
    ```
    对于 85 个片段，循环了解码 85 次。在 Python 层面频繁执行张量切片、NumPy/List 转换以及字符映射循环，受制于 Python 解释器的 GIL 锁与低执行效率。而 PyTorch 原生的 `AutoModel.generate` 所有的解码逻辑均被编译成 C++（FunASR 核心）在底层高并发融合运行，近乎零 Python 开销。

2.  **ONNX 算子零散与反量化开销**：
    - 量化模型在 GPU (CUDA) 运行时，会产生频繁的 **动态反量化 (Dequantization) 回 FP32** 的显存开销。
    - ONNX 导出的图结构包含了成百上千个零散的小算子，由于缺乏像 PyTorch 编译级（或 TensorRT 级别）的**算子融合 (Operator Fusion)** 优化，框架调度开销（Framework Overhead）极高，抵消了量化所节约的矩阵计算时间。

---

## 3. 专属提问高级模型的 Prompt (可以直接复制)

您在新会话中将这段 Prompt 直接输入给高级模型，能帮助它瞬间接管项目并进行最深度的攻关：

```markdown
我正在开发基于 FunASR 的语音转写项目 (funclip-pro)。
我们目前对 8001 端口的原生 PyTorch ASR 服务 (SenseVoiceSmall) 和 8002 端口的 ONNX INT8 量化服务 (SenseVoiceSmall-ONNX) 进行了 CPU 6核锁定状态下的四向跑评测试。

测试音频为 4 分钟 WAV 音频，被 VAD 划分为了 85 个音频片段，测试结果如下：
1. PyTorch-GPU FP32 耗时: 7.78 秒
2. PyTorch-CPU FP32 耗时: 23.23 秒
3. ONNX-GPU INT8 量化耗时: 26.94 秒
4. ONNX-CPU INT8 量化耗时: 33.49 秒

【现状剖析】
我们初步定位了 ONNX 的性能瓶颈：
- 瓶颈一：在 asr_onnx_service.py 中，ONNX 的解码和 CTC 后处理（Greedy Search、argmax、去重和 token2text 映射）全部是在 Python 级别用 for 循环手写的（比如遍历 batch 执行 x.argmax 和 tokens2text），带来了极大的 Python 解释器 GIL 锁与 CPU 循环开销。而 PyTorch 版底层是 C++ 级融合解码。
- 瓶颈二：导出的 ONNX 模型算子极度散乱，在 onnxruntime 运行时缺乏图级别的算子融合优化，且在 GPU 上可能存在频繁的反量化（INT8->FP32）额外吞吐开销。

【请你解答并提供具体的代码优化方案】
1. 针对“解码后处理的 Python 性能蚕食”，我们如何重构 asr_onnx_service.py 内部的解码逻辑？有没有办法绕过 Python 循环，使用 NumPy 向量化操作，或者通过 FunASR 内部原生的 C++ 解码工具来批量解算 logits 从而消灭 CPU-GIL 延迟？请给出具体的优化代码。
2. 针对 ONNX 算子离散与量化速度缓慢，我们如何配置 onnxruntime 的 Graph Optimization Level？或者有没有比 ctranslate2 或其他推理引擎更适合 SenseVoiceSmall INT8 模型的推理引擎导出/部署方式？
3. 如果我们想要彻底点亮 ONNX 在 CPU 上的量化加速潜力，应该如何优化 waveform 到 features 的特征提取 (extract_feat) 并发度？
```
