# Handoff: ASR GPU/CPU 性能评估与 ONNX 优化瓶颈剖析

## 1. 任务当前状态 (Current State)

本阶段完成了对 **8001 端口原生 PyTorch ASR 服务** (`asr_service.py`) 的彻底优化：
1. **CPU 大核绑定与多线程限制**：
   - 进程头部通过 `psutil` 绑定当前进程的 CPU 亲和性至前 6 个大核心 (`[0, 1, 2, 3, 4, 5]`)。
   - 设置全局多线程限制 `OMP_NUM_THREADS = "6"` 等，并执行 `torch.set_num_threads(6)`。
   - 成功防范了 CPU 被多线程跑满 100% 抢占导致系统锁死死机的隐患。
2. **大 Batch 合并推理管线**：
   - 将原来 for 循环单句 ASR 的重复推理与频繁 GPU/CPU 拷贝，重构为 **“1 次显存装载 -> 1 次大 Batch 并发推理 -> 1 次显存卸载与清空”** 的高效流水线（在 `finally` 块中通过 `MODEL.model.to("cpu")` + `torch.cuda.empty_cache()` 可靠释放显存）。
3. **标点模型后处理集成**：
   - 在 startup 时于 CPU 上加载本地 `CT-Punc` 模型，并在大 Batch 推理生成 `raw_text` 后进行后处理，使文本还原排版标点。
4. **测试情况**：
   - 新增了单元测试 `tests/test_pytorch_affinity.py`、`tests/test_pytorch_punc_load.py`、`tests/test_pytorch_inference_refactor.py`、`tests/test_pytorch_route.py`。
   - 编写了服务 API 集成测试 `tests/test_pytorch_service_integration.py`。
   - 以上测试已经全数通过，转写文本中成功包含了中英文标点。

---

## 2. 跑评对照测试数据 (Benchmark Results)

我们通过测试音频 `E:\下载\下载\李雪花2.wav`（时长约 4 分钟，VAD 切分为 85 个片段）在以下四个硬件与引擎维度下进行了精确的跑评：

| 测试维度 | 运行硬件 | 推理引擎版本 | 耗时 (秒) | 核心字数 | 文本吻合度 (对齐PyTorch) | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch-GPU** | GPU (移入显存) | FP32 (大 Batch 合并) | **7.78 秒** | 1692 字 | 100.00% | 重构后性能暴增 |
| **PyTorch-CPU** | CPU (纯6线程限制) | FP32 (大 Batch 合并) | **23.23 秒** | 1692 字 | 100.00% | 运行于 CPU FP32 浮点 |
| **ONNX-GPU** | GPU (显卡调用) | INT8 (大 Batch 量化) | **26.94 秒** | 1692 字 | 98.52% | 相比 PyTorch-GPU 慢 3.46 倍 |
| **ONNX-CPU** | CPU (纯6线程限制) | INT8 (大 Batch 量化) | **33.49 秒** | 1692 字 | 98.52% | 相比 PyTorch-CPU 慢 1.44 倍 |

---

## 3. 遗留核心课题：ONNX 推理性能瓶颈成因

在本次重构中我们发现了**反直觉的性能差距**：
*   在 GPU 上，ONNX GPU 的推理速度不如优化后的 PyTorch GPU。
*   在 CPU 上，量化后的 ONNX CPU 推理（33.49秒）依然慢于 PyTorch CPU 的浮点推理（23.23秒）。

经代码剖析，我们将性能瓶颈定位于以下两个核心原因：

### 瓶颈 A：ONNX 包装类中高频的 Python 级循环与解码开销
在 `asr_onnx_service.py` 内部的 `SenseVoiceSmall` 的 `__call__` 解码逻辑中：
- 贪婪搜索（Greedy Search）、CTC logits 排序与去重逻辑是**完全基于 Python 循环手写**的：
  ```python
  for b in range(end_idx - beg_idx):
      x = ctc_logits[b, : encoder_out_lens[b].item(), :]
      yseq = x.argmax(dim=-1)
      yseq = torch.unique_consecutive(yseq, dim=-1)
      ...
      asr_res.append(tokenizer.tokens2text(token_int))
  ```
- 每次请求切出 85 个 batch，代码就在 Python 层面执行了 85 次张量操作、CPU 拷贝，以及 85 次在 `tokenizer.tokens2text` 中的列表遍历。这导致了极大的 Python 解释器 GIL 锁延迟和循环开销。
- 与之相比，PyTorch `AutoModel.generate` 内部的解码和 CTC 转换全部在 C++（FunASR 核心库）中融合完成，几乎没有 Python 级别的开销。

### 瓶颈 B：ONNX 导出时缺乏算子融合 (Operator Fusion)
- PyTorch 能够使用 Intel MKL-DNN 算子库对 FP32 进行寄存器流水线级别的优化，并自动把 Attention 和 Linear 等算子融合成单个高效执行的 kernel。
- 导出的 ONNX 在运行时（onnxruntime）面对的是成百上千个零散的小算子。如果没有进行针对性的量化融合编译（如 TensorRT / CTranslate2），小算子之间频繁的数据拷贝和框架调用开销（Framework Overhead）会吞噬掉量化带来的计算红利。

---

## 4. 下一步行动与对高级模型的提问方向

我们建议将以下课题移交给高级模型进行攻关：
1. **改写/替换 Python Decode 解码器**：
   - 考虑如何把 `asr_onnx_service.py` 内部手写的 `__call__` 解码部分用 C++ 改写，或者引入 FunASR 内部原生的 C++ 批量解码器，彻底消除 Python 循环。
2. **优化 ONNX Runtime 执行配置与图优化**：
   - 尝试在 `SenseVoiceSmall` 载入时，开启高级别的 ONNX 图优化选项：
     ```python
     opts = ort.SessionOptions()
     opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
     ```
     并检查是否能通过 `ctranslate2` 或者把模型导出为更具 GPU/CPU 自适应能力的格式来加速 INT8 推理，解决反量化（Dequantization）开销。
3. **特征提取并发化**：
   - 检查 `extract_feat` 阶段，将其改写为并行计算，减少数据总线在 Host (CPU) 和 Device (GPU) 之间的多次往返拷贝。
