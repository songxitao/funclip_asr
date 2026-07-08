# Handoff: ASR GPU 优化第一阶段完成及 PyTorch 原生服务 (8001端口) 优化规划

## Session Metadata
- **Created**: 2026-07-08T21:30:00+08:00
- **Project**: `E:\project\funclip-pro`
- **Branch**: `main` (Git status clean, latest commit: `a21516e`)
- **Key Target**: 尖子

---

## 1. Current State Summary (第一阶段已完成工作)
我们成功实现了基于 ONNX GPU 加速的 SenseVoiceSmall 独立微服务（运行在 **8002 端口**）的彻底优化：
1. **CPU 锁死与物理核心硬绑定**：
   - 引入了 `psutil` 硬性绑定进程 CPU 亲和性至前 6 个大核心 (`[0, 1, 2, 3, 4, 5]`)。
   - 限制 `torch` 及各类底层矩阵库在核心内只开辟最多 6 个线程，彻底腾空了其余 26 个 CPU 逻辑处理器，完全消除了 CPU 全核拉满 100% 导致系统卡死/死机的隐患。
2. **大 Batch 推理加速管线**：
   - 重构了 ASR 包装类 `SenseVoiceSmall` 的 `__call__` 解码循环。解开了 ONNX 只能解码单 Batch 的限制，实现了对整个 batch logit 维度的遍历提取。
   - 结合 VAD 切分后的音频分片，将原来串行 for 循环单句转写改为了一次性打包传入多 Batch (batch_size=16) 推理。
3. **加回 CPU 标点模型集成**：
   - 在启动时以 CPU (限制6线程) 载入本地 [punc_ct-transformer_zh-cn-common-vocab272727-pytorch](file:///E:/project/funclip-pro/model/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch)，在 ASR 识别完文本后进行后处理，使文本还原完美排版标点。
4. **测试情况**：
   - 自动化测试 `tests/test_affinity.py`、`tests/test_batch_decode.py`、`tests/test_onnx_service_integration.py` 已经全数通过。在 6 核限制下，4分钟长音频 ASR 转写+标点集成在 55.41 秒内极速算完，无任何 CPU 疯跑异常。

---

## 2. Pending Work (下一阶段开发计划：优化 8001 端口 PyTorch 原生服务)

下个 Session 的工作重点是：对原生的 PyTorch ASR 服务 [asr_service.py](file:///E:/project/funclip-pro/asr_service.py)（监听 **8001 端口**）进行类似的重构优化。

### ⚠️ 核心显存约束（GPU VRAM Limits）
*   **痛点**：当前显卡显存极度紧张（12GB VRAM），原生 PyTorch 模型较大。如果不及时释放，容易在跑大模型工作流时发生 CUDA OOM。
*   **解决方案**：
    *   **必须保留**原生的“用时加载到 GPU，用完立刻卸载回 CPU / 清空缓存”的机制，即：
        - `MODEL.model.to("cuda")`（移到 GPU）
        - `finally` 块中的 `MODEL.model.to("cpu")` + `torch.cuda.empty_cache()`（卸载回 CPU 释放显存）
    *   **大 Batch 优化**：将原来“for 循环 30 次单句 ASR” 带来的 **30 次显存频繁反复装载卸载**（这也是 CPU 疯跑死机的主因），优化为 **“1 次显存装载 + 1 次 Batch ASR 推理 + 1 次显存卸载”** 的高效流水线。
    *   **6核硬锁定**：脚本头部加入 `psutil` 硬绑定 CPU `[0, 1, 2, 3, 4, 5]` 核心，限制 CPU 运算开销。
    *   **标点加回**：加载本地 CT-Punc 标点模型，并在 CPU 上后处理文本。

---

## 3. 下阶段开发 TODO 清单 (Action Items)

### Task 1: 物理核心硬绑定与 CPU 多线程资源限制
- [ ] 在 [asr_service.py](file:///E:/project/funclip-pro/asr_service.py) 的最头部导入 `psutil` 并硬锁定逻辑核心为 `[0, 1, 2, 3, 4, 5]`：
  ```python
  import os
  import psutil
  try:
      psutil.Process().cpu_affinity([0, 1, 2, 3, 4, 5])
  except Exception as e:
      print(f"亲和性设置失败: {e}")
  ```
- [ ] 在头部注入 `OMP_NUM_THREADS = "6"` 等环境变量，并限制 `torch.set_num_threads(6)`。

### Task 2: 显存“单次载入卸载”与多 Batch 合并推理管线重构
- [ ] 找到 [asr_service.py:107-122](file:///E:/project/funclip-pro/asr_service.py#L107-L122) 的 ASR 推理循环，将其重构。
- [ ] 在 `finally` 之前，把 VAD 分割后的所有 chunks 收集进列表，一次性喂给原生的 `MODEL.generate(input=chunks, batch_size_s=0, language="auto", use_itn=True)` 进行并发推理：
  ```python
  # 修改前 (for 循环 30 次单句，频繁读写)：
  # for start_ms, end_ms in opt_segs:
  #     ...
  #     res = MODEL.generate(input=chunk, cache={}, language="auto", use_itn=True)
  
  # 修改后 (1次装载GPU -> 1次Batch ASR -> 1次卸载CPU)：
  chunks = []
  for start_ms, end_ms in opt_segs:
      s_idx = int(start_ms * 16)
      e_idx = int(end_ms * 16)
      chunk = audio[max(0, s_idx-800):min(len(audio), e_idx+800)]
      if len(chunk) < 1600: continue
      chunks.append(chunk)
      
  if chunks:
      res_batch = MODEL.generate(input=chunks, batch_size_s=0, language="auto", use_itn=True)
      # 收集并过滤特殊标签
      texts = []
      for r in res_batch:
          raw = r.get('text', '').strip()
          clean = re.sub(r"<\|.*?\|>", "", raw).strip()
          if clean:
              texts.append(clean)
      raw_text = "\n".join(texts)
  ```

### Task 3: 标点模型集成与转写后处理
- [ ] 在 `asr_service.py` 启动 startup 时，将 `PUNC_MODEL` (CT-Punc 标点模型) 加载在 CPU 6 线程上。
- [ ] 在 ASR 大 Batch 推理拿到 `raw_text` 后，调用 `PUNC_MODEL.generate(input=raw_text)` 并提取文本作为最终返回。
- [ ] 将 API 路由 `/transcribe` 的 `vad_split` 参数的默认值修改为 `Form(True)`。

### Task 4: 自动化测试与验证
- [ ] 创建 `tests/test_pytorch_service_integration.py` 接口测试：
  - 测试开始时用 subprocess 自动在 `8001` 端口拉起 `asr_service.py`；
  - 使用相同长音频做转写请求，指定 `vad_split=True`；
  - 验证返回结果文本是否带有标点符号（，。！？），且测试完成后自动杀掉 `8001` 微服务。
