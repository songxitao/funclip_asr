# 8001 端口 PyTorch 原生服务优化与标点模型集成设计方案

## 1. 目标与背景

由于当前显卡显存极度紧张（12GB VRAM），原生的 PyTorch 模型（SenseVoiceSmall 等）在运行大模型工作流时，若不及时释放显存极易发生 CUDA OOM。

原有的 `asr_service.py` 在长音频 VAD 切句模式下，采用 `for` 循环逐句推理，使得模型在 GPU 与 CPU 之间频繁地进行移入和卸载（对于包含 30 句音频片段的文件，会执行 30 次模型加载/卸载与显存清空）。这不仅严重降低了推理效率，还是导致 CPU 疯跑和死机的核心因素。

本方案旨在对监听 8001 端口的原生 PyTorch ASR 服务 `asr_service.py` 进行重构，实现以下目标：
1. **CPU 锁死防范**：硬锁定进程 CPU 亲和性至 6 个大物理核心，限制 PyTorch 底层多线程数量，消除 CPU 满载假死风险。
2. **大 Batch 推理管线**：将 ASR 推理重构为“1 次 GPU 装载 -> 1 次多 Batch 推理 -> 1 次 GPU 卸载及显存清空”的高效流水线。
3. **标点模型加回**：在启动时载入本地 CT-Punc 模型并部署在 CPU 上运行，进行后处理，使文本还原完美排版标点。
4. **效果与性能比对测试**：通过对比 8001 端口（PyTorch-GPU 服务）与 8002 端口（ONNX-GPU 服务），量化分析加速比与字错吻合度 (CER)。

---

## 2. 详细设计

### 2.1 物理核心与线程开销限制

在 `asr_service.py` 的脚本最头部引入 CPU 亲和性硬锁定和底层多线程限制，防止在 CPU 侧运行 VAD 和标点模型时产生资源抢占：

```python
import os
import psutil

# 1. 绑定当前进程亲和性至前 6 个大核心 [0, 1, 2, 3, 4, 5]
try:
    psutil.Process().cpu_affinity([0, 1, 2, 3, 4, 5])
except Exception as e:
    print(f"亲和性设置失败: {e}")

# 2. 限制各底层矩阵库在核心内开辟的线程数
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"

import torch
torch.set_num_threads(6)
```

### 2.2 标点模型启动载入

在 `@app.on_event("startup")` 事件的 `load_models()` 中：
*   增加全局变量 `PUNC_MODEL`。
*   在 CPU 上以 6 线程限制加载本地 CT-Punc 标点模型。模型路径：
    `E:\project\funclip-pro\model\models\damo\punc_ct-transformer_zh-cn-common-vocab272727-pytorch`。

```python
global MODEL, VAD_MODEL, PUNC_MODEL
punc_path = r"E:\project\funclip-pro\model\models\damo\punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

try:
    # ... 原有 MODEL 与 VAD_MODEL 载入 ...
    
    # 3. 加载标点模型在 CPU 上
    PUNC_MODEL = AutoModel(
        model=punc_path,
        trust_remote_code=True,
        device="cpu",
        disable_update=True
    )
    logger.info("标点模型加载成功！")
except Exception as e:
    logger.error(f"模型加载失败: {e}")
    raise e
```

### 2.3 大 Batch 合并推理与显存极速释放

重构推理主体 `_run_inference(audio_path, vad_split)`。具体变化点：
1. **显存生命周期收敛**：`MODEL.model.to("cuda")` 和 `VAD_MODEL.model.to("cuda")`（若开启 VAD）在 try 的最开始移入 GPU；在 `finally` 块中执行移回 CPU 与显存清空（`torch.cuda.empty_cache()`）。
2. **列表拼装与 Batch 喂入**：
   - 提取 VAD 切片，将原来的循环 generate 改为将所有 numpy 形式 of 音频 chunk 存入 `chunks` 列表。
   - 一次性传入模型推理：
     ```python
     if chunks:
         res_batch = MODEL.generate(
             input=chunks,
             batch_size_s=0,
             disable_pbar=True,
             language="auto",
             use_itn=True
         )
         texts = []
         for r in res_batch:
             raw = r.get('text', '').strip()
             clean = re.sub(r"<\|.*?\|>", "", raw).strip()
             if clean:
                 texts.append(clean)
         raw_text = "\n".join(texts)
     ```
3. **标点后处理后置**：
   - ASR 推理生成合并后的 `raw_text` 后，在 CPU 侧运行标点模型加上标点。
     ```python
     if raw_text:
         punc_res = PUNC_MODEL.generate(input=raw_text)
         if punc_res and len(punc_res) > 0:
             final_text = punc_res[0].get('text', '').strip()
         else:
             final_text = raw_text
     else:
         final_text = ""
     ```

### 2.4 API 默认路由改动

为了使转写更具健壮性并支持长音频入库，将路由中 `vad_split` 的默认值由 `Form(False)` 修改为 `Form(True)`：

```python
@app.post("/transcribe")
async def transcribe(
    request: Request, 
    file: UploadFile = File(...), 
    vad_split: bool = Form(True)  # 默认开启 VAD
):
```

---

## 3. 验证计划与测试设计

### 3.1 Pytorch 服务集成测试

编写 `tests/test_pytorch_service_integration.py`。
*   **启动服务**：通过 `subprocess.Popen` 启动 `python asr_service.py` 服务并监听 8001 端口。
*   **请求与校验**：使用 `requests` 上传 `E:\下载\下载\李雪花2.wav` 并发送 `/transcribe` 请求。
*   **校验指标**：
    1. 接口返回 HTTP 200，JSON 中包含 "text" 属性。
    2. 返回的文本中确实包含了中文标点符号（验证标点集成）。
    3. 清理：自动 terminate 并 wait 后台 ASR 服务，防止显存和端口遗留占用。

### 3.2 双服务（8001 vs 8002）对照测试

编写 `tests/test_asr_comparison.py`，用于直接在终端执行，展示性能及吻合度报表。
*   **流程**：
    1. 分别启动（或假设后台已启动）8001 和 8002 端口的微服务。
    2. 采用测试音频 `E:\下载\下载\李雪花2.wav`。
    3. 调用 8001 (PyTorch) 接口，记录耗时 $t_1$ 并获取结果文本 $Text_1$。
    4. 调用 8002 (ONNX) 接口，记录耗时 $t_2$ 并获取结果文本 $Text_2$。
    5. 计算 Speedup 加加速比：$Speedup = \frac{t_1}{t_2}$。
    6. **文字吻合度算法**：编写轻量级的 Levenshtein 编辑距离算法（无需额外 pip 依赖），计算字符错误率 (CER)。
       $$CER = \frac{EditDistance(Text_1, Text_2)}{max(len(Text_1), len(Text_2))}$$
    7. 输出易读的控制台性能与字错吻合度对比表格。
