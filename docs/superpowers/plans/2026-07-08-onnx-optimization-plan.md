# SenseVoiceSmall ONNX GPU 优化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 funclip-pro 的 ONNX GPU 微服务大 Batch 推理加速、CPU 物理核心亲和性锁定以及后处理标点模型集成。

**Architecture:** 通过 psutil 硬锁核心在 0-3；在 ASR 包装类中重写解码循环支持 batch 并行；增加 CPU 端的 CT-Punc 标点模型后处理。

**Tech Stack:** Python, FastAPI, ONNX Runtime GPU, PyTorch, psutil

---

### Task 1: 物理核心硬绑定与 CPU 多线程资源防线限制

**Files:**
- Modify: `asr_onnx_service.py:1-42`

- [ ] **Step 1: 编写测试脚本验证核心锁定行为**

  创建新测试文件 `tests/test_affinity.py`：
  ```python
  import os
  import sys
  import psutil

  def test_cpu_affinity():
      # 验证当前进程核心数已经被限制在前 4 个逻辑核心上
      affinity = psutil.Process().cpu_affinity()
      print("Current CPU Affinity:", affinity)
      # 确认只绑定在核心 [0, 1, 2, 3] 上
      assert set(affinity).issubset({0, 1, 2, 3})
      assert len(affinity) <= 4
  ```

- [ ] **Step 2: 运行测试并确保其在未修改时失败（或跳过因为尚未注入）**

  在未绑定核心时，测试必然失败。
  运行：`E:\conda\envs\asr_ui_env\python.exe -m pytest tests/test_affinity.py -v`
  预期：FAIL (提示 `AssertionError`)

- [ ] **Step 3: 修改 asr_onnx_service.py 头部以加入 psutil 亲和性锁定和多线程库限制**

  更新 [asr_onnx_service.py](file:///E:/project/funclip-pro/asr_onnx_service.py) 的头部：
  ```python
  import os
  import psutil

  # 1. 强力锁定 CPU 核心在前 4 个上，防止 100% 跑满卡死
  try:
      psutil.Process().cpu_affinity([0, 1, 2, 3])
  except Exception as e:
      print(f"警告：设置 CPU 亲和性失败: {e}")

  # 2. 设置线程数软防线
  os.environ["OMP_NUM_THREADS"] = "4"
  os.environ["MKL_NUM_THREADS"] = "4"
  os.environ["OPENBLAS_NUM_THREADS"] = "4"
  os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
  os.environ["NUMEXPR_NUM_THREADS"] = "4"
  ```

- [ ] **Step 4: 在 `tests/test_affinity.py` 的测试开头引入 `asr_onnx_service` 并运行测试验证**

  修改 `tests/test_affinity.py`：
  ```python
  import os
  import sys
  # 导入 ASR 服务模块，这会触发头部 affinity 绑定
  import asr_onnx_service
  import psutil

  def test_cpu_affinity():
      affinity = psutil.Process().cpu_affinity()
      print("Current CPU Affinity:", affinity)
      assert set(affinity).issubset({0, 1, 2, 3})
  ```
  运行：`E:\conda\envs\asr_ui_env\python.exe -m pytest tests/test_affinity.py -v`
  预期：PASS

- [ ] **Step 5: 提交**
  ```bash
  git add asr_onnx_service.py tests/test_affinity.py
  git commit -m "feat: add cpu physical affinity binding and threads limit"
  ```

---

### Task 2: 重写 SenseVoiceSmall 包装类以支持真正的大 Batch 多句解码

**Files:**
- Modify: `asr_onnx_service.py:47-78`

- [ ] **Step 1: 编写大 Batch 解码单元测试**

  创建测试文件 `tests/test_batch_decode.py`：
  ```python
  import os
  import sys
  import numpy as np
  import torch

  sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  sys.path.append(r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmall")
  from asr_onnx_service import SenseVoiceSmall

  def test_batch_decode_multi():
      model_path = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmall-ONNX"
      # 初始化 batch_size = 2
      model = SenseVoiceSmall(model_dir=model_path, batch_size=2, quantize=True, device_id="0")
      
      # 制造 2 个短静音片段
      wave1 = np.zeros(16000, dtype=np.float32)
      wave2 = np.zeros(16000, dtype=np.float32)
      
      res = model([wave1, wave2])
      print("ASR Batch Result:", res)
      # 验证返回的结果列表长度为 2
      assert isinstance(res, list)
      assert len(res) == 2
  ```

- [ ] **Step 2: 运行测试并确信在未修改包装类前失败**

  运行：`E:\conda\envs\asr_ui_env\python.exe -m pytest tests/test_batch_decode.py -v`
  预期：FAIL (因为基类只返回了第 0 个样本结果，`len(res) == 1`，断言失败)

- [ ] **Step 3: 修改 `asr_onnx_service.py` 里的 `SenseVoiceSmall` 的 `__call__` 实现以支持多 batch**

  替换 `asr_onnx_service.py` 里的 `SenseVoiceSmall` 类的 `__call__`：
  ```python
      def __call__(self, wav_content, language=[0], textnorm=[15], tokenizer=None, **kwargs):
          if tokenizer is None:
              # 内部默认 Tokenizer
              class DefaultTokenizer:
                  def __init__(self, tokens):
                      self.tokens = tokens
                  def tokens2text(self, ids):
                      res = []
                      for i in ids:
                          t = self.tokens[i]
                          if t.startswith("<|") and t.endswith("|>"):
                              continue
                          if t == "<space>":
                              res.append(" ")
                          elif t == "<unk>":
                              continue
                          else:
                              res.append(t)
                      return "".join(res)
              tokenizer = DefaultTokenizer(self.tokens)

          # 核心代码：重写底层以支持 batch_size > 1 时的遍历解码
          import numpy as np
          waveform_list = self.load_data(wav_content, self.frontend.opts.frame_opts.samp_freq)
          waveform_nums = len(waveform_list)
          asr_res = []
          
          for beg_idx in range(0, waveform_nums, self.batch_size):
              end_idx = min(waveform_nums, beg_idx + self.batch_size)
              feats, feats_len = self.extract_feat(waveform_list[beg_idx:end_idx])
              ctc_logits, encoder_out_lens = self.infer(
                  feats, 
                  feats_len, 
                  np.array(language, dtype=np.int32), 
                  np.array(textnorm, dtype=np.int32)
              )
              # 返回 torch.Tensor 便于按维度解析
              ctc_logits = torch.from_numpy(ctc_logits).float()
              
              # 支持多 batch 的结果解析
              for b in range(end_idx - beg_idx):
                  x = ctc_logits[b, : encoder_out_lens[b].item(), :]
                  yseq = x.argmax(dim=-1)
                  yseq = torch.unique_consecutive(yseq, dim=-1)

                  mask = yseq != self.blank_id
                  token_int = yseq[mask].tolist()
                  
                  asr_res.append(tokenizer.tokens2text(token_int))
          return asr_res
  ```

- [ ] **Step 4: 运行测试验证大 batch 解码通过**

  运行：`E:\conda\envs\asr_ui_env\python.exe -m pytest tests/test_batch_decode.py -v`
  预期：PASS (成功输出两个包含空字符串的识别结果，断言成功)

- [ ] **Step 5: Commit**
  ```bash
  git add asr_onnx_service.py tests/test_batch_decode.py
  git commit -m "feat: rewrite SenseVoiceSmall __call__ to support batch decoding"
  ```

---

### Task 3: API 推理接口大 Batch 化、标点模型集成与流水线重构

**Files:**
- Modify: `asr_onnx_service.py:93-242`

- [ ] **Step 1: 编写 API 接口集成测试**

  创建测试文件 `tests/test_onnx_service_integration.py`：
  ```python
  import requests
  import time

  def test_transcribe_api():
      url = "http://127.0.0.1:8002/transcribe"
      audio_file = r"E:\下载\下载\李雪花2.wav"
      
      # 运行 API 测试请求，并指定 vad_split 为 true 触发大 batch 标点逻辑
      with open(audio_file, "rb") as f:
          files = {"file": f}
          data = {"vad_split": "true"}
          resp = requests.post(url, files=files, data=data)
          
      assert resp.status_code == 200
      res_json = resp.json()
      assert "text" in res_json
      assert len(res_json["text"]) > 0
      
      # 验证结果中是否带有了标点符号（，。！？）
      text = res_json["text"]
      print("ASR Result Sample:", text[:100])
      has_punc = any(p in text for p in ["，", "。", "？", "！"])
      assert has_punc, "转写文本未带上任何标点符号"
  ```

- [ ] **Step 2: 修改 `asr_onnx_service.py` 引入标点模型加载和多 Batch 流水线**

  1. 在 `asr_onnx_service.py` 头部全局变量区声明并在 `load_models()` 中加载 `PUNC_MODEL`：
     ```python
     MODEL = None
     VAD_MODEL = None
     PUNC_MODEL = None  # 新增标点模型全局变量
     
     @app.on_event("startup")
     def load_models():
         global MODEL, VAD_MODEL, PUNC_MODEL
         model_path = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmall-ONNX"
         vad_path = r"E:\project\funclip-pro\model\models\damo\speech_fsmn_vad_zh-cn-16k-common-pytorch"
         punc_path = r"E:\project\funclip-pro\model\models\damo\punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
         
         logger.info("正在加载 ONNX GPU ASR 模型、CPU VAD 模型和 CPU 标点模型...")
         try:
             # 1. 加载 ASR (将 batch_size 修改为 16 以适配大 batch 推理)
             MODEL = SenseVoiceSmall(
                 model_dir=model_path,
                 batch_size=16,
                 quantize=True,
                 device_id="0",
                 intra_op_num_threads=4
             )
             # 2. 加载 VAD
             VAD_MODEL = AutoModel(
                 model=vad_path,
                 trust_remote_code=True,
                 device="cpu",
                 disable_update=True,
                 disable_pbar=True
             )
             VAD_MODEL.model.to("cpu")
             VAD_MODEL.kwargs["device"] = "cpu"
             
             # 3. 加载 PUNC 标点模型
             PUNC_MODEL = AutoModel(
                 model=punc_path,
                 trust_remote_code=True,
                 device="cpu",
                 disable_update=True,
                 disable_pbar=True
             )
             PUNC_MODEL.model.to("cpu")
             PUNC_MODEL.kwargs["device"] = "cpu"
             
             logger.info("所有模型加载成功！")
         except Exception as e:
             logger.error(f"模型加载失败: {e}")
             raise e
     ```

  2. 重新实现 `_run_inference` 推理管线，修改为大 Batch 推理 + 标点后处理：
     ```python
     def _run_inference(audio_path: str, vad_split: bool = True) -> str:
         """在独立线程中运行的同步推理逻辑，默认支持开启 VAD"""
         if not vad_split:
             res = MODEL(audio_path)
             if res and len(res) > 0:
                 raw_text = res[0].strip()
                 clean_text = re.sub(r"<\|.*?\|>", "", raw_text).strip()
                 return clean_text
             return ""
         else:
             import librosa
             audio, _ = librosa.load(audio_path, sr=16000)
             
             # 1. 运行 VAD 切分
             vad_out = VAD_MODEL.generate(input=audio_path, batch_size_s=5000, max_single_segment_time=60000)
             raw_segs = vad_out[0]['value'] if vad_out and len(vad_out) > 0 and 'value' in vad_out[0] else [[0, len(audio)/16*1000]]
             
             # 2. 合并段
             def _merge_vad_segments(segments, max_gap_ms=300, max_duration_ms=8000):
                 if not segments: return []
                 merged = []
                 curr_start, curr_end = segments[0]
                 for next_start, next_end in segments[1:]:
                     gap = next_start - curr_end
                     duration = (curr_end - curr_start) + (next_end - next_start)
                     if gap < max_gap_ms and duration < max_duration_ms:
                         curr_end = next_end 
                     else:
                         merged.append([curr_start, curr_end]) 
                         curr_start, curr_end = next_start, next_end
                 merged.append([curr_start, curr_end])
                 return merged
                 
             opt_segs = _merge_vad_segments(raw_segs)
             
             # 3. 收集所有音频切片，打包准备批量输入
             chunks = []
             for start_ms, end_ms in opt_segs:
                 s_idx = int(start_ms * 16)
                 e_idx = int(end_ms * 16)
                 chunk = audio[max(0, s_idx-800):min(len(audio), e_idx+800)]
                 if len(chunk) < 1600: continue
                 chunks.append(chunk)
                 
             if not chunks:
                 return ""
                 
             # 一次性调用 ASR 模型并行处理整个批次
             texts = MODEL(chunks)
             
             clean_texts = []
             for t in texts:
                 clean = re.sub(r"<\|.*?\|>", "", t).strip()
                 if clean:
                     clean_texts.append(clean)
                     
             raw_text = "\n".join(clean_texts)
             
             # 4. 后处理加回标点符号
             if PUNC_MODEL is not None and raw_text.strip():
                 try:
                     punc_out = PUNC_MODEL.generate(input=raw_text)
                     if punc_out and len(punc_out) > 0:
                         raw_text = punc_out[0].get('text', raw_text)
                 except Exception as punc_err:
                     logger.error(f"标点符号后处理失败: {punc_err}")
                     
             return raw_text
     ```

  3. 将 FastAPI 的 `/transcribe` 接口的 `vad_split: bool = Form(False)` 修改为 `Form(True)` 以默认开启 VAD。

- [ ] **Step 3: 启动 ASR 微服务进行集成验证**

  启动服务进程：`E:\conda\envs\asr_ui_env\python.exe asr_onnx_service.py` (将其后台挂载或者等待其启动完成，在 8002 端口)。
  运行集成测试：`E:\conda\envs\asr_ui_env\python.exe -m pytest tests/test_onnx_service_integration.py -v`
  预期：接口调用成功返回 200，并获得带有合理标点符号（，。！？）的识别文本。

- [ ] **Step 4: 提交**
  ```bash
  git add asr_onnx_service.py tests/test_onnx_service_integration.py
  git commit -m "feat: complete batch pipeline inference and integrate CT-Punc model"
  ```
