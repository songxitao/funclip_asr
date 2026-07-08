# ASR 原生 PyTorch 服务 (8001端口) 性能优化与对照测试实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成 8001 端口原生 PyTorch ASR 服务 `asr_service.py` 的重构（锁定大 CPU 6 核心、大 Batch 一次性推理、CT-Punc 标点模型集成与显存极致释放），并实现原生 vs ONNX 效果与字错率 (CER) 的对照测试。

**Architecture:** 
1. 通过 `psutil` 强制锁定进程 CPU 亲和性，并通过设置全局 `torch.set_num_threads(6)` 和环境变量控制 CPU 多线程膨胀。
2. startup 时加载 CT-Punc 标点模型在 CPU 上运行。
3. 重构 `asr_service.py` 中的 `_run_inference` 接口，将长音频片段收集进 `chunks` 列表，一次性移到 GPU 进行 Batch 推理，并在 `finally` 块中立即释放显存回 CPU 并清空显存缓存，推理后使用标点模型对文本后处理。
4. 编写完整的双端对照测试脚本，计算字错率 (CER) 和加速比，使用 Levenshtein 动态规划算法计算文本距离。

**Tech Stack:** Python 3.10, FastAPI, PyTorch (FunASR), psutil, requests, pytest

---

### Task 1: 物理核心硬绑定与 CPU 多线程资源限制

**Files:**
- Modify: `asr_service.py`

- [ ] **Step 1: 编写测试用例验证 `asr_service.py` 内部的亲和性和线程设置**

  创建: `tests/test_pytorch_affinity.py`
  ```python
  import os
  import psutil
  import torch

  def test_cpu_affinity_and_threads():
      # 模拟运行 asr_service 内部的设置，检查亲和性
      affinity = psutil.Process().cpu_affinity()
      # 检查是否绑定到前 6 个核心中的一部分或者全部
      assert len(affinity) == 6 or set(affinity).issubset({0, 1, 2, 3, 4, 5})
      # 检查 torch 线程限制
      assert torch.get_num_threads() == 6
  ```

- [ ] **Step 2: 运行测试以验证失败**

  在终端运行测试（由于 `asr_service.py` 尚未导入绑定，我们这里可以直接运行该测试）：
  Run: `pytest tests/test_pytorch_affinity.py -v`
  Expected: FAIL (因为目前没有绑定，`psutil.Process().cpu_affinity()` 包含全部 32 个逻辑处理器，且 torch 默认线程数大于 6)

- [ ] **Step 3: 修改 `asr_service.py` 头部写入物理核心硬绑定与线程限制**

  在 `asr_service.py` 的第 1 行之前注入以下代码：
  ```python
  import os
  import psutil

  # 硬锁定 CPU 物理核心至前 6 个大核心 [0, 1, 2, 3, 4, 5]
  try:
      psutil.Process().cpu_affinity([0, 1, 2, 3, 4, 5])
  except Exception as e:
      print(f"CPU 亲和性设置失败: {e}")

  # 限制底层计算库的多线程并发数量以节省 CPU 算力，防止疯跑锁死
  os.environ["OMP_NUM_THREADS"] = "6"
  os.environ["MKL_NUM_THREADS"] = "6"
  os.environ["OPENBLAS_NUM_THREADS"] = "6"

  import torch
  torch.set_num_threads(6)
  ```

- [ ] **Step 4: 运行测试以验证通过**

  Run: `pytest tests/test_pytorch_affinity.py -v`
  Expected: PASS

- [ ] **Step 5: 提交代码**

  Run:
  ```bash
  git add asr_service.py tests/test_pytorch_affinity.py
  git commit -m "perf: 锁定 asr_service.py CPU 亲和性为前 6 核并限制 torch 为 6 线程"
  ```

---

### Task 2: Startup 阶段加载标点模型

**Files:**
- Modify: `asr_service.py`

- [ ] **Step 1: 编写加载标点模型的验证测试**

  创建: `tests/test_pytorch_punc_load.py`
  ```python
  def test_punc_model_loaded():
      import asr_service
      # 模拟执行 startup 事件
      asr_service.load_models()
      assert asr_service.PUNC_MODEL is not None
      assert asr_service.PUNC_MODEL.device == "cpu"
  ```

- [ ] **Step 2: 运行测试以验证失败**

  Run: `pytest tests/test_pytorch_punc_load.py -v`
  Expected: FAIL (因为 `PUNC_MODEL` 尚未在 `asr_service.py` 中定义且未被加载)

- [ ] **Step 3: 修改 `asr_service.py` 加载标点模型**

  在 `asr_service.py:27-30` 定义全局变量：
  ```python
  MODEL = None
  VAD_MODEL = None
  PUNC_MODEL = None  # 新增标点模型全局变量
  ```

  修改 `asr_service.py` 的 `load_models()` 函数（大约在第 33 行）：
  ```python
  @app.on_event("startup")
  def load_models():
      global MODEL, VAD_MODEL, PUNC_MODEL
      model_path = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmall"
      vad_path = r"E:\project\funclip-pro\model\models\damo\speech_fsmn_vad_zh-cn-16k-common-pytorch"
      punc_path = r"E:\project\funclip-pro\model\models\damo\punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
      
      logger.info("正在 GPU (CUDA) 上加载 ASR 和 VAD 模型，在 CPU 上加载标点模型...")
      try:
          # 1. 加载 ASR 语音识别模型
          MODEL = AutoModel(
              model=model_path,
              trust_remote_code=True,
              device="cpu",
              disable_update=True
          )
          # 2. 加载 VAD 语音活动检测模型（用于长音频切句）
          VAD_MODEL = AutoModel(
              model=vad_path,
              trust_remote_code=True,
              device="cpu",
              disable_update=True,
              disable_pbar=True
          )
          # 3. 加载标点模型（锁定在 CPU 上，避免抢占 GPU 显存）
          PUNC_MODEL = AutoModel(
              model=punc_path,
              trust_remote_code=True,
              device="cpu",
              disable_update=True
          )
          logger.info("ASR、VAD 和标点模型全部加载成功！")
      except Exception as e:
          logger.error(f"模型加载失败: {e}")
          raise e
  ```

- [ ] **Step 4: 运行测试以验证通过**

  Run: `pytest tests/test_pytorch_punc_load.py -v`
  Expected: PASS

- [ ] **Step 5: 提交代码**

  Run:
  ```bash
  git add asr_service.py tests/test_pytorch_punc_load.py
  git commit -m "feat: 在 startup 中实现标点模型 PUNC_MODEL 载入"
  ```

---

### Task 3: 推理主体大 Batch 重构与标点后处理

**Files:**
- Modify: `asr_service.py`

- [ ] **Step 1: 编写推理主体改动的单元测试**

  创建: `tests/test_pytorch_inference_refactor.py`
  ```python
  import pytest
  import asr_service

  def test_run_inference_with_punc():
      # 先确保模型加载
      if asr_service.MODEL is None:
          asr_service.load_models()
      # 输入测试文件
      audio_path = r"E:\下载\下载\李雪花2.wav"
      # 执行带有 VAD 切分和大 Batch 的推理
      text = asr_service._run_inference(audio_path, vad_split=True)
      # 验证输出不为空且成功加上了标点符号（包含逗号或句号等字符）
      assert len(text) > 0
      # 校验标点符号
      assert any(p in text for p in ["，", "。", "？", "！", ",", "."])
  ```

- [ ] **Step 2: 运行测试以验证失败**

  Run: `pytest tests/test_pytorch_inference_refactor.py -v`
  Expected: FAIL (因为推理函数中仍未集成大 Batch 推理和标点还原后处理)

- [ ] **Step 3: 修改 `asr_service.py` 中的 `_run_inference` 函数**

  重构 `_run_inference` 函数（大约在原第 60 行开始）：
  ```python
  def _run_inference(audio_path: str, vad_split: bool = False) -> str:
      """在独立线程中运行的同步推理逻辑，支持大 Batch 合并推理与 CPU 标点后处理"""
      global MODEL, VAD_MODEL, PUNC_MODEL
      MODEL.model.to("cuda")
      MODEL.kwargs["device"] = "cuda"
      if vad_split:
          VAD_MODEL.model.to("cuda")
          VAD_MODEL.kwargs["device"] = "cuda"
          
      try:
          if not vad_split:
              # --- 轨道一：极速单句模式 (适合 Dify 语音对话提问) ---
              res = MODEL.generate(input=audio_path, cache={}, language="auto", use_itn=True)
              if res and len(res) > 0:
                  raw_text = res[0].get('text', '').strip()
                  # 过滤情绪/事件富文本标签
                  clean_text = re.sub(r"<\|.*?\|>", "", raw_text).strip()
                  # 即使是单句，也使用标点模型进行后处理
                  if clean_text:
                      punc_res = PUNC_MODEL.generate(input=clean_text)
                      if punc_res and len(punc_res) > 0:
                          return punc_res[0].get('text', '').strip()
                      return clean_text
              return ""
          else:
              # --- 轨道二：长音频 VAD 切句模式 (适合 Dify 知识库索引入库) ---
              import librosa
              
              # 1. 加载音频波形
              audio, _ = librosa.load(audio_path, sr=16000)
              
              # 2. 运行 VAD 切分得到静音区间
              vad_out = VAD_MODEL.generate(input=audio_path, batch_size_s=5000, max_single_segment_time=60000)
              raw_segs = vad_out[0]['value'] if vad_out and len(vad_out) > 0 and 'value' in vad_out[0] else [[0, len(audio)/16*1000]]
              
              # 3. 合并小静音切片，保证每段大约 8 秒以内
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
              
              # 4. 收集所有音频 chunk 进行一次性大 Batch 合并推理，避免多次显存重复装载
              chunks = []
              for start_ms, end_ms in opt_segs:
                  s_idx = int(start_ms * 16)
                  e_idx = int(end_ms * 16)
                  chunk = audio[max(0, s_idx-800):min(len(audio), e_idx+800)]
                  if len(chunk) < 1600: continue
                  chunks.append(chunk)
                  
              raw_text = ""
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
                  
              # 5. 标点模型后处理 (在 CPU 上进行)
              if raw_text:
                  punc_res = PUNC_MODEL.generate(input=raw_text)
                  if punc_res and len(punc_res) > 0:
                      final_text = punc_res[0].get('text', '').strip()
                  else:
                      final_text = raw_text
              else:
                  final_text = ""
                  
              return final_text
      finally:
          # 6. 一次性将模型移回 CPU 并清空显存缓存，极致优化释放显存
          MODEL.model.to("cpu")
          MODEL.kwargs["device"] = "cpu"
          if vad_split:
              VAD_MODEL.model.to("cpu")
              VAD_MODEL.kwargs["device"] = "cpu"
          torch.cuda.empty_cache()
  ```

- [ ] **Step 4: 运行测试以验证通过**

  Run: `pytest tests/test_pytorch_inference_refactor.py -v`
  Expected: PASS

- [ ] **Step 5: 提交代码**

  Run:
  ```bash
  git add asr_service.py tests/test_pytorch_inference_refactor.py
  git commit -m "perf: 重构 _run_inference 实现大 Batch 合并推理与标点后处理，可靠释放 GPU 显存"
  ```

---

### Task 4: API 路由与默认 VAD 状态修改

**Files:**
- Modify: `asr_service.py`

- [ ] **Step 1: 编写接口路由参数默认值测试**

  创建: `tests/test_pytorch_route.py`
  ```python
  from fastapi.testclient import TestClient
  import asr_service

  def test_transcribe_route_default_vad():
      client = TestClient(asr_service.app)
      # 检查 API 的 signature，确保 vad_split 参数的默认值是 True
      import inspect
      sig = inspect.signature(asr_service.transcribe)
      param = sig.parameters.get("vad_split")
      assert param is not None
      # 校验 Form 默认值为 True
      assert param.default.default is True
  ```

- [ ] **Step 2: 运行测试以验证失败**

  Run: `pytest tests/test_pytorch_route.py -v`
  Expected: FAIL (因为目前 `vad_split: bool = Form(False)`)

- [ ] **Step 3: 修改 `asr_service.py` 中的 `transcribe` 接口定义**

  修改 `asr_service.py` 的 `/transcribe` 路由声明（大约在原第 131 行）：
  ```python
  @app.post("/transcribe")
  async def transcribe(
      request: Request, 
      file: UploadFile = File(...), 
      vad_split: bool = Form(True)  # 修改默认值为 True，开启 VAD 切分
  ):
  ```

- [ ] **Step 4: 运行测试以验证通过**

  Run: `pytest tests/test_pytorch_route.py -v`
  Expected: PASS

- [ ] **Step 5: 提交代码**

  Run:
  ```bash
  git add asr_service.py tests/test_pytorch_route.py
  git commit -m "feat: 修改 /transcribe 路由 vad_split 默认值为 Form(True)"
  ```

---

### Task 5: 完整的 PyTorch 服务端集成测试

**Files:**
- Create: `tests/test_pytorch_service_integration.py`

- [ ] **Step 1: 编写集成测试，包含完整的服务启停、API 发送与标点文本校验**

  创建新文件 `tests/test_pytorch_service_integration.py`：
  ```python
  import subprocess
  import time
  import requests
  import pytest
  import os

  @pytest.fixture(scope="module", autouse=True)
  def run_pytorch_service():
      # 使用 conda 环境中的 python 启动 asr_service.py 并设置环境变量
      env = os.environ.copy()
      process = subprocess.Popen(
          [r"E:\conda\envs\asr_ui_env\python.exe", "asr_service.py"],
          creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
          env=env
      )
      # 等待 15 秒加载模型
      time.sleep(15)
      yield
      # 优雅终止
      process.terminate()
      try:
          process.wait(timeout=5)
      except subprocess.TimeoutExpired:
          process.kill()

  def test_pytorch_transcribe_api():
      url = "http://127.0.0.1:8001/transcribe"
      audio_file = r"E:\下载\下载\李雪花2.wav"
      
      with open(audio_file, "rb") as f:
          files = {"file": (os.path.basename(audio_file), f, "audio/wav")}
          response = requests.post(url, files=files, data={"vad_split": "true"})
          
      assert response.status_code == 200
      res_json = response.json()
      assert "text" in res_json
      text = res_json["text"]
      print(f"\n[PyTorch-GPU 集成测试转写结果前100字]: {text[:100]}")
      
      # 校验文本中必须成功包含中文标点符号
      assert any(p in text for p in ["，", "。", "？", "！"])
  ```

- [ ] **Step 2: 运行集成测试验证其能正确通过**

  Run: `pytest tests/test_pytorch_service_integration.py -s -v`
  Expected: PASS，并在输出中看到转写的文本片段。

- [ ] **Step 3: 提交代码**

  Run:
  ```bash
  git add tests/test_pytorch_service_integration.py
  git commit -m "test: 添加 PyTorch-GPU 服务端 API 集成测试"
  ```

---

### Task 6: 双服务 ASR 原生 vs ONNX 对照测试脚本

**Files:**
- Create: `tests/test_asr_comparison.py`

- [ ] **Step 1: 编写性能耗时与 CER (字错吻合度) 对比脚本**

  编写轻量级编辑距离算法，向 8001 和 8002 接口发送请求，生成控制台报告。
  创建新文件 `tests/test_asr_comparison.py`：
  ```python
  import os
  import time
  import requests

  def compute_levenshtein_distance(s1: str, s2: str) -> int:
      """纯 Python 实现编辑距离，防止外部三方库依赖缺失"""
      if len(s1) < len(s2):
          return compute_levenshtein_distance(s2, s1)
      if len(s2) == 0:
          return len(s1)

      previous_row = range(len(s2) + 1)
      for i, c1 in enumerate(s1):
          current_row = [i + 1]
          for j, c2 in enumerate(s2):
              insertions = previous_row[j + 1] + 1
              deletions = current_row[j] + 1
              substitutions = previous_row[j] + (c1 != c2)
              current_row.append(min(insertions, deletions, substitutions))
          previous_row = current_row

      return previous_row[-1]

  def run_comparison():
      pytorch_url = "http://127.0.0.1:8001/transcribe"
      onnx_url = "http://127.0.0.1:8002/transcribe"
      audio_file = r"E:\下载\下载\李雪花2.wav"
      
      if not os.path.exists(audio_file):
          print(f"❌ 找不到测试音频文件: {audio_file}")
          return

      print(f"🎬 开始比对测试...")
      print(f"测试音频: {audio_file}")
      print(f"音频大小: {os.path.getsize(audio_file) / 1024 / 1024:.2f} MB")
      print("-" * 60)

      # 1. 调用 PyTorch-GPU 服务
      print("⏳ 发起 PyTorch-GPU 转写请求 (8001 端口)...")
      t1_start = time.time()
      try:
          with open(audio_file, "rb") as f:
              files = {"file": (os.path.basename(audio_file), f, "audio/wav")}
              r1 = requests.post(pytorch_url, files=files, data={"vad_split": "true"})
          t1_end = time.time()
          latency_pytorch = (t1_end - t1_start) * 1000
          if r1.status_code == 200:
              text_pytorch = r1.json().get("text", "").strip()
              print(f"✅ PyTorch-GPU 请求成功! 耗时: {latency_pytorch:.2f} ms")
          else:
              print(f"❌ PyTorch-GPU 响应失败: {r1.status_code} - {r1.text}")
              return
      except Exception as e:
          print(f"❌ 无法连接 PyTorch-GPU 服务 (请确保已启动 asr_service.py): {e}")
          return

      # 2. 调用 ONNX-GPU 服务
      print("\n⏳ 发起 ONNX-GPU 转写请求 (8002 端口)...")
      t2_start = time.time()
      try:
          with open(audio_file, "rb") as f:
              files = {"file": (os.path.basename(audio_file), f, "audio/wav")}
              r2 = requests.post(onnx_url, files=files, data={"vad_split": "true"})
          t2_end = time.time()
          latency_onnx = (t2_end - t2_start) * 1000
          if r2.status_code == 200:
              text_onnx = r2.json().get("text", "").strip()
              print(f"✅ ONNX-GPU 请求成功! 耗时: {latency_onnx:.2f} ms")
          else:
              print(f"❌ ONNX-GPU 响应失败: {r2.status_code} - {r2.text}")
              return
      except Exception as e:
          print(f"❌ 无法连接 ONNX-GPU 服务 (请确保已启动 asr_onnx_service.py): {e}")
          return

      # 3. 数据分析与对照报告
      print("\n" + "=" * 25 + " 对照测试报告 " + "=" * 25)
      print(f"1. PyTorch-GPU 耗时: {latency_pytorch / 1000:.2f} s")
      print(f"2. ONNX-GPU 耗时:   {latency_onnx / 1000:.2f} s")
      speedup = latency_pytorch / latency_onnx
      print(f"3. 性能加速比 (PyTorch / ONNX): {speedup:.2f}x")

      # 计算字符差异
      # 为避免标点符号差异干扰字错对比，先移除所有的标点与空白符进行核心识别率对比
      clean_p = "".join(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", text_pytorch))
      clean_o = "".join(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", text_onnx))

      edit_dist = compute_levenshtein_distance(clean_p, clean_o)
      max_len = max(len(clean_p), len(clean_o))
      cer = (edit_dist / max_len) * 100 if max_len > 0 else 0.0
      print(f"4. 去标点文字编辑距离: {edit_dist} 字")
      print(f"5. 字符错误率 (CER / 差异度): {cer:.2f}%")
      print(f"6. 文本字数比较: PyTorch {len(clean_p)} 字 vs ONNX {len(clean_o)} 字")

      print("\n--- 文本前100字展示 ---")
      print(f"PyTorch:\n{text_pytorch[:100]}...")
      print(f"\nONNX:\n{text_onnx[:100]}...")
      print("=" * 64)

  if __name__ == "__main__":
      import re
      run_comparison()
  ```

- [ ] **Step 2: 运行测试并查看其成功**

  为便于在没有单独拉起两个后台服务的测试环境中进行部分模拟或在完整启动服务后运行。我们在此确保该文件可以被直接执行。
  在终端执行：`python -m pytest tests/test_pytorch_service_integration.py`（这是前置集成测试）并在两个服务都存活状态下运行 `python tests/test_asr_comparison.py`。
  Expected: 能打印出两个服务的对比报表，且差异度（CER）应该在可接受范围内（例如 < 5%）。

- [ ] **Step 3: 提交代码**

  Run:
  ```bash
  git add tests/test_asr_comparison.py
  git commit -m "test: 添加双服务 ASR 性能与字错吻合度 (CER) 对照测试脚本"
  ```
