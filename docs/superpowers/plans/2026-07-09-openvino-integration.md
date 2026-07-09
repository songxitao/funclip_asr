# SenseVoiceSmall OpenVINO Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 funclip-pro 项目中集成 Intel OpenVINO 推理引擎以替代 ONNX Runtime，并硬限制进程 6 核绑定及 6 线程环境变量，最大化 CPU 推理速度。

**Architecture:** 在 `asr_onnx_service.py` 头部通过 `psutil` 硬绑定 CPU 前 6 个核心，并在模型初始化中利用 `openvino.Core` 代替 ONNX Runtime 编译 ONNX 模型；重写 `infer` 方法以直接调用 `CompiledModel` 并获取推理结果；修复并跑通 OpenVINO 跑评脚本的特征输入维度为 560；补齐核心绑定、大 batch 标点微服务集成的相关单元测试与接口测试。

**Tech Stack:** Python, OpenVINO, FastAPI, psutil, pytest, NumPy

---

### Task 1: 物理核心硬绑定与 CPU 多线程环境变量限制

**Files:**
- Create: `tests/test_affinity.py`
- Modify: `asr_onnx_service.py`

- [ ] **Step 1: 编写物理绑核校验单元测试**

  创建测试文件 `tests/test_affinity.py`：
  ```python
  import os
  import sys
  import psutil
  import pytest

  def test_cpu_affinity():
      # 强行导入 ASR 服务，这应该会触发服务头部的绑核逻辑
      sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
      import asr_onnx_service
      
      affinity = psutil.Process().cpu_affinity()
      print("Current CPU Affinity:", affinity)
      # 确认进程运行核心被硬限制在前 6 个逻辑核心 [0, 1, 2, 3, 4, 5] 上
      assert set(affinity).issubset({0, 1, 2, 3, 4, 5})
      assert len(affinity) <= 6
  ```

- [ ] **Step 2: 运行测试并确保其失败**

  运行命令：
  ```bash
  E:\conda\envs\asr_ui_env\python.exe -m pytest tests/test_affinity.py -v
  ```
  预期输出：由于 `asr_onnx_service.py` 还未修改，当前进程会拥有所有 CPU 逻辑核心权限，断言失败（FAIL 提示 `AssertionError`）。

- [ ] **Step 3: 修改 asr_onnx_service.py 头部以添加 psutil 亲和性锁定和环境变量限制**

  在 [asr_onnx_service.py](file:///E:/project/funclip-pro/asr_onnx_service.py) 头部首行插入以下代码：
  ```python
  import os
  import psutil

  # 1. 强力锁定 CPU 核心在前 6 个上，防止核心频繁切换开销与卡死
  try:
      psutil.Process().cpu_affinity([0, 1, 2, 3, 4, 5])
  except Exception as e:
      print(f"警告：设置 CPU 亲和性失败: {e}")

  # 2. 设置多线程计算库的环境变量软防线为 6，防止 CPU 线程无秩序抢占导致过度抢占开销
  for env_var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
      os.environ[env_var] = "6"
  ```

- [ ] **Step 4: 重新运行测试以验证硬绑定成功**

  运行命令：
  ```bash
  E:\conda\envs\asr_ui_env\python.exe -m pytest tests/test_affinity.py -v
  ```
  预期输出：PASS

- [ ] **Step 5: 提交代码**

  ```bash
  git add tests/test_affinity.py asr_onnx_service.py
  git commit -m "feat: add psutil physical cpu affinity binding and thread limits"
  ```

---

### Task 2: 跑评时延维度修复与 OpenVINO 时延基准跑通

**Files:**
- Modify: `tests/test_openvino_speed.py`

- [ ] **Step 1: 修改模拟特征输入维度**

  定位到 [tests/test_openvino_speed.py](file:///E:/project/funclip-pro/tests/test_openvino_speed.py) 的第 14 行左右，将原本 `feat_dim = 80` 修改为 `feat_dim = 560`（即拼接因子 7 乘基准维度 80）。
  具体修改后对应的第 13-16 行：
  ```python
  # 模拟数据输入
  batch_size = 1
  time_len = 100
  feat_dim = 560  # 修正：SenseVoiceSmall 拼接因子为 7，维度实际应为 7 * 80 = 560
  ```

- [ ] **Step 2: 运行跑评测试**

  运行命令：
  ```bash
  E:\conda\envs\asr_ui_env\python.exe tests/test_openvino_speed.py
  ```
  预期输出：测试正常运行完毕，并打印出 ORT-CPU vs OpenVINO-CPU 在不同 Batch 下的推理吞吐和性能加速对比，无报错。

- [ ] **Step 3: 提交代码**

  ```bash
  git add tests/test_openvino_speed.py
  git commit -m "fix: adjust mock feature dimension to 560 in openvino speed test"
  ```

---

### Task 3: ASR 模型加载与推理的 OpenVINO 原生 Core 重构

**Files:**
- Modify: `asr_onnx_service.py`
- Test: `tests/test_onnx_decode_refactor.py`

- [ ] **Step 1: 运行现有的 numpy 批量解码单元测试并观察报错**

  运行命令：
  ```bash
  E:\conda\envs\asr_ui_env\python.exe -m pytest tests/test_onnx_decode_refactor.py -v
  ```
  预期输出：由于 `SenseVoiceSmall` 此时还基于 `onnxruntime.InferenceSession` 且我们在 asr_onnx_service 中可能改变了依赖，测试通过但目前并非由 OpenVINO 运行。

- [ ] **Step 2: 重构 SenseVoiceSmall 类以接入 OpenVINO 引擎**

  在 [asr_onnx_service.py](file:///E:/project/funclip-pro/asr_onnx_service.py) 中，对 `SenseVoiceSmall` 进行重写：
  
  1. 引入 `openvino.Core`：
     ```python
     from openvino import Core
     ```
  2. 修改 `__init__` 函数：
     ```python
         def __init__(self, model_dir, batch_size=1, quantize=True, device_id="-1", intra_op_num_threads=6, **kwargs):
             super().__init__(model_dir, batch_size=batch_size, device_id=device_id, quantize=quantize, **kwargs)
             # 加载 tokens.json 以还原文本
             tokens_path = os.path.join(model_dir, "tokens.json")
             with open(tokens_path, "r", encoding="utf-8") as f:
                 self.tokens = json.load(f)
                 
             # 初始化 OpenVINO Core 并读取编译
             model_path = os.path.join(model_dir, "model_quant.onnx" if quantize else "model.onnx")
             self.ie = Core()
             ov_model = self.ie.read_model(model_path)
             
             # 使用指定的 6 线程进行 CPU 专属优化编译
             self.compiled_model = self.ie.compile_model(ov_model, "CPU", config={
                 "INFERENCE_NUM_THREADS": str(intra_op_num_threads),
                 "NUM_STREAMS": "1"
             })
     ```
  3. 重写 `infer` 函数，完全替换掉 ONNX Runtime 接口：
     ```python
         def infer(self, feats, feats_len, language, textnorm):
             # 直接传入 numpy 数组列表
             results = self.compiled_model([feats, feats_len, language, textnorm])
             
             # 映射输出节点
             ctc_logits = results[self.compiled_model.output(0)]
             encoder_out_lens = results[self.compiled_model.output(1)]
             
             return ctc_logits, encoder_out_lens
     ```

- [ ] **Step 3: 重新运行 numpy 解码测试验证转写输出一致性**

  运行命令：
  ```bash
  E:\conda\envs\asr_ui_env\python.exe -m pytest tests/test_onnx_decode_refactor.py -v
  ```
  预期输出：PASS（转写文字与原 PyTorch 结果依旧 100% 一致，且没有 ORT 运行时反量化警告，说明 OpenVINO 接口重构成功且结果正确）。

- [ ] **Step 4: 提交代码**

  ```bash
  git add asr_onnx_service.py
  git commit -m "feat: replace ONNX Runtime with native OpenVINO Core engine"
  ```

---

### Task 4: 接口大 Batch 化标点集成验证测试

**Files:**
- Create: `tests/test_onnx_service_integration.py`

- [ ] **Step 1: 编写微服务集成接口测试**

  创建测试文件 `tests/test_onnx_service_integration.py`：
  ```python
  import os
  import sys
  import time
  import requests
  import subprocess
  import pytest

  def test_openvino_transcribe_api():
      # 启动后台服务进程在 8002 端口上
      # 这里使用当前 python 解释器启动 asr_onnx_service
      env = os.environ.copy()
      env["FORCE_CPU"] = "1"
      
      cmd = [sys.executable, "asr_onnx_service.py"]
      proc = subprocess.Popen(cmd, env=env)
      time.sleep(10)  # 等待服务加载模型并启动完毕
      
      try:
          url = "http://127.0.0.1:8002/transcribe"
          # 使用项目中已有的测试音频文件
          audio_file = r"E:\下载\下载\李雪花2.wav"
          if not os.path.exists(audio_file):
              pytest.skip(f"音频文件 {audio_file} 不存在，跳过该接口测试")
              
          with open(audio_file, "rb") as f:
              files = {"file": f}
              data = {"vad_split": "true"}
              resp = requests.post(url, files=files, data=data)
              
          assert resp.status_code == 200
          res_json = resp.json()
          assert "text" in res_json
          text = res_json["text"]
          print("OpenVINO Integrated API Output:", text[:100])
          assert len(text) > 0
          
          # 校验是否带上了标点符号，证明 CT-Punc 模型正常串联
          has_punc = any(p in text for p in ["，", "。", "？", "！"])
          assert has_punc, "微服务转写文本未成功加上标点"
          
      finally:
          proc.terminate()
          proc.wait()
  ```

- [ ] **Step 2: 运行集成接口测试**

  运行命令：
  ```bash
  E:\conda\envs\asr_ui_env\python.exe -m pytest tests/test_onnx_service_integration.py -v -s
  ```
  预期输出：PASS（打印出带标点符号的识别文本结果）。

- [ ] **Step 3: 提交代码**

  ```bash
  git add tests/test_onnx_service_integration.py
  git commit -m "test: add integration test for OpenVINO ASR FastAPI endpoint"
  ```

---

### Task 5: 最终 A/B 性能测试对比跑评与基准数据收集

**Files:**
- Run: `tests/test_asr_comparison_cpu.py`

- [ ] **Step 1: 运行 A/B 性能跑评脚本以收集最新的 OpenVINO-CPU vs PyTorch-CPU 吞吐数据**

  运行命令：
  ```bash
  E:\conda\envs\asr_ui_env\python.exe tests/test_asr_comparison_cpu.py
  ```
  预期输出：在拉起后台的 PyTorch-CPU 服务与 OpenVINO-CPU 服务后，对比 85 个音频片的推理性能。OpenVINO-CPU 转写耗时应大幅优于 PyTorch-CPU（耗时通常可降为 10~15 秒以内，约 1.5x~2x 的 PyTorch-CPU 加速）。同时文字结果吻合度应在 98% 以上。

- [ ] **Step 2: 整理并记录 A/B 对比结果报告**

  根据输出的性能表格，在 `HANDOFF.md` 或优化报告中记录最新数据。
