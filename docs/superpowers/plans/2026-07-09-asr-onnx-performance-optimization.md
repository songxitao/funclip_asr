# ASR ONNX Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 优化 funclip-pro 的 ASR ONNX 推理性能，通过图融合优化、线程绑定和并发特征提取，显著降低 CPU 上的推理延迟，确保文字吻合度无损（CER = 0%）。

**Architecture:** 
- 在 `model_bin.py` 中修改 `SenseVoiceSmallONNX.__init__`，注入高级 ONNX Runtime SessionOptions（限制 6 物理线程、图优化、NHWC 布局、内存复用等）。
- 将 `model_bin.py` 中的 `extract_feat` 特征提取函数改写为使用 `ThreadPoolExecutor(max_workers=4)` 并发处理，从而加速 CPU Pipeline。
- 使用 `tests/test_onnx_performance.py` 进行端到端基准跑评与等价性验证。

**Tech Stack:** Python 3.10, ONNX Runtime, NumPy, concurrent.futures, PyTest

---

### Task 1: Baseline 性能跑评与验证

**Files:**
- Test: `tests/test_onnx_performance.py`

- [ ] **Step 1: 运行现有的性能测试脚本**

Run: `E:\conda\envs\asr_ui_env\python.exe tests/test_onnx_performance.py`
Expected: 脚本运行完成，不发生报错。输出当前 NumPy 向量化解码后的 CPU Pipeline 热启动耗时、文字转写片段。记录该 baseline 性能指标。

---

### Task 2: ORT 运行图高级优化与多线程绑定

**Files:**
- Modify: `model/models/iic/SenseVoiceSmall/utils/model_bin.py`

- [ ] **Step 1: 注入高级 SessionOptions 并限制线程**

在 `model_bin.py` 导入 `onnxruntime as ort`，并在 `SenseVoiceSmallONNX.__init__` 中创建 `self.ort_infer` 后，使用高级 options 重新初始化 InferenceSession。

```python
        # 在 model_bin.py 中修改 SenseVoiceSmallONNX.__init__：
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.add_session_config_entry("session.enable_layout_nhwc", "1")
        opts.intra_op_num_threads = 6
        opts.inter_op_num_threads = 1
        opts.enable_mem_pattern = True
        opts.enable_mem_reuse = True
        
        # 重新创建 session 并覆盖原本的
        providers = self.ort_infer.session.get_providers()
        self.ort_infer.session = ort.InferenceSession(
            model_file, sess_options=opts, providers=providers
        )
```

- [ ] **Step 2: 编写单元测试验证 SessionOptions 是否注入成功**

编写测试脚本 `tests/test_session_options.py` 验证配置是否生效：

```python
import os
import sys
sys.path.append(r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmall")
from utils.model_bin import SenseVoiceSmallONNX
import onnxruntime as ort

def test_session_options():
    model_dir = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmall-ONNX"
    model = SenseVoiceSmallONNX(model_dir, batch_size=1, quantize=True, device_id="-1", intra_op_num_threads=6)
    session = model.ort_infer.session
    options = session.get_session_options()
    
    assert options.intra_op_num_threads == 6
    assert options.inter_op_num_threads == 1
    assert options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    print("Session Options 注入成功！")

if __name__ == "__main__":
    test_session_options()
```

- [ ] **Step 3: 运行测试验证**

Run: `E:\conda\envs\asr_ui_env\python.exe -m pytest tests/test_session_options.py`
Expected: Test passes.

- [ ] **Step 4: 提交代码**

```bash
git add model/models/iic/SenseVoiceSmall/utils/model_bin.py tests/test_session_options.py
git commit -m "perf: inject advanced SessionOptions to ONNX InferenceSession"
```

---

### Task 3: 特征提取 `extract_feat` 并发化

**Files:**
- Modify: `model/models/iic/SenseVoiceSmall/utils/model_bin.py`

- [ ] **Step 1: 特征提取并发化重构**

在 `model_bin.py` 中引入 `concurrent.futures` 并重构 `extract_feat` 函数：

```python
    def extract_feat(self, waveform_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        import concurrent.futures
        feats, feats_len = [], []
        
        def _extract_single(waveform):
            speech, _ = self.frontend.fbank(waveform)
            feat, feat_len = self.frontend.lfr_cmvn(speech)
            return feat, feat_len

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(_extract_single, waveform_list))

        for feat, feat_len in results:
            feats.append(feat)
            feats_len.append(feat_len)

        feats = self.pad_feats(feats, np.max(feats_len))
        feats_len = np.array(feats_len).astype(np.int32)
        return feats, feats_len
```

- [ ] **Step 2: 编写单元测试验证特征提取等价性**

在 `tests/test_extract_feat.py` 中编写等价性测试，比较串行与并发特征提取的输出是否完全一致：

```python
import os
import sys
import numpy as np
sys.path.append(r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmall")
from utils.model_bin import SenseVoiceSmallONNX

def test_extract_feat_equivalence():
    model_dir = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmall-ONNX"
    model = SenseVoiceSmallONNX(model_dir, batch_size=4, quantize=True, device_id="-1")
    
    # 模拟几个不同长度的音频信号
    waveforms = [np.random.randn(16000 * i) for i in range(1, 4)]
    
    # 获取并发模式下的特征提取结果
    feats, feats_len = model.extract_feat(waveforms)
    
    # 临时改回串行方法验证
    original_feats = []
    original_lens = []
    for waveform in waveforms:
        speech, _ = model.frontend.fbank(waveform)
        feat, feat_len = model.frontend.lfr_cmvn(speech)
        original_feats.append(feat)
        original_lens.append(feat_len)
    original_feats = model.pad_feats(original_feats, np.max(original_lens))
    original_lens = np.array(original_lens).astype(np.int32)
    
    # 比较
    assert np.allclose(feats, original_feats, atol=1e-5)
    assert np.array_equal(feats_len, original_lens)
    print("并发特征提取等价性验证成功！")

if __name__ == "__main__":
    test_extract_feat_equivalence()
```

- [ ] **Step 3: 运行测试**

Run: `E:\conda\envs\asr_ui_env\python.exe -m pytest tests/test_extract_feat.py`
Expected: Test passes.

- [ ] **Step 4: 提交代码**

```bash
git add model/models/iic/SenseVoiceSmall/utils/model_bin.py tests/test_extract_feat.py
git commit -m "perf: parallelize extract_feat using ThreadPoolExecutor"
```

---

### Task 4: 端到端跑评验证与 A/B 对照

**Files:**
- Test: `tests/test_onnx_performance.py`

- [ ] **Step 1: 重新运行跑评测试脚本**

Run: `E:\conda\envs\asr_ui_env\python.exe tests/test_onnx_performance.py`
Expected: 运行无报错，记录优化后的各项耗时指标。

- [ ] **Step 2: 对比优化前后结果，验证 CER**

计算文字吻合度（CER）是否为 0%（即与 Task 1 Baseline 输出文字完全对齐）。

- [ ] **Step 3: 汇总 A/B 对照性能报告**

编写并保存 `README_PERF_OPTIMIZED.md`，汇总优化前后的总耗时、RTF 和 Latency 降幅，提交报告。
