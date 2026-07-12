# 基于 VAD 活性段滑窗提纯的段级说话人分离实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现基于 VAD 活性区间内滑窗提纯 Mean Embedding 的离线聚类方案，彻底降低静音段带来的 FA 虚警并解决大段声纹污染与文本错序问题。

**Architecture:** ASR 和 Diarization 的时序切分完全绑定为 VAD 活性段（1:1）。对每个 VAD 段，在内部使用 1.5s 窗、0.5s 步长滑动切片，各窗提取 Cam++ Embedding 并取平均（Mean）得到本段的代表向量，最后对所有段进行全局谱聚类。

**Tech Stack:** Python, FunASR (Cam++), Scikit-Learn (SpectralClustering), Pytest

---

### Task 1: 新增段内滑窗提纯函数 `extract_embedding_sliding_mean`

**Files:**
- Modify: `speaker_engine.py` (新增方法)
- Create: `tests/test_vad_sliding.py` (单元测试)

- [ ] **Step 1: 编写失败的单元测试**
  在 `tests/test_vad_sliding.py` 中编写 `test_extract_embedding_sliding_mean`，使用 `unittest.mock` 模拟 `extract_embedding` 返回固定的一维 numpy 向量，验证该方法是否能对输入的 1D numpy 数组成功滑动切片，并正确计算有效子向量的算术平均值及 L2 归一化。
  
  ```python
  import pytest
  import numpy as np
  from unittest.mock import MagicMock, patch
  from speaker_engine import CampPlusSpeaker

  def test_extract_embedding_sliding_mean():
      # 构造一个 4.5s (16k) 的音频数据，应切出 7 个窗 (1.5s 窗, 0.5s 步长)
      sr = 16000
      audio = np.random.randn(int(4.5 * sr))
      
      # Mock CampPlusSpeaker 初始化和模型
      with patch('funasr.AutoModel') as mock_auto:
          speaker = CampPlusSpeaker(model_dir="mock_dir", device="cpu")
          # 模拟 extract_embedding，让它交替返回向量或 None
          # 这样可以测试 None 过滤
          mock_embs = [
              np.array([1.0, 0.0]),
              None,
              np.array([0.0, 1.0]),
              np.array([1.0, 1.0]),
              None,
              np.array([0.0, 0.0]),
              np.array([-1.0, 0.0])
          ]
          speaker.extract_embedding = MagicMock(side_effect=mock_embs)
          
          # 执行提纯
          emb = speaker.extract_embedding_sliding_mean(audio, sr=sr)
          
          # 校验
          # 有效向量有：[1,0], [0,1], [1,1], [0,0], [-1,0]
          # 算术平均为: [1/5, 2/5] = [0.2, 0.4]
          # L2 归一化后应为: [0.2, 0.4] / sqrt(0.2^2 + 0.4^2) = [1/sqrt(5), 2/sqrt(5)] = [0.4472136, 0.89442719]
          assert emb is not None
          expected = np.array([0.2, 0.4])
          expected = expected / np.linalg.norm(expected)
          np.testing.assert_array_almost_equal(emb, expected, decimal=5)
  ```

- [ ] **Step 2: 运行测试以验证其失败**
  运行：`chcp 65001 >$null; [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; E:\conda\envs\asr_ui_env\python.exe -m pytest tests/test_vad_sliding.py::test_extract_embedding_sliding_mean -s -p no:cacheprovider`
  预期：FAIL 并报错 `ImportError: cannot import name 'CampPlusSpeaker'`（如果因为未定义该方法或模块）或找不到该测试。

- [ ] **Step 3: 实现 `extract_embedding_sliding_mean`**
  在 `speaker_engine.py` 的 `CampPlusSpeaker` 类中实现该方法：
  
  ```python
      def extract_embedding_sliding_mean(self, chunk_16k, sr=16000, win_sec=1.5, step_sec=0.5) -> Optional[np.ndarray]:
          """对单个 VAD 大段在其内部进行滑窗切片，提声纹向量并计算平均值（Mean）。"""
          if chunk_16k is None:
              return None
          # 将 chunk 转为一维 numpy
          if hasattr(chunk_16k, "cpu"):
              arr = chunk_16k.cpu().numpy()
          elif hasattr(chunk_16k, "numpy"):
              arr = chunk_16k.numpy()
          else:
              arr = np.asarray(chunk_16k)
          
          # 获取滑窗
          from speaker_engine import segment_sliding_window
          windows = segment_sliding_window(arr, sr, win_sec, step_sec)
          if not windows:
              # 退化提取整段
              return self.extract_embedding(arr)
          
          embs = []
          for _, _, samp in windows:
              emb = self.extract_embedding(samp)
              if emb is not None:
                  embs.append(emb)
          
          if not embs:
              # 全失败，退化为整段提取
              return self.extract_embedding(arr)
          
          # 算术平均并归一化
          mean_emb = np.mean(embs, axis=0)
          norm = np.linalg.norm(mean_emb)
          if norm > 1e-6:
              mean_emb = mean_emb / norm
          return mean_emb
  ```

- [ ] **Step 4: 运行测试以验证其通过**
  运行：`chcp 65001 >$null; [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; E:\conda\envs\asr_ui_env\python.exe -m pytest tests/test_vad_sliding.py::test_extract_embedding_sliding_mean -s -p no:cacheprovider`
  预期：PASS

- [ ] **Step 5: 提交更改**
  运行：
  `chcp 65001 >$null; [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; git add speaker_engine.py tests/test_vad_sliding.py; git commit -m "feat: 新增 VAD 活性段内部滑窗声纹提纯函数及单测"`

---

### Task 2: 实现 `cluster` 对 `vad_sliding` 策略的支持

**Files:**
- Modify: `speaker_engine.py` (扩展 `cluster` 方法)
- Modify: `tests/test_vad_sliding.py` (新增聚类测试)

- [ ] **Step 1: 编写聚类测试**
  在 `tests/test_vad_sliding.py` 中编写 `test_cluster_vad_sliding`：
  
  ```python
  def test_cluster_vad_sliding():
      with patch('funasr.AutoModel') as mock_auto:
          speaker = CampPlusSpeaker(model_dir="mock_dir", device="cpu")
          # 模拟三个 VAD 段
          chunks = [np.random.randn(32000), np.random.randn(48000), np.random.randn(32000)]
          
          # Mock extract_embedding_sliding_mean，使三个段分别对应不同声纹
          mock_mean_embs = [
              np.array([1.0, 0.0]),
              np.array([0.9, 0.1]),
              np.array([0.0, 1.0])
          ]
          speaker.extract_embedding_sliding_mean = MagicMock(side_effect=mock_mean_embs)
          
          # 执行聚类，策略指定为 vad_sliding，预期分成两组 (spk1, spk1, spk2)
          # 设置 n_speakers=2
          result = speaker.cluster(chunks, strategy="vad_sliding", n_speakers=2)
          
          # 验证返回结构 {seg_idx: speaker_id}
          assert len(result) == 3
          assert result[0] == result[1]
          assert result[0] != result[2]
          assert result[0] in [1, 2]
  ```

- [ ] **Step 2: 运行测试以验证其失败**
  运行：`chcp 65001 >$null; [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; E:\conda\envs\asr_ui_env\python.exe -m pytest tests/test_vad_sliding.py::test_cluster_vad_sliding -s -p no:cacheprovider`
  预期：FAIL 并报错 `ValueError: ValueError` 或聚类不支持 `"vad_sliding"`。

- [ ] **Step 3: 扩展 `cluster` 方法**
  在 `speaker_engine.py` 中的 `cluster` 方法首部和分支处理中，增加对 `strategy == "vad_sliding"` 的支持：
  
  ```python
      def cluster(self, audio_chunks: List, strategy: str = "single",
                  overseg_threshold: float = _OVERSEG_THRESHOLD,
                  merge_threshold: float = _MERGE_THRESHOLD,
                  n_speakers: Optional[int] = None,
                  seg_times: Optional[List[Tuple[int, int]]] = None) -> dict:
          # ... 原代码不变 ...
          # 原先是调用 self._extract_all(audio_chunks)
          # 我们将其重构为根据策略分发提取
          if strategy == "vad_sliding":
              embeddings = []
              valid = []
              for i, chunk in enumerate(audio_chunks):
                  emb = self.extract_embedding_sliding_mean(chunk)
                  if emb is not None:
                      embeddings.append(emb)
                      valid.append(i)
          else:
              embeddings, valid = self._extract_all(audio_chunks)
              
          result = {i: "?" for i in range(len(audio_chunks))}
          
          if len(embeddings) < 2:
              for i in range(len(audio_chunks)):
                  result[i] = 1 if len(embeddings) == 1 else "?"
              return result
  
          emb_matrix = np.vstack(embeddings)
          
          # 在聚类分发中，让 vad_sliding 复用 spectral 聚类逻辑
          if strategy in ["single", "two_stage"]:
              pass # 走原来分支
          
          if strategy == "single":
              labels = self._ahc(emb_matrix, threshold=_DIST_THRESHOLD, n=n_speakers)
          elif strategy in ["spectral", "vad_sliding"]: # 👈 支持 vad_sliding
              n = len(embeddings)
              if n_speakers is not None:
                  n_clusters = n_speakers
              else:
                  n_clusters = max(2, min(20, n // 10))
              n_clusters = min(n_clusters, n - 1, 20)
              if n_clusters < 1:
                  n_clusters = 1
              if n_clusters == 1 or n <= 1:
                  labels = np.zeros(n, dtype=int)
              else:
                  sc = SpectralClustering(
                      n_clusters=n_clusters,
                      affinity='nearest_neighbors',
                      n_neighbors=min(10, n - 1),
                      assign_labels='kmeans',
                      random_state=42,
                  )
                  labels = sc.fit_predict(emb_matrix)
          else:
              # two_stage
              # ... 保持原代码 ...
  ```

- [ ] **Step 4: 运行测试以验证其通过**
  运行：`chcp 65001 >$null; [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; E:\conda\envs\asr_ui_env\python.exe -m pytest tests/test_vad_sliding.py::test_cluster_vad_sliding -s -p no:cacheprovider`
  预期：PASS

- [ ] **Step 5: 提交更改**
  运行：
  `chcp 65001 >$null; [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; git add speaker_engine.py tests/test_vad_sliding.py; git commit -m "feat: cluster 支持 vad_sliding 声纹提纯离线聚类策略"`

---

### Task 3: 在 `asr_onnx_service.py` 中引入 `vad_sliding` 支持与文本合并

**Files:**
- Modify: `asr_onnx_service.py:442-480`
- Modify: `tests/test_sliding_integration.py` (集成测试适配)

- [ ] **Step 1: 编写集成测试**
  在 `tests/test_sliding_integration.py` 中增加对 `vad_sliding` 策略的测试案例，验证当调用包含此参数的 API 时，返回的 segments 拥有对应的识别文字 `text`，且 `diarized_text` 能拼接出正确的说话人转写内容。
  
  ```python
  def test_vad_sliding_service_integration():
      # 此处可以使用 FastAPI TestClient 或者直接 Mock _run_inference 过程
      # 在 test_sliding_integration.py 现有结构中添加对 diarize_strategy="vad_sliding" 的测试
      from fastapi.testclient import TestClient
      from asr_onnx_service import app
      
      client = TestClient(app)
      # 构造模拟请求
      # 由于集成测试在原 test_sliding_integration.py 里 mock 了 FakeSpk
      # 我们同样可以 mock _get_spk_model 并进行端点 POST 校验
  ```

- [ ] **Step 2: 运行测试以验证其失败**
  运行：`chcp 65001 >$null; [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; E:\conda\envs\asr_ui_env\python.exe -m pytest tests/test_sliding_integration.py -s -p no:cacheprovider`
  预期：FAIL (因为 asr_onnx_service 里还没有 `"vad_sliding"` 的逻辑分发)

- [ ] **Step 3: 修改 `asr_onnx_service.py` 合并逻辑**
  在 `asr_onnx_service.py` 的 `_run_inference` 函数中添加 `"vad_sliding"` 的分发。走 `"vad_sliding"` 时，它和旧 `"two_stage"` 逻辑类似，依然利用 `chunks` 做分人，但是 `strategy` 传给 `cluster` 是 `"vad_sliding"`。同时回填 `text`：
  
  ```python
      # asr_onnx_service.py L443 说话人分离逻辑修改：
      segments = []
      diarized_text = ""
      if diarize and chunks:
          try:
              if diarize_strategy == "sliding":
                  # 保持原有盲滑窗分支（备用）
                  merged = _get_spk_model().cluster_sliding(
                      y, sr=16000, strategy="spectral",
                      n_speakers=num_speakers, win_sec=1.5, step_sec=0.5,
                  )
                  for st, en, spk in merged:
                      segments.append({
                          "start": int(st * 1000),
                          "end": int(en * 1000),
                          "speaker": str(spk),
                          "text": "",
                      })
              else:
                  # 包含 vad_sliding, two_stage, spectral, single 等基于 VAD 段聚类的策略
                  spk_cache = _get_spk_model().cluster(
                      chunks, strategy=diarize_strategy, seg_times=seg_meta, n_speakers=num_speakers
                  )
                  for i, (start_ms, end_ms) in enumerate(seg_meta):
                      spk = spk_cache.get(i, "?")
                      seg_text = clean_texts[i] if i < len(clean_texts) else ""
                      segments.append({
                          "start": start_ms,
                          "end": end_ms,
                          "speaker": str(spk),
                          "text": seg_text,  # 👈 正确合并 ASR 文本
                      })
              diarized_text = "\n".join(
                  f"[说话人{seg['speaker']}] {seg['text']}" for seg in segments if seg["text"]
              )
          except Exception as spk_err:
              logger.error(f"说话人分离失败，退回无标注: {spk_err}", exc_info=True)
  ```

- [ ] **Step 4: 运行测试以验证其通过**
  运行：`chcp 65001 >$null; [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; E:\conda\envs\asr_ui_env\python.exe -m pytest tests/test_sliding_integration.py -s -p no:cacheprovider`
  预期：PASS

- [ ] **Step 5: 提交更改**
  运行：
  `chcp 65001 >$null; [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; git add asr_onnx_service.py; git commit -m "feat: API 服务支持 vad_sliding 策略并合并输出文本"`

---

### Task 4: 更新评测脚本 `der_eval.py` 并运行全量验证

**Files:**
- Modify: `der_eval.py` (更新入参 choices)

- [ ] **Step 1: 修改 `der_eval.py` 允许 vad_sliding**
  在 `der_eval.py` 的 argparse 中允许 `vad_sliding` 选项：
  
  ```python
  # der_eval.py L165 附近：
  parser.add_argument('--diarize_strategy', type=str, default='two_stage',
                      choices=['single', 'two_stage', 'spectral', 'vad_sliding'], # 👈 加入 vad_sliding
                      help='diarization strategy')
  ```

- [ ] **Step 2: 运行单场及全量 20 场评测**
  启动 API 服务（如未启动）：
  `chcp 65001 >$null; [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; E:\conda\envs\asr_ui_env\python.exe asr_onnx_service.py`
  
  运行 `vad_sliding` 策略的评测命令（以 R8002_M8002 为例）：
  `chcp 65001 >$null; [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; E:\conda\envs\asr_ui_env\python.exe ali_der_eval.py R8002_M8002 --strategy vad_sliding`
  
  运行全量评测脚本：
  `chcp 65001 >$null; [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; E:\conda\envs\asr_ui_env\python.exe run_ali_der_full.py --strategy vad_sliding`

  预期：评测正常结束，能输出包含 FA/MISS/CONF 在内的完整 DER，并且 FA 指标显回落到极低状态（< 0.5%）。

- [ ] **Step 3: 提交更改**
  运行：
  `chcp 65001 >$null; [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; git add der_eval.py; git commit -m "feat: 评测选项添加 vad_sliding 并在 AliMeeting 测试集完成全量验证"`
