# AliMeeting 近场说话人分离 + 滑窗 Segmentation 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把说话人分离的 segmentation 从「VAD 段」改成「滑窗」，在 AliMeeting 近场测试集上跑全量 20 场 DER，验证 CONF 是否下降。

**Architecture:** speaker_engine 新增 `cluster_sliding()` 方法——整段音频内部 1.5s 滑窗切分，逐窗提 Cam++ embedding，复用现有 spectral 聚类，合并相邻同人窗还原说话人段。Cam++ 模型和聚类逻辑不动，ASR 的 VAD 不动，只换 diarization 的"怎么切段"。asr_onnx_service 的 diarize 分支接滑窗。

**Tech Stack:** Python 3.11（conda asr_ui_env）/ funasr Cam++ / sklearn SpectralClustering / fastapi / numpy / soundfile

---

## 0. 接手必读上下文（你没参与讨论，先看这个）

### 0.1 项目背景
- 项目：`E:\project\funclip-pro`，FastAPI ASR + 说话人分离服务（端口 8002）。
- 说话人分离现状：`VAD 切段 → 每段提 1 个 Cam++ embedding → spectral 聚类 → 贴 speaker`。
- 问题：VAD 为 ASR 设计，段大（~8s），段内多人交替 → embedding 是混合向量 → 聚类 CONF 高。
- 试跑数据：AliMeeting near R8002_M8002，DER=49.21%，MISS=0.7%（近场几乎无 MISS），**CONF=48.3% 是主因**。

### 0.2 滑窗方案核心（必懂）
- **窗为最小单位**（1.5s 窗 + 0.5s 步长），VAD 段在 diarization 里**退场**（仅框有语音区域避静音）。
- 每窗独立提向量 → 聚类 → 每窗贴标签 → 合并相邻同人窗。
- Cam++ 和聚类**完全不动**，只换"怎么切段喂给它"。
- 跨界窗（A/B 边界混合窗）是少数，聚类是全局找分堆，少数淹没不了多数。
- 局限：快速抢话/叠词（切换 < 1.5s）会失效。

### 0.3 硬约束（违反会出事）
- **Python 环境**：唯一可用是 `E:\conda\envs\asr_ui_env\python.exe`（Python 3.11.14，funasr/sherpa_onnx/torch 齐全）。系统 Python 缺依赖会崩。
- **读代码**：理解现有结构用 `codegraph node <file>`（已装，1.2.0）。改代码时可以读相关函数。不要盲改。
- **GPU**：RTX 4080，CUDA 可用，Cam++/VAD 已迁 CUDA 带 CPU 回退。
- **模型路径**：硬编码在 asr_onnx_service.py（`E:\project\funclip-pro\model\...`），别动。
- **测试**：单测用 conda env python 跑 pytest。集成测试起 :8002 服务。
- **不擅自装依赖**：遇 ModuleNotFoundError 先问用户（AGENTS.md 规则）。

### 0.4 已有产物（直接用）
| 产物 | 路径 | 状态 |
|---|---|---|
| Design spec | `docs/superpowers/specs/2026-07-12-alimeeting-diarization-sliding-window-design.md` | ✅ 已提交 |
| 近场预处理脚本 | `ali_near_prep.py` | ✅ 已写，未提交（Task 1 提交） |
| DER 评测脚本 | `ali_der_eval.py` | ✅ 已写，未提交（Task 1 提交） |
| 数据集 | `testset/Test_Ali/Test_Ali/{Test_Ali_far,Test_Ali_near}/` | ✅ 已下（6GB，已 gitignore） |
| 试跑产物 | `testset/ali_near_prep/R8002_M8002_mixed.wav` + `.rttm` | ✅ 已生成 |

### 0.5 现有代码关键符号（codegraph 已确认）
`speaker_engine.py` 的 `CampPlusSpeaker`：
- `extract_embedding(chunk_16k) -> np.ndarray | None`（单段提向量，输入 16k numpy）
- `_extract_all(audio_chunks) -> list`（批量）
- `cluster(audio_chunks, strategy, n_speakers, seg_times) -> dict`（strategy=single/spectral/two_stage）
- spectral 分支：`SpectralClustering(affinity='nearest_neighbors', n_neighbors=min(10,n-1))`

`asr_onnx_service.py`：
- `_get_spk_model()` 惰性加载 Cam++
- `/transcribe` 端点接收 `diarize`、`num_speakers`、`diarize_strategy` 参数
- diarize 时调 `spk.cluster(audio_chunks, strategy=..., n_speakers=...)`

---

## 文件结构

| 文件 | 责任 | 操作 |
|---|---|---|
| `ali_near_prep.py` | 近场混音 + TextGrid→RTTM | 提交（已有）|
| `ali_der_eval.py` | 调服务 + 算单场 DER | 提交（已有）|
| `speaker_engine.py` | 新增 `segment_sliding_window` + `cluster_sliding` | 修改 |
| `asr_onnx_service.py` | diarize 分支接滑窗 | 修改 |
| `tests/test_sliding_segmentation.py` | 滑窗单测 | 新建 |
| `tests/test_sliding_integration.py` | 滑窗 + 服务集成测试 | 新建 |
| `run_ali_der_full.py` | 全量 20 场评测 | 新建 |

---

## Task 1: 提交已有试跑脚本

**Files:**
- 已存在：`ali_near_prep.py`、`ali_der_eval.py`

- [ ] **Step 1: 确认脚本在且能跑**

Run: `E:/conda/envs/asr_ui_env/python.exe ali_near_prep.py R8002_M8002`
Expected: 输出 "验证通过。产物在 ...testset/ali_near_prep"

- [ ] **Step 2: 提交**

```bash
git add ali_near_prep.py ali_der_eval.py
git commit -m "test: AliMeeting 近场预处理与 DER 评测脚本（试跑验证通过）"
```

---

## Task 2: 滑窗切分函数 + 单测（TDD）

**Files:**
- Create: `tests/test_sliding_segmentation.py`
- Modify: `speaker_engine.py`（新增模块级函数 `segment_sliding_window`）

- [ ] **Step 1: 写失败测试**

Create `tests/test_sliding_segmentation.py`:
```python
import numpy as np
from speaker_engine import segment_sliding_window


def test_sliding_window_basic():
    audio = np.zeros(16000 * 3, dtype=np.float32)  # 3s 静音
    wins = segment_sliding_window(audio, 16000, win_sec=1.5, step_sec=0.5)
    # 3s: 起始 0, 0.5, 1.0, 1.5 → 4 个完整窗（1.5-3.0 是最后一个完整窗）
    assert len(wins) == 4
    assert abs(wins[0][0] - 0.0) < 0.01
    assert abs(wins[0][1] - 1.5) < 0.01
    assert abs(wins[-1][1] - 3.0) < 0.01


def test_sliding_window_samples_length():
    audio = np.zeros(16000 * 2, dtype=np.float32)
    wins = segment_sliding_window(audio, 16000, win_sec=1.5, step_sec=0.5)
    for st, en, samp in wins:
        assert len(samp) == 16000 * 1.5  # 每窗 1.5s = 24000 采样


def test_sliding_window_tail_partial():
    # 2.2s 音频：完整窗 0-1.5/0.5-2.0，尾部 1.5-2.2 不足一窗但应保留
    audio = np.zeros(int(16000 * 2.2), dtype=np.float32)
    wins = segment_sliding_window(audio, 16000, win_sec=1.5, step_sec=0.5)
    assert wins[-1][1] > 2.0  # 尾部包含到 2.2
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `E:/conda/envs/asr_ui_env/python.exe -m pytest tests/test_sliding_segmentation.py -v`
Expected: FAIL（`ImportError: cannot import name 'segment_sliding_window'`）

- [ ] **Step 3: 在 speaker_engine.py 顶部（class 外）加函数**

先用 `codegraph node speaker_engine.py` 看现有 import 和结构，然后在 import 区下方、class CampPlusSpeaker 上方加：
```python
def segment_sliding_window(audio, sr, win_sec=1.5, step_sec=0.5):
    """整段音频 -> 滑窗列表 [(start_sec, end_sec, samples_np)]。

    Args:
        audio: 1D numpy 数组，整段音频
        sr: 采样率
        win_sec: 窗长（秒），默认 1.5
        step_sec: 步长（秒），默认 0.5（重叠 1.0s）
    Returns:
        list of (start_sec, end_sec, samples)，每个 samples 是 win_sec*sr 长
    """
    win = int(win_sec * sr)
    step = int(step_sec * sr)
    n = len(audio)
    windows = []
    start = 0
    while start + win <= n:
        end = start + win
        windows.append((start / sr, end / sr, audio[start:end]))
        start += step
    # 尾部不足一窗但有声学内容，保留（提向量时 Cam++ 自行处理短段）
    if start < n and n - start >= sr * 0.5:  # 至少 0.5s 才留
        windows.append((start / sr, n / sr, audio[start:]))
    return windows
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `E:/conda/envs/asr_ui_env/python.exe -m pytest tests/test_sliding_segmentation.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add speaker_engine.py tests/test_sliding_segmentation.py
git commit -m "feat: 滑窗 segmentation 切分函数 segment_sliding_window + 单测"
```

---

## Task 3: cluster_sliding 方法 + 单测（TDD）

**Files:**
- Modify: `speaker_engine.py`（CampPlusSpeaker 新增 `cluster_sliding` 方法）
- Modify: `tests/test_sliding_segmentation.py`（加 cluster_sliding 测试）

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_sliding_segmentation.py`：
```python
from unittest.mock import patch
from speaker_engine import CampPlusSpeaker


def test_cluster_sliding_merges_same_speaker():
    """mock extract_embedding，让前 3 窗返回向量A、后 3 窗返回向量B，
    验证合并后得到 2 段（A 段 + B 段）。"""
    spk = CampPlusSpeaker.__new__(CampPlusSpeaker)  # 不加载模型
    # 构造 6 个窗的 embedding：前3个是A，后3个是B
    emb_a = np.array([1.0, 0.0, 0.0])
    emb_b = np.array([0.0, 1.0, 0.0])
    embeddings = [emb_a]*3 + [emb_b]*3
    with patch.object(spk, 'extract_embedding', side_effect=embeddings):
        audio = np.zeros(16000 * 6, dtype=np.float32)  # 6s -> 多个窗
        merged = spk.cluster_sliding(audio, sr=16000, n_speakers=2,
                                     win_sec=1.5, step_sec=0.5)
    # 应合并成 2 段（A 连续 + B 连续）
    assert len(merged) == 2
    assert merged[0][2] != merged[1][2]  # 两段不同人


def test_cluster_sliding_single_speaker():
    """所有窗同一人，合并成 1 段。"""
    spk = CampPlusSpeaker.__new__(CampPlusSpeaker)
    emb = np.array([1.0, 0.0])
    with patch.object(spk, 'extract_embedding', side_effect=[emb]*5):
        audio = np.zeros(16000 * 5, dtype=np.float32)
        merged = spk.cluster_sliding(audio, sr=16000, n_speakers=1,
                                     win_sec=1.5, step_sec=0.5)
    assert len(merged) == 1


def test_cluster_sliding_none_embedding_filled():
    """某窗 extract_embedding 返回 None，应用前后窗标签填充，不崩。"""
    spk = CampPlusSpeaker.__new__(CampPlusSpeaker)
    emb = np.array([1.0, 0.0])
    embeddings = [emb, None, emb, emb]  # 第2窗失败
    with patch.object(spk, 'extract_embedding', side_effect=embeddings):
        audio = np.zeros(16000 * 4, dtype=np.float32)
        merged = spk.cluster_sliding(audio, sr=16000, n_speakers=1,
                                     win_sec=1.5, step_sec=0.5)
    assert len(merged) >= 1  # 不崩即可
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `E:/conda/envs/asr_ui_env/python.exe -m pytest tests/test_sliding_segmentation.py -v`
Expected: FAIL（`AttributeError: 'CampPlusSpeaker' object has no attribute 'cluster_sliding'`）

- [ ] **Step 3: 在 CampPlusSpeaker 类内加 cluster_sliding 方法**

先用 `codegraph node speaker_engine.py` 看 cluster() 方法的位置和现有 spectral 分支写法，然后在 cluster() 方法后加：
```python
    def cluster_sliding(self, audio_16k, sr=16000, strategy="spectral",
                        n_speakers=None, win_sec=1.5, step_sec=0.5):
        """整段音频滑窗 segmentation + 聚类 + 合并相邻同人窗。

        与 cluster() 的区别：不用外部传入的 VAD 段，内部按固定窗滑切。
        Cam++ 提向量和 spectral 聚类逻辑复用 cluster() 的实现。

        Args:
            audio_16k: 1D numpy，整段 16k 音频
            sr: 采样率（默认 16000）
            strategy: 聚类策略（默认 spectral，复用现有）
            n_speakers: oracle-K；None 则自动估
            win_sec: 窗长秒（默认 1.5）
            step_sec: 步长秒（默认 0.5）
        Returns:
            list of (start_sec, end_sec, speaker_id)，合并后的说话人段
        """
        windows = segment_sliding_window(audio_16k, sr, win_sec, step_sec)
        if not windows:
            return []
        # 逐窗提 embedding
        embeddings = []
        valid_idx = []
        for i, (st, en, samp) in enumerate(windows):
            emb = self.extract_embedding(samp)
            if emb is not None:
                embeddings.append(emb)
                valid_idx.append(i)
        if not embeddings:
            # 全失败，整段标 1
            return [(windows[0][0], windows[-1][1], 1)]
        emb_matrix = np.vstack(embeddings)
        # 聚类（复用 spectral 逻辑）
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
        # 每窗贴标签（speaker_id 从 1 起）
        win_labels = [None] * len(windows)
        for i, lab in zip(valid_idx, labels):
            win_labels[i] = int(lab) + 1
        # 无效窗用前一个有效标签填充（首窗无效用 1）
        last_valid = 1
        for i in range(len(win_labels)):
            if win_labels[i] is None:
                win_labels[i] = last_valid
            else:
                last_valid = win_labels[i]
        # 合并相邻同人窗
        merged = []
        cur_spk = win_labels[0]
        cur_start = windows[0][0]
        cur_end = windows[0][1]
        for i in range(1, len(windows)):
            st, en, _ = windows[i]
            if win_labels[i] == cur_spk:
                cur_end = en
            else:
                merged.append((cur_start, cur_end, cur_spk))
                cur_spk = win_labels[i]
                cur_start = st
                cur_end = en
        merged.append((cur_start, cur_end, cur_spk))
        return merged
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `E:/conda/envs/asr_ui_env/python.exe -m pytest tests/test_sliding_segmentation.py -v`
Expected: 6 passed（Task2 的 3 个 + Task3 的 3 个）

- [ ] **Step 5: 提交**

```bash
git add speaker_engine.py tests/test_sliding_segmentation.py
git commit -m "feat: CampPlusSpeaker.cluster_sliding 滑窗聚类+合并方法 + 单测"
```

---

## Task 4: asr_onnx_service diarize 分支接滑窗

**Files:**
- Modify: `asr_onnx_service.py`（diarize 处理逻辑，新增 `diarize_strategy=sliding` 分支）
- Create: `tests/test_sliding_integration.py`

- [ ] **Step 1: 用 codegraph 定位现有 diarize 调用**

Run: `codegraph node asr_onnx_service.py | grep -nE "diarize|cluster\(|segments|spk_model"`
找到当前 `spk_model.cluster(audio_chunks, strategy=..., n_speakers=...)` 的调用位置，理解 segments 怎么构造。

- [ ] **Step 2: 写集成测试（mock 服务，验证 sliding 分支走通）**

Create `tests/test_sliding_integration.py`:
```python
"""集成测试：diarize_strategy=sliding 时，服务返回带 speaker 的 segments。
用 monkeypatch mock Cam++ 和 ASR，不加载真实模型。"""
import numpy as np
from unittest.mock import patch


def test_transcribe_sliding_diarize_returns_speaker_segments():
    from fastapi.testclient import TestClient
    import asr_onnx_service as svc

    # mock _get_spk_model 返回假 speaker，cluster_sliding 返回固定段
    class FakeSpk:
        def cluster_sliding(self, audio, sr=16000, **kw):
            return [(0.0, 1.5, 1), (1.5, 3.0, 2)]
    with patch.object(svc, '_get_spk_model', return_value=FakeSpk()), \
         patch.object(svc, '_decode', return_value=["测试文本"]):
        # 还需 mock VAD/model 加载——视实际依赖补 mock
        client = TestClient(svc.app)
        # 造 3s 静音 wav 临时文件
        import tempfile, wave, array
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(array.array("h", [0]*48000).tobytes())
        resp = client.post("/transcribe",
            files={"file": ("t.wav", open(tmp.name,"rb"), "audio/wav")},
            data={"diarize": "true", "diarize_strategy": "sliding",
                  "num_speakers": "2", "vad_strategy": "never"})
    assert resp.status_code == 200
    segs = resp.json().get("segments", [])
    assert len(segs) >= 1
    assert all("speaker" in s for s in segs)
```

- [ ] **Step 3: 运行测试，确认失败**

Run: `E:/conda/envs/asr_ui_env/python.exe -m pytest tests/test_sliding_integration.py -v`
Expected: FAIL（`diarize_strategy=sliding` 不被识别，或返回不含 speaker）

- [ ] **Step 4: 改 asr_onnx_service.py 接 sliding 分支**

在 diarize 处理处，原 `spk_model.cluster(audio_chunks, strategy=diarize_strategy, ...)` 旁边加分支：
```python
# 伪代码——实际行号用 codegraph 确认后填
if diarize_strategy == "sliding":
    # 滑窗模式：整段音频喂 cluster_sliding，不走 VAD 段
    merged = spk_model.cluster_sliding(
        audio_16k_np, sr=16000,
        strategy="spectral", n_speakers=num_speakers,
        win_sec=1.5, step_sec=0.5)
    segments = [{"start": int(st*1000), "end": int(en*1000),
                 "speaker": str(spk), "text": ""} for st, en, spk in merged]
else:
    # 原 VAD 段路径（single/spectral/two_stage）保持不变
    ...
```
注意：`audio_16k_np` 是整段 16k 单声道 numpy，需在 diarize 前准备好（读上传文件 → librosa.load(sr=16000)）。具体变量名用 codegraph 看现有代码定。

- [ ] **Step 5: 运行测试，确认通过**

Run: `E:/conda/envs/asr_ui_env/python.exe -m pytest tests/test_sliding_integration.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add asr_onnx_service.py tests/test_sliding_integration.py
git commit -m "feat: /transcribe diarize_strategy=sliding 滑窗说话人分离分支 + 集成测试"
```

---

## Task 5: 全量 20 场评测脚本

**Files:**
- Create: `run_ali_der_full.py`

- [ ] **Step 1: 写全量评测脚本**

Create `run_ali_der_full.py`:
```python
# -*- coding: utf-8 -*-
"""AliMeeting near 全量 20 场 DER 评测。
流程：每场混音+RTTM(ali_near_prep) -> POST :8002?diarize_strategy=sliding -> DER -> 加权平均。
用法: E:/conda/envs/asr_ui_env/python.exe run_ali_der_full.py [--strategy sliding|spectral|two_stage]
前置：1) :8002 服务在跑  2) testset/Test_Ali 已下
"""
import sys, glob, json
from pathlib import Path
from ali_near_prep import BASE as NEAR_BASE, OUT as PREP_OUT, mix_to_mono, write_wav_mono, build_rttm, spk_id_from_name
from ali_der_eval import eval_one
from der_eval import compute_der, parse_rttm
import requests, soundfile as sf, time

STRATEGY = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--strategy" else "sliding"


def list_sessions():
    wav_dir = NEAR_BASE / "audio_dir"
    sessions = sorted(set(p.name.split("_N_")[0] for p in wav_dir.glob("*_N_SPK*.wav")))
    return sessions


def prep_session(session):
    """混音 + RTTM（复用 ali_near_prep 逻辑），返回 (mixed_wav, rttm_path)。"""
    wavs = sorted((NEAR_BASE / "audio_dir").glob(f"{session}_N_SPK*.wav"))
    tgs = sorted((NEAR_BASE / "textgrid_dir").glob(f"{session}_N_SPK*.TextGrid"))
    if len(wavs) < 2:
        return None, None
    mixed, sr = mix_to_mono(wavs)
    mix_wav = PREP_OUT / f"{session}_mixed.wav"
    write_wav_mono(mix_wav, mixed, sr)
    rttm_path = PREP_OUT / f"{session}.rttm"
    build_rttm(session, tgs, rttm_path)
    return mix_wav, rttm_path


def eval_full():
    sessions = list_sessions()
    print(f"共 {len(sessions)} 场 | strategy={STRATEGY}")
    results = []
    gFA = gMISS = gCONF = gREF = 0
    for i, sess in enumerate(sessions, 1):
        print(f"\n[{i}/{len(sessions)}] {sess}")
        mix_wav, rttm = prep_session(sess)
        if not mix_wav:
            print("  跳过（不足2路）"); continue
        # 调服务（复用 ali_der_eval 的 POST 逻辑，但 strategy 用本脚本参数）
        der, detail = eval_one(sess)  # 注意：eval_one 内部 strategy 写死 spectral，需改为读全局或传参
        results.append({"session": sess, "DER": der, "detail": detail})
        gFA += detail["FA"]; gMISS += detail["MISS"]
        gCONF += detail["CONF"]; gREF += detail["REF"]
        print(f"  DER={der*100:.2f}%")
    gder = (gFA + gMISS + gCONF) / gREF if gREF > 0 else 0
    print(f"\n===== 全量 {len(results)} 场 加权平均 =====")
    print(f"global DER = {gder*100:.2f}%")
    print(f"FA={gFA} MISS={gMISS} CONF={gCONF} REF={gREF}")
    out = {"strategy": STRATEGY, "global_DER": gder,
           "global": {"FA": gFA, "MISS": gMISS, "CONF": gCONF, "REF": gREF},
           "per_session": results}
    Path("test_results/ali_der_full.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已写入 test_results/ali_der_full.json")


if __name__ == "__main__":
    eval_full()
```

**注意：** `ali_der_eval.py` 的 `eval_one()` 内 `diarize_strategy` 写死 `spectral`，需先改成接受参数（或在 `eval_one` 加 `strategy` 参数透传）。这是 Task 5 的前置小改。

- [ ] **Step 2: 改 ali_der_eval.py 让 strategy 可传参**

把 `eval_one(session)` 改成 `eval_one(session, strategy="spectral")`，POST 的 `data` 里 `diarize_strategy` 用参数。

- [ ] **Step 3: 提交**

```bash
git add run_ali_der_full.py ali_der_eval.py
git commit -m "test: AliMeeting near 全量20场DER评测脚本 + eval_one strategy参数化"
```

---

## Task 6: 跑全量评测 + 对比报告

**Files:**
- Output: `test_results/ali_der_full.json`、`docs/superpowers/specs/2026-07-12-ali-der-result-report.md`

- [ ] **Step 1: 确认 :8002 服务在跑（含 sliding 改动）**

如果服务是旧版，杀掉重启：
```bash
# 找占用进程
netstat -ano | grep :8002
# taskkill /PID <pid> /F  （Windows）
# 重启
E:/conda/envs/asr_ui_env/python.exe asr_onnx_service.py &
```

- [ ] **Step 2: 先跑 1 场验证 sliding 分支**

Run: `E:/conda/envs/asr_ui_env/python.exe ali_der_eval.py R8002_M8002`
Expected: 返回 DER，且 CONF 应**明显低于**改造前的 48.3%（若仍 ~48%，说明 sliding 没生效或问题不在粒度，需排查）

- [ ] **Step 3: 跑全量 20 场（sliding）**

Run: `E:/conda/envs/asr_ui_env/python.exe run_ali_der_full.py --strategy sliding`
Expected: 输出 global DER + 写入 `test_results/ali_der_full.json`，耗时约 1-2 小时

- [ ] **Step 4: 跑全量 20 场（spectral/旧VAD段，对照组）**

Run: `E:/conda/envs/asr_ui_env/python.exe run_ali_der_full.py --strategy spectral`
Expected: 对照组 DER，写入 `test_results/ali_der_full_spectral.json`

- [ ] **Step 5: 出对比报告**

Create `docs/superpowers/specs/2026-07-12-ali-der-result-report.md`，含：
- sliding vs spectral(VAD段) 全量 DER 对比
- FA/MISS/CONF 分占比对比
- 结论：滑窗是否降了 CONF
- 若 sliding 明显更优，说明"VAD 段污染"假设成立

- [ ] **Step 6: 提交结果**

```bash
git add test_results/ali_der_full.json test_results/ali_der_full_spectral.json \
        docs/superpowers/specs/2026-07-12-ali-der-result-report.md
git commit -m "test: AliMeeting near 全量20场DER结果 + sliding vs spectral对比报告"
```

---

## Self-Review（计划自审，接手模型不必读）

1. **Spec 覆盖**：spec 的滑窗方案 → Task 2-4；评测设计 → Task 5-6；改造范围 → Task 2-4。✅
2. **占位符**：Task 4 Step 4 有"伪代码——实际行号用 codegraph 确认"，因不读源码不能给精确行号，但给了明确方法（codegraph 定位）和代码骨架。可接受。
3. **类型一致**：`segment_sliding_window` 返回 `[(start, end, samples)]`，`cluster_sliding` 消费它返回 `[(start, end, speaker_id)]`，一致。`speaker_id` 从 1 起，与现有 cluster() 一致。
4. **风险提示**：Task 6 Step 2 是关键验证点——若 sliding 没降 CONF，假设不成立，需回头查。已标注。

---

## 执行交接

计划已存 `docs/superpowers/plans/2026-07-12-alimeeting-diarization-sliding-window.md`。

两种执行方式：
1. **Subagent-Driven（推荐）**：每个 Task 派一个新 subagent，任务间 review，迭代快
2. **Inline Execution**：在本会话内逐 Task 执行，批量 + 检查点

尖子要交给其他模型接手，建议用 Subagent-Driven：把这份计划 + design spec 一起给接手模型，让它按 Task 1-6 顺序执行。
