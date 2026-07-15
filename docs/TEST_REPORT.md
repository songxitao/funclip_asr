# funclip-pro CER / DER 端到端评测报告

> 日期：2026-07-11 | 被测对象：FastAPI `asr_onnx_service.py`（:8002）  
> 环境：conda `asr_ui_env` (Python 3.11.14)，GPU RTX 4080，CUDA 可用

---

## 1. 评测目标

对 ASR 服务做端到端准确率评测：

| 维度 | 测试集 | 指标 | 规模 |
|---|---|---|---|
| ASR 准确率 | AISHELL-1 | CER（字错率） | 7176 条，冒烟 200 + 采样 1000 |
| 说话人分离 | AISHELL-4 | DER（分离错误率） | 20 场会议，本轮测 1 条 |

---

## 2. CER 结果（ASR 准确率）✅ 有效

| 规模 | 引擎 | CER | 耗时 | 状态 |
|---|---|---|---|---|
| 冒烟 200 条 | Sherpa-CPU | 5.03% | 505s | ✅ |
| 采样 1000 条 (seed=42) | PyTorch-GPU | 6.53% | 395s | ✅ |
| 全量 7176 条 | — | — | — | ⏸ 冻结 |

**结论**：AISHELL-1 近场朗读场景，6.53% CER 可信。模型表现正常。

---

## 3. DER 结果（说话人分离）与改进历程

### 3.1 测试数据

AISHELL-4 测试集 20 场 8 声道会议，本轮测 `L_R003S01C02`（6 说话人，~39min）。

### 3.2 三轮迭代

| 轮次 | 通道策略 | 聚类策略 | DER | CONF | hyp 分布 | 结论 |
|---|---|---|---|---|---|---|
| v1 | ch0 硬编码 | two_stage | 71.82% | 83,749 | `{1:483,2:3,3:2,...}` 98%→1簇 | 崩塌，不可用 |
| v2（P0）| ch3 SNR选优 | two_stage | 74.92% | 87,814 | `{1:497,2:3,...}` 98%→1簇 | 更差，根因不在通道 |
| v3（P1）| ch3 SNR选优 | **spectral** | **32.18%** | **11,399** | `{1:64,2:142,3:183,4:14,5:42,6:61}` | ✅ 在分人了 |

### 3.3 关键发现

- **P0 多通道选优无效**：从 ch0（能量排第 5）换到 ch3（能量排第 1），DER 反而从 71.82% 升到 74.92%。远场单通道下换哪个通道都不解决问题。
- **P1 SpectralClustering 有效**：CONF 从 87,814 降到 11,399（-87%），hyp 分布从「98% 挤一个簇」变为「6 个簇均衡分布」——聚类从崩溃变为真正在工作。
- **MISS 稳定在 ~45K**：三轮测试中 MISS 始终在 44,303~45,737 之间，与聚类策略无关，是 VAD 切分边界与 collar 对齐问题。

---

## 4. 代码改动明细

### 4.1 `speaker_engine.py`

| 改动 | 说明 |
|---|---|
| 新增 import `SpectralClustering` | 来自 sklearn.cluster（已内置，零新依赖）|
| 新增 `strategy="spectral"` 分支 | ~24 行，`SpectralClustering(affinity='nearest_neighbors', n_neighbors=10)` |
| 自动估 K：`max(2, min(20, n//10))` | oracle-K 时用传入值 |
| 安全约束：`n_clusters = min(n_clusters, n-1, 20)` | 防 K>=N 崩溃 |

### 4.2 `der_eval.py`

| 改动 | 说明 |
|---|---|
| `to_mono_ch0()` 改为 SNR 选优 | 新增 `_select_best_channel()`，选取能量最高通道 |
| 新增 `--diarize_strategy` CLI 参数 | 默认 `spectral`，可切 `two_stage` 对比 |

### 4.3 `asr_onnx_service.py`

无需改动：`diarize_strategy` 已透传给 `speaker_engine.cluster(strategy=...)`，spectral 天然命中新增分支。

---

## 5. 当前状态总览

| 项 | 状态 | 值 |
|---|---|---|
| ASR CER | ✅ 有效 | 6.53%（1000 样本，GPU）|
| ASR 全量 CER | ⏸ 冻结 | — |
| DER two_stage | ❌ 不可用 | 71.82% |
| DER spectral | 🟡 勉强可用 | 32.18% |
| DER 全量 20 场 | ⏸ 未跑 | — |

---

## 6. TODO / 下一步

### 6.1 短期（立即可做）

- [ ] **DER 全量 20 场**：跑完 AISHELL-4 全部会议，取平均 DER（当前只有 1 条）
- [ ] **MISS 根因排查**：45K MISS 稳定存在，疑似 VAD 切分与 RTTM collar 不对齐，需单独排查
- [ ] **回退 `to_mono_ch0` 为原始 ch0**：P0 SNR 选优已证实无效且让代码变复杂，建议回退到简洁的 ch0

### 6.2 中期

- [ ] **换测试集**：AISHELL-4 是 8 声道学术场景，与实际部署不匹配。考虑换近场/单声道会议测试集
- [ ] **调整 spectral 参数**：`n_neighbors`、`n_clusters` 自动估 K 策略可调优
- [ ] **CER 全量 7176**：如需最终 CER 结论

### 6.3 长期

- [ ] **多通道 embedding 融合**：虽 P0 选优失败，但多通道 embedding 均值/拼接理论上仍有价值
- [ ] **pyannote 标准 DER**：当前纯 Python DER 与 pyannote.metrics 可能有差异，用于论文对比时需对齐

---

## 7. 可复跑命令

```bash
# 环境
conda activate asr_ui_env

# 启动服务
python asr_onnx_service.py   # :8002

# CER 采样（GPU，1000 条）
python cer_eval_parallel.py \
  --wav_dir testset/aishell1_test_extracted/wav \
  --transcript testset/aishell1_test_extracted/transcript.txt \
  --engine gpu --sample 1000 --workers 4 --seed 42 \
  --out test_results/cer_sample_1k.jsonl

# DER spectral（单条）
python der_eval.py \
  --audio_dir testset/dia-aishell4-test/audio/test \
  --rttm_dir  testset/dia-aishell4-test/rttm/test \
  --limit 1 --timeout 1800 \
  --diarize_strategy spectral \
  --out test_results/der_spectral.json

# DER two_stage（对照）
python der_eval.py \
  --audio_dir testset/dia-aishell4-test/audio/test \
  --rttm_dir  testset/dia-aishell4-test/rttm/test \
  --limit 1 --timeout 1800 \
  --diarize_strategy two_stage \
  --out test_results/der_two_stage.json
```

---

## 8. 产物清单

| 文件 | 说明 |
|---|---|
| `test_results/cer_sample_1k.jsonl` | CER 采样结果（1000 条）|
| `test_results/cer_sample_1k_summary.json` | CER 汇总（6.53%）|
| `test_results/der_single.json` | DER v1（ch0 + two_stage，71.82%）|
| `test_results/der_single_v2.json` | DER v2（ch3 + two_stage，74.92%）|
| `test_results/der_spectral.json` | DER v3（ch3 + spectral，32.18%）|
| `TEST_REPORT.md` | 本报告 |
