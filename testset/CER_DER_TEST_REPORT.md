# funclip-pro CER + DER 端到端评测 · 工作总结与待决问题

> 本文档由测试执行 agent 整理，记录本轮 CER（ASR 字错率）/ DER（说话人分离错误率）端到端评测的**任务清单、所有代码更改、测试结果，以及 DER 失真的根因分析**。
> 目的：把现状与未决问题清晰交接给后续负责修复 DER 的高级模型 / 工程师。

---

## 1. 评测目标与范围

- **被测对象**：FastAPI 服务 `asr_onnx_service.py`（默认 `:8002`）的 ASR 与说话人模型。
- **环境**：conda `asr_ui_env`（Python 3.11.14），GPU = RTX 4080（torch 2.3.1+cu121，CUDA 可用）。
- **范围（与用户确认）**：
  - 不跑全量 pytest；只测 `:8002` 服务的 ASR(CER) + 说话人(DER)。
  - CER：AISHELL-1 测试集，先冒烟再采样，**全量 7176 条用户明确要求暂不做**。
  - DER：AISHELL-4 会议集，本轮只跑 1 条会议（全量 20 条待二次确认）。
  - 用户授权：缺依赖直接装进虚拟环境，不必先确认。

### 测试集（位于 `testset/`，**不纳入 git**）
| 集合 | 内容 | 用途 |
|---|---|---|
| `aishell1_test_extracted/` | 7176 条 wav + `transcript.txt` | CER（ASR 准确率） |
| `dia-aishell4-test/` | 20 个 8 声道 flac（~39min/个）+ 对应 `.rttm` | DER（说话人分离） |

> 服务 `/transcribe` 返回 `segments:[{start, end, speaker, text}]`，**start/end 单位为毫秒(ms)**，speaker 为整数字符串。DER 计算需 ÷1000 转秒。

---

## 2. 任务清单与状态

| # | 任务 | 状态 |
|---|---|---|
| 1 | 读取服务与评测脚本的启动/调用方式 | ✅ 完成 |
| 2 | CER 干跑校验（7176 wav ↔ transcript 匹配） | ✅ 完成（7176/7176 完美匹配） |
| 3 | 启动 FastAPI ASR 服务（后台进程） | ✅ 完成（本次已停，端口释放） |
| 4 | CER 冒烟 200 条 | ✅ 完成（CER=5.03%） |
| 5 | CER 全量 7176 条 | ⏸ **冻结**（用户要求暂不做） |
| 6 | 新建 der_eval.py（纯 Python 标准 DER） | ✅ 完成 |
| 7 | DER 单条 AISHELL-4 会议评测 | 🟡 跑通但**结果无效**（见 §4 根因） |
| 8 | DER 探针：短音频 diarize 实测 segments 字段 | ✅ 完成 |
| 9 | 生成 TEST_REPORT.md | ⏳ 改由本文档承接 |

---

## 3. 代码更改明细（本次测试相关的全部改动）

### 3.1 `asr_onnx_service.py`（已跟踪，M，+124 行）
两轮 GPU 提速改造：
1. **Cam++ 说话人模型 → CUDA**（带 CPU 回退）：`_get_spk_model()` 中 `device="cpu"` 改为 `try device="cuda" except ... device="cpu"`，不阻断服务启动。
2. **VAD (fsmn) → CUDA**（带 CPU 回退）：启动时 `AutoModel(... device="cpu")` 改为 `try device="cuda" except cpu`，同步 `.model.to()` 与 `kwargs["device"]`。PUNC 标点模型保持 CPU（非瓶颈）。
3. `/transcribe` 已支持 `diarize=true`（返回带 `speaker` 标签的 segments）与 `num_speakers`（oracle-K，默认 None）。
4. 单进程 uvicorn，`GPU_SEMAPHORE=3`。

### 3.2 `speaker_engine.py`（untracked，本次修复）
修复 **Cam++ 上 CUDA 的 `.numpy()` 崩溃**（这是 DER 早期全失败的根因链）：
- 第 64 行：输入 `chunk_16k` 先 `.cpu().numpy()`。
- 第 75 行：`model.generate()` 返回的 `spk_embedding` 是 **cuda 张量**，`np.asarray(cuda_tensor)` 崩溃 → 改为 `emb.cpu()` 后再 `np.asarray().flatten()`。
- 现象：未修复时服务端满屏 `[Speaker] 向量提取失败: can't convert cuda:0 device type tensor to numpy`，说话人向量全失败，服务对长音频只能给 `'?'` 标签。

### 3.3 `cer_eval_parallel.py`（untracked，本次新增）
CER 并行评测器：线程池并发打 `:8002` 的 `POST /transcribe`。
- 默认 `--engine gpu`（PyTorch-GPU）。
- 新增 `--sample N` **随机采样**（固定 `--seed`，避免只取前 N 条的分布偏差）。
- `--base_urls` 逗号分隔多实例轮询（为将来水平扩展提速预留）。
- JSONL 断点续跑。

### 3.4 `der_eval.py`（untracked，本次新增）
纯 Python 标准 DER 计算器（零额外依赖）：
- 8 声道 flac → 抽 **ch0 单声道 flac**（对齐 RTTM 单参考通道 + 规避服务 50MB 上传上限；40min 单声道 16k wav≈76MB 超限）。
- 调 `POST /transcribe?diarize=true`，取 `segments` 的 speaker 标签。
- 解析参考 RTTM，标准 DER：**0.25s collar + 贪心说话人映射**，统计 FalseAlarm / Missed / Confusion，全局按参考语音时长加权。
- **oracle-K**：先 parse_rttm 取真实人数，POST 时传 `num_speakers=K`（见 §4）。
- 已用 6 个合成场景单元自测（perfect / permuted / partial / falsealarm / overcluster / collar）全部正确。

### 3.5 其他（untracked，非本次测试核心产出，未纳入暂存）
`cer_eval.py`、`extract_aishell1_test.py`、`run_cer_test.ps1`、`一键启动_ASR_API服务.bat` 为会话前已存在文件。
> 注：工作区另有 `dify_openapi.yaml` 未暂存改动（Dify 集成：host.docker.internal + response_format 参数），**与本次 CER/DER 测试无关**，未纳入本次暂存。

---

## 4. 测试结果

### 4.1 CER（ASR 准确率）✅ 有效、可信
| 规模 | 引擎 | CER | 备注 |
|---|---|---|---|
| 冒烟 200 | CPU(Sherpa) | 5.03% | 0 失败，505s |
| 采样 1000（seed=42） | PyTorch-GPU | **6.53%** | 0 失败，395s，dist=948 / gt_len=14518 |

→ AISHELL-1 是近场朗读，模型表现正常，**6.53% 可作结论**。

### 4.2 DER（说话人分离）⚠️ 当前数字不可作系统水平参考
单条会议 `L_R003S01C02.flac`（AISHELL-4，6 说话人，REF≈2312s）：

| 聚类模式 | DER | 现象 |
|---|---|---|
| 阈值聚类（`num_speakers=None`，碎成 140 簇） | **31.01%** | 过分割，每微段单独对齐参考 → DER **虚低** |
| oracle-K=6（强制 6 簇） | **71.82%** | hyp 分布 `{'1':483, '2':2, '3':18, '4':1, '5':1, '6':1}` → **95% 时间被塞进 1 个簇** → 大段错归 → CONF 占 46.8% → DER **虚高** |

两个数字**方向相反、都失真**，都不代表该会议上 diarization 的真实水平。

---

## 5. DER 失真根因分析（重点，给修复者）

### 5.1 已排除的嫌疑
- ❌ **不是** `int('?')` 表象 bug（已修 `str(s["speaker"])`）。
- ❌ **不是** Cam++ CUDA 崩溃（已修 `speaker_engine.py` 第 75 行，短音频探针确认 speaker 返回真实整数）。
- ❌ **不是** VAD 漏检：hyp 段数 **506 ≈ 参考 512**，VAD 基本正常。
- ❌ **不是** 参考说话人分布极端：6 人时长较均衡（006-M 37.9% / 005-M 24.6% / 002-F 16.5% / 004-M 13.4% / 003-F 3.9% / 007-M 3.6%）。

### 5.2 真因（方法学 / 任务层）
**AISHELL-4 是远场 8 通道麦克风阵列会议，而 `der_eval.py` 抽的是 ch0 单通道**——等于把阵列带来的说话人区分增益全扔了。后果：
- Cam++ embedding 在远场单通道上区分度极差；
- `two_stage` 聚类第二阶段（agglomerative，强制 K=6）把 95% 的段合并进 1 个说话人；
- 阈值模式则过分割成 140 簇（虚低）。

即：**DER 失真源于"单通道输入 + 远场场景 + 聚类策略不匹配"，不是 ASR/说话人模型本身崩溃。**

---

## 6. 待高级模型解决的方向（how-to-fix 线索）

1. **【首选】多通道输入**：`der_eval.py` 的 `to_mono_ch0()` 改为多通道处理——
   - 方案 A：选能量最强 / 信噪比最佳的通道；
   - 方案 B：通道拼接 embedding（每个通道各提 Cam++ 向量后融合）；
   - 方案 C：beamforming（需阵列几何 / 频域滤波）。
   AISHELL-4 的 RTTM 绑的是单参考通道，需保证 hyp/ref 时间轴对齐。
2. **换更鲁棒的聚类后端**：用 pyannote 的 AHC / SpectralClustering 替换 `two_stage`（用户已授权装 venv），或调整 `two_stage` 阈值与 `oracle_K` 配合。
3. **embedding 融合**：远场下对多通道 embedding 做均值/拼接后再聚类，提升区分度。

---

## 7. 可复跑命令

```bash
# 启动服务（修改后需重启加载）
E:/conda/envs/asr_ui_env/python.exe E:/project/funclip-pro/asr_onnx_service.py   # :8002

# CER 采样（GPU，1000 条）
E:/conda/envs/asr_ui_env/python.exe cer_eval_parallel.py \
  --wav_dir testset/aishell1_test_extracted/wav \
  --transcript testset/aishell1_test_extracted/transcript.txt \
  --engine gpu --sample 1000 --workers 4 --seed 42 \
  --out test_results/cer_sample_1k.jsonl

# DER 单条（oracle-K，默认抽 ch0 单通道）
E:/conda/envs/asr_ui_env/python.exe der_eval.py \
  --audio_dir testset/dia-aishell4-test/audio/test \
  --rttm_dir  testset/dia-aishell4-test/rttm/test \
  --limit 1 --timeout 1800 --out test_results/der_single.json
```

---

## 8. 当前产物（已落盘，未提交）
- `test_results/cer_sample_1k.jsonl` + `_summary.json`：CER 采样结果（6.53%）。
- `test_results/der_single.json` + `_run.log`：DER 最近一次（oracle-K=6，71.82%，失真）。
- `test_results/cer_full.jsonl`：早期全量尝试的 153 条（已冻结，非最终）。
