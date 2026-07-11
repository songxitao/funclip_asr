# 测试方案设计：funclip-pro FastAPI ASR(CER) + 说话人分离(DER)

> 日期：2026-07-11 ｜ 遵循 superpowers 流程（brainstorming → 本设计 → writing-plans → 执行）
> 项目：`E:\project\funclip-pro` ｜ 被测对象：FastAPI 推理服务 `asr_onnx_service.py`（端口 8002）

## 0. 目标与范围（已与尖子确认）

- **不跑全量 pytest 单元套件**（仓库内 pytest 文件众多且含早期阶段文件，非本次重点）。
- **只针对 FastAPI 的 ASR 与说话人模型做端到端准确率测试**：
  1. **CER**（字错率）：AISHELL-1 test set（7176 条）→ ASR 准确率。
  2. **DER**（说话人分离错误率）：AISHELL-4 test set（20 条会议音频）→ 说话人分离质量。
- **CER 规模**：先冒烟 200 条验证管线，再跑全量 7176。
- **DER 规模**：默认先跑 1 条会议（~40min）验证管线，给出 DER；全量 20 条作为可选扩展（耗时约 1–2h，需二次确认再跑）。

## 1. 环境事实（已探针验证）

| 项 | 状态 |
|---|---|
| Python 环境 | `E:\conda\envs\asr_ui_env\python.exe`（3.11.14）✅ 核心依赖齐全 |
| 关键依赖 | funasr / sherpa_onnx / fastapi / librosa / torchaudio ✅ |
| GPU | CUDA 可用（RTX 4080，torch 2.3.1+cu121）✅ |
| 模型软链 | `E:\project\funclip-pro\model -> /e/FunClip/FunClip/model` ✅ 可达 |
| DER 库 | pyannote / dscore **均未安装** ⚠️ |

**约束**：按 AGENTS.md「缺模块先确认再装」规则，**不擅自 pip install pyannote/dscore**；
DER 评测用**纯 Python 实现的标准算法**（见 §3.2），零额外依赖、结果可比。

## 2. 被测接口（来自 `asr_onnx_service.py`，已 grep 确认）

`POST /transcribe`（multipart 音频 + Form 参数）：
- `file`：音频文件
- `vad_strategy`：`auto|always|never`（auto：短音频<5s 走 trim 直解，长音频走 FSMN VAD）
- `engine`：`cpu|gpu`（可选覆盖；auto 时 CUDA 可用且长音频→PyTorch-GPU，否则 Sherpa-CPU）
- `diarize`：`true|false`
- `diarize_strategy`：`single|two_stage`（默认 two_stage）
- `num_speakers`：可选 oracle-K（已知会议人数时最稳）
- 返回 JSON：`{text, latency_ms, engine, segments:[{start,end,text,speaker}], diarized_text}`
  - `diarize=true` 时：`segments` 携带每段 `speaker` 标签（Camp++ 聚类 id，从 1 起）
- 启动方式：`python asr_onnx_service.py`（uvicorn，端口 8002）；Cam++ 说话人模型**惰性加载**（首次 diarize 请求才加载）

> 注：执行时先做 1 次短音频 diarize 探针，实测 `segments` 的 `start/end` 单位（秒/毫秒）与字段名，再据此定稿 `der_eval.py`，避免读源码（遵循 HANDOFF「不直读源码」约定，改用实测）。

## 3. 测试设计

### 3.1 CER —— AISHELL-1 ASR 准确率（复用 `cer_eval.py`）

`cer_eval.py` 已就绪：遍历 wav → `POST /transcribe`（不带 diarize）→ 字符级编辑距离算 CER（默认去标点，与业界一致）。

| 步骤 | 命令（在 asr_ui_env 中） | 预期 |
|---|---|---|
| 干跑校验 | `python cer_eval.py --wav_dir testset/aishell1_test_extracted/wav --transcript testset/aishell1_test_extracted/transcript.txt --dry-run` | 匹配 7176/7176 |
| 冒烟 200 | `python cer_eval.py ... --limit 200 --base_url http://localhost:8002` | 输出 CER%（~30s） |
| 全量 7176 | `python cer_eval.py ... --base_url http://localhost:8002` | 最终 CER%（~15–25min） |

- 默认 `vad_strategy=auto` → AISHELL-1 短句走 Sherpa-CPU INT8 路径（快、零显存）。
- 可选扩展：用 `engine=cpu` / `engine=gpu` 各跑一次冒烟，对比双引擎 CER（INT8 vs FP32 差异约 ~51 字符级别）。

### 3.2 DER —— AISHELL-4 说话人分离（新建 `der_eval.py`，纯 Python）

**前置处理（多声道对齐）**：AISHELL-4 音频为 8 声道 16kHz flac（~40min/条），RTTM 描述的是房间混合参考信号。
脚本先用 `soundfile`/`librosa` 把每条 flac **转单声道（取 ch0）→ 临时 mono wav**，再送 `/transcribe?diarize=true`，保证 hyp/ref 时间轴一致。

**评测流程（单条会议）**：
1. `POST /transcribe?diarize=true`（可选 `num_speakers=<会议人数>` 走 oracle-K）→ 取 `segments`。
2. 解析参考 RTTM → `(start, end, ref_speaker)` 区间列表。
3. 把 hyp/ref 落到 **0.25s 时间栅格**，应用 **0.25s forgiveness collar**（忽略每段边界 ±0.25s）。
4. **贪心说话人映射**：hyp 聚类 id ↔ ref 标签按重叠最大匹配（最小化 confusion）。
5. 统计 Missed / FalseAlarm / Confusion / Total，算 **DER = (Miss+FA+Conf)/Total**。
6. 输出该会议 DER、检测到的说话人数、与 ref 人数对比。

**运行规模**：
- 默认：1 条会议（如 `S_R003S04C01`）→ 验证管线 + 给 DER（单条约数分钟，CPU Cam++）。
- 全量 20 条 → 平均 DER（可选，~1–2h，需二次确认）。

**DER 算法实现要点（无外部库）**：
- RTTM 解析：标准 `SPEAKER <file> 1 <start> <dur> ... <spk>` 格式。
- 时间栅格法（0.25s 步长）避免区间合并的边界 bug；collar 用区间运算排除边界。
- 贪心映射：先按 ref 人数构造 cost matrix（confusion 量），贪心分配 hyp→ref。
- 输出 JSON + 文本摘要，写入 `testset/dia-aishell4-test/der_report.json`。

### 3.3 服务生命周期管理

- 起一个 `asr_onnx_service.py` 后台进程（端口 8002），CER 与 DER 复用同一服务。
- 启动后轮询 `http://localhost:8002` 直到就绪（参考 `run_cer_test.ps1` 的等待逻辑，最长 ~80s）。
- 测试结束按需保留/停止（默认跑完即停，释放 GPU 显存）。

## 4. 产物与报告

- `docs/superpowers/specs/2026-07-11-cer-der-test-plan-design.md`（本设计）
- 执行后产出 **`TEST_REPORT.md`**（根目录或 docs/），含：
  - 环境指纹（Python / CUDA / 引擎路由）
  - CER：干跑匹配数、冒烟 200 CER%、全量 7176 CER%、耗时
  - DER：每条会议 DER、说话人数对比、聚合 DER（若跑全量）
  - 命令清单（可直接复跑）
- 不自动 commit（用户未要求）；如需提交再确认。

## 5. 风险与缓解

| 风险 | 缓解 |
|---|---|
| AISHELL-4 单条 ~40min，全量 20 条耗时 1–2h | 默认只跑 1 条；全量需二次确认 |
| 8 声道音频与 RTTM 不对齐 | 转单声道 ch0 后再送服务 |
| `segments` 字段/单位未知 | 执行前 1 次短音频 diarize 探针实测 |
| pytest 全目录收集触发 capture 冲突 | 本次根本不跑 pytest 套件，规避 |
| pyannote/dscore 缺失 | 纯 Python DER，零依赖；如需 pyannote 标准 DER 另行确认安装 |

## 6. 验收标准

- [ ] 服务在 8002 正常起停，CER 与 DER 共用。
- [ ] CER 干跑匹配 7176/7176；冒烟 200 与全量 7176 均产出 CER%。
- [ ] DER 在 ≥1 条 AISHELL-4 会议上产出合理 DER 与说话人映射说明。
- [ ] 产出 `TEST_REPORT.md`，命令可复跑。
- [ ] 全程未擅自安装任何包（遵守 AGENTS.md 约束）。
