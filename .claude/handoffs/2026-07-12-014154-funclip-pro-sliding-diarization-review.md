# Handoff: funclip-pro 滑窗说话人分离（sliding diarization）代码审查交接

> 目的：本 handoff 供**代码审查者**使用。重点交代「滑窗说话人分离」功能的来龙去脉、改了哪些文件、设计决策、已知问题、如何复现验证，以便审查者零歧义地评审代码质量与正确性。所有相对路径均相对于项目根 `E:\project\funclip-pro`。

## Session Metadata
- Created: 2026-07-12 01:42 (GMT+8)
- Project: E:\project\funclip-pro
- Branch: main
- Session duration: 约 3 小时（07-11 晚至 07-12 凌晨），跨多轮对话
- 审查对象：commit `48fc641` → `b53d672`（共 7 笔，均已在 main 分支，工作区干净）

### Recent Commits (本次功能相关，由新到旧)
```
b53d672 fix: ali_der_eval __main__ 透传 strategy 参数(sliding 验证生效)
dead2b1 chore: gitignore 排除 AliMeeting 测试集 Test_Ali(6GB) 与试跑产物 ali_near_prep
5d97f8d test: AliMeeting near 全量20场DER评测脚本 + eval_one strategy参数化
6286b12 feat: /transcribe diarize_strategy=sliding 滑窗说话人分离分支 + 集成测试
cad4c79 feat: CampPlusSpeaker.cluster_sliding 滑窗聚类+合并方法 + 单测
48fc641 feat: 滑窗 segmentation 切分函数 segment_sliding_window + 单测
e73d6a0 test: AliMeeting 近场预处理与 DER 评测脚本（试跑验证通过）
```

## Handoff Chain
- **Continues from**: `HANDOFF.md` (2026-07-09 版，记录 AISHELL-4 远场 CER/DER 评测阶段，本次已超越但未推翻)
- **Supersedes**: 无

## Current State Summary

项目 funclip-pro 是一个 FastAPI 语音转写 + 说话人分离服务。本次工作围绕「说话人分离（diarization）DER 虚高」问题展开：原方案用 VAD 段（为 ASR 设计、段长 ~8s、段内常混多人）喂给 Cam++ 提 embedding，导致聚类被污染、混淆错误（CONF）占 DER 的 48%。经 brainstorm 论证，改用**滑窗 segmentation**（1.5s 窗 + 0.5s 步长）替代 VAD 段作为说话人分离的输入切分方式，Cam++ 模型与 spectral 聚类逻辑不变。代码已实现并提交（Task 1-5），单场零样本验证显示 sliding 相对旧 spectral/VAD 段 DER 从 49.21% 降到 29.76%。**剩余 Task 6（全量 20 场评测 + 对比报告）尚未运行**，因为耗时 1-2 小时，待用户确认。`ali_der_eval.py` 的 CLI 入口 bug（漏传 strategy 参数）已发现并修复（b53d672）。

## Codebase Understanding

### Architecture Overview

```
客户端上传音频 → /transcribe 端点 (asr_onnx_service.py)
  ├─ VAD (funasr) 切段 → ASR 识别文字 → clean_texts[]
  └─ 说话人分离分支（diarize=true 时）:
       旧: spk_model.cluster(chunks, strategy=diarize_strategy, ...)   ← 用 VAD 段
       新: spk_model.cluster_sliding(y, sr=16000, ...)                ← 用整段音频内部滑窗
            segment_sliding_window() 切 1.5s 窗
            → 逐窗 extract_embedding() (Cam++)
            → SpectralClustering 聚类
            → 合并相邻同人窗 → segments[]
  → 合并文字+说话人 → 返回 {segments:[{start,end,speaker,text}]}
```

关键分离点：**说话人分离的输入切分方式**（VAD 段 vs 滑窗）是本次唯一改动的链路环节。ASR 链路、Cam++ 模型、spectral 聚类算法均复用，未改动。

### Critical Files

| File | Purpose | Relevance |
|------|---------|-----------|
| speaker_engine.py | `CampPlusSpeaker` 类：Cam++ embedding + 聚类 | **核心改动**：新增 `segment_sliding_window()` 模块函数 + `cluster_sliding()` 方法 |
| asr_onnx_service.py | FastAPI `/transcribe` 端点 | **核心改动**：diarize 分支新增 `diarize_strategy == "sliding"` 走 `cluster_sliding` |
| tests/test_sliding_segmentation.py | 滑窗切分 + 合并单测 | 审查单测覆盖是否充分（mock embedding，不加载模型） |
| tests/test_sliding_integration.py | sliding 分支集成测试 | 审查 mock 是否真实覆盖服务分支 |
| ali_der_eval.py | 单场 DER 评测脚本（试跑） | 含已修复的 CLI bug（b53d672） |
| ali_near_prep.py | AliMeeting near 预处理：混音 + TextGrid→RTTM | 评测前置，零依赖 |
| run_ali_der_full.py | 全量 20 场评测脚本 | Task 5 产物，未实际运行全量 |
| der_eval.py | `compute_der()` 纯 Python DER 计算（0.25s collar） | 评测口径，注意非 pyannote 标准 |
| docs/superpowers/specs/2026-07-12-alimeeting-diarization-sliding-window-design.md | 设计 spec | 审查设计依据 |
| docs/superpowers/plans/2026-07-12-alimeeting-diarization-sliding-window.md | 实施计划 | 代码与计划的对应性审查 |

### Key Patterns Discovered

- **Cam++ 推理兼容 cuda/cpu**：`extract_embedding()` 内部处理 tensor，返回 numpy，调用方无需关心设备。
- **聚类策略枚举**：`cluster()` 支持 `single`/`spectral`/`two_stage`；新增的 `cluster_sliding()` 复用 `spectral` 分支的 `SpectralClustering` 参数（n_clusters 估算、affinity、n_neighbors、random_state=42）。
- **speaker_id 约定**：`cluster()` 返回 labels 从 0 起 +1 存；`cluster_sliding()` 同样从 1 起，与服务端 `str(spk)` 一致。
- **`y` 变量**：`/transcribe` 内部 `_run_inference()` 中 `y` 是整段 16k 单声道 numpy，sliding 分支直接使用它（不经 VAD 处理）。

## Work Completed

### Tasks Finished
- [x] 诊断：AISHELL-4 远场 DER 虚高（MISS 45K 未排、仅测 1 场、oracle-K 作弊）
- [x] 决定弃 AISHELL-4 远场，换 AliMeeting 近场测试集（已下载到 `testset/Test_Ali`）
- [x] 写 design spec + 实施计划（6 Task，TDD）
- [x] 子智能体实现 Task 1-5：滑窗切分函数、cluster_sliding、服务 sliding 分支、全量脚本、单测
- [x] 单测验收：`pytest tests/` → 7 passed（segmentation 6 + integration 1）
- [x] gitignore 排雷：排除 Test_Ali(6GB) + ali_near_prep 产物
- [x] 起服务跑单场 sliding，发现数字与旧方案雷同 → 定位 `ali_der_eval.py` CLI 漏传 strategy
- [x] 修复 CLI bug（b53d672），重跑验证：sliding DER=29.76% vs spectral 49.21%

### Files Modified (本次功能，按 commit)

| File | Changes | Rationale |
|------|---------|-----------|
| speaker_engine.py | 新增 `segment_sliding_window()`（模块级）+ `CampPlusSpeaker.cluster_sliding()` | 滑窗切分 + 聚类 + 合并，替代 VAD 段喂 Cam++ |
| asr_onnx_service.py | diarize 分支新增 `if diarize_strategy == "sliding": cluster_sliding(y, ...)` | 服务暴露 sliding 策略 |
| tests/test_sliding_segmentation.py | 新增（切分 3 测 + 合并逻辑 3 测） | 单元验证滑窗逻辑 |
| tests/test_sliding_integration.py | 新增（mock Cam++ + ASR，验证 sliding 分支走通） | 集成验证服务端分支 |
| ali_near_prep.py | 新增（零依赖混音 + TextGrid→RTTM） | 评测前置 |
| ali_der_eval.py | 新增 + **修复 `__main__` 透传 strategy**（b53d672） | 单场评测，bug 修复 |
| run_ali_der_full.py | 新增（全量 20 场，支持 --strategy） | Task 5 产物 |
| .gitignore | 排除 `testset/Test_Ali/`、`testset/ali_near_prep/` | 排 6GB 数据集雷 |
| docs/superpowers/specs/2026-07-12-alimeeting-diarization-sliding-window-design.md | 新增 | 设计依据 |
| docs/superpowers/plans/2026-07-12-alimeeting-diarization-sliding-window.md | 新增 | 实施计划 |

### Decisions Made

| Decision | Options Considered | Rationale |
|----------|-------------------|-----------|
| 说话人分离改滑窗 segmentation，不动 Cam++/聚类 | A 调小 VAD max_duration（治标硬切）/ B 滑窗（根本解）/ C VAD段内二次切分（折中） | 滑窗是业界标准（pyannote），VAD 段为 ASR 优化、段内混多人污染 embedding；A 硬切会切坏、C 仍受 VAD 边界限制 |
| 测试集弃 AISHELL-4 远场，换 AliMeeting near | 近场部署 / 学术对标 / 两者 | 实际部署是近场单声道，AISHELL-4 远场 8 麦阵列单通道与部署不匹配；near 领夹麦混单通道最接近部署场景 |
| 用子智能体实现代码 | 主智能体直接写 / 子智能体写 | 用户要求用子智能体接手 plan 的实施 |
| `y`（整段 16k）喂 sliding | 复用 VAD chunks | sliding 要自己切窗，不需要外部 VAD 段；VAD 仅服务 ASR 文字 |
| 不实际跑全量 Task 6 | 立即跑 / 等确认 | 耗时 1-2h，用户未确认是否现在跑 |

## Pending Work

## Immediate Next Steps (审查者行动项)
1. **代码审查**：审查者重点看 `cluster_sliding()` 的 None 填充逻辑（无效窗用前序标签填，首窗无效填 1）是否合理，以及合并相邻同人窗是否丢失边界精度。
2. **运行全量 Task 6**：`python run_ali_der_full.py --strategy sliding` + `--strategy spectral`，产出 20 场加权平均对比报告（验证 CONF 全量是否稳定低于旧方案）。
3. **补强评测**：当前 DER 用纯 Python `compute_der`（0.25s collar），与 pyannote 标准 DER 可能存在口径差异，学术对标场景需补 pyannote 校验。

### Blockers/Open Questions
- [ ] **全量 20 场未跑**：单场 29.76% 是否代表整体未知；需 Task 6 验证。
- [ ] **SPK4 时长仅 19.7s**（单场）：少数说话人是否被滑窗漏检，需看全量分布。
- [ ] **FA 略增**（0.3%→3.4%）：滑窗切细后短段误标有人，是否可接受需权衡。
- [ ] **叠词/快速抢话场景**：滑窗 1.5s 窗的天花板，当前未专门处理（design spec 已声明局限）。
- [ ] **`diarize_strategy` 参数透传链**：`/transcribe` 的 Form 参数 → `_run_inference` → sliding 分支，审查者需确认参数名全链路一致。

### Deferred Items
- far 对照评测（AliMeeting far 单通道 vs near，证明远场吃亏）——用户暂未要求。
- oracle-K 是否保留——`cluster_sliding` 支持 `n_speakers` 参数（oracle），但部署用不到，评测时传真实人数。

## Important Context (审查者必读)

- **这是给审查者的交接，不是实施交接**：不要重复实现，重点评审 `speaker_engine.py` 与 `asr_onnx_service.py` 的 sliding 改动、单测质量、CLI bug 修复。
- **核心假设已单场验证成立**：VAD 段粒度是 DER 虚高主因，滑窗 DER 49%→30%。但全量未验证，审查时可要求先跑 Task 6 再下结论。
- **测试集 6GB 已被 gitignore**：`testset/Test_Ali/` 不入库，审查者本地需已有该数据集才能复现评测（已下载在 `E:\project\funclip-pro\testset\Test_Ali`）。
- **单测用 mock**：`test_sliding_segmentation.py` 用 `unittest.mock.patch` mock `extract_embedding`，不加载真实 Cam++ 模型；集成测试 mock 了 Cam++ 和 ASR。**未覆盖真实模型端到端**。
- **服务端口 8002**：评测时服务需运行（`python asr_onnx_service.py`，conda `asr_ui_env`）。

### Assumptions Made
- 假设 1.5s 窗 + 0.5s 步长适合会议对话（单人通常说 ≥1s），未对窗长做调优扫描。
- 假设 `y` 在 `/transcribe` 内始终是整段 16k 单声道（需审查确认无分支改成其他形态）。
- 假设 spectral 聚类对 ~2000 窗可行（单场 34min 约 2000 窗，未做性能压测）。

### Potential Gotchas
- **CLI bug 已修但易复发**：`ali_der_eval.py` 的 `__main__` 之前漏传 `strategy`，导致 sliding 静默回退 spectral。审查 `eval_one` 调用链时注意参数透传。
- **`cluster_sliding` 返回 segments 的 text 为空**：sliding 分支不回填 ASR 文字（design 如此），DER 评测只看 start/end/speaker，不影响；但若有下游依赖 text 需注意。
- **pytest capture bug**：本机 pytest 收尾偶发 `I/O operation on closed file`，用 `-s -p no:cacheprovider` 可规避，非代码问题。
- **codegraph 未索引后写脚本**：`ali_der_eval.py`/`run_ali_der_full.py` 是后期脚本，codegraph 可能未索引，审查时直接 Read 即可（本项目允许实现阶段读代码）。

## Environment State

### Tools/Services Used
- Python 环境（唯一）：`E:\conda\envs\asr_ui_env\python.exe`（funasr/sklearn/torch/soundfile/pytest 齐全，CUDA 可用）
- codegraph 1.2.0（代码符号查询，本项目规范：诊断阶段只读 md + codegraph）
- FastAPI 服务 `asr_onnx_service.py`，端口 8002

### Active Processes
- 后台 ASR 服务：`:8002`（撰写本 handoff 时运行，带 sliding 改动）。如需重启：`E:\conda\envs\asr_ui_env\python.exe asr_onnx_service.py`
- 注意：若 8002 被旧服务占用，需先 `taskkill /PID <pid> /F`（用 `netstat -ano | grep :8002` 查 pid）

### Environment Variables
- 无新增环境变量。模型路径硬编码在 `asr_onnx_service.py`（不要改）。

## Related Resources
- 设计 spec：`docs/superpowers/specs/2026-07-12-alimeeting-diarization-sliding-window-design.md`
- 实施计划：`docs/superpowers/plans/2026-07-12-alimeeting-diarization-sliding-window.md`
- 测试集：`testset/Test_Ali/Test_Ali/{Test_Ali_near,Test_Ali_far}/{audio_dir,textgrid_dir}`
- 单场评测结果（已验证）：R8002_M8002 sliding DER=29.76%（CONF 26.3%、MISS 0%、FA 3.4%、hyp 444 段）；旧 spectral DER=49.21%（CONF 48.3%、hyp 102 段）
- 历史诊断报告：`TEST_REPORT.md`、`testset/CER_DER_TEST_REPORT.md`

---

**Security Reminder**: Before finalizing, run `validate_handoff.py` to check for accidental secret exposure.
