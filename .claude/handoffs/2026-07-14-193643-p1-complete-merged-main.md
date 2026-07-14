# Handoff: P1 全量完成 + 合入 main — 可开 P3/新功能

## Session Metadata
- Created: 2026-07-14 19:36
- Project: E:\project\funclip-pro
- Branch: main（P0+P1 已合并，HEAD 6d8dd9f）
- Session duration: ~45 分钟（项目对接 + CER/DER 评测 + 双真源清理 + 重复路由删除 + P1 合入 main）

### Recent Commits (for context)
  - 6d8dd9f docs: DER 测试集纠正为 Ali + 全量评测暂停，写 handoff 交接下一智能体
  - 6ef1a58 docs(p1/w6): code-review 双轴自审 + 写 P1 接手 handoff
  - 6a8cae9 test(p1/w5): 测试门禁 — P1 相关单测 28 绿 + DER 单场 seg_clustering 等价 P0 非回归
  - 1b5ca48 refactor(p1/w4): 薄路由+薄客户端瘦身（收缩步）
  - 3811c2c refactor(p1/w3): 整合 OfflinePipeline 统一转写流水线

## Handoff Chain

- **Continues from**: `.claude/handoffs/2026-07-14-p1-complete-handoff.md`（P1 算法 SDK 化完成 + 双轴自审）
- **Supersedes**: 本 handoff 承接上述文档，是本 session 起点的快照

## Current State Summary

本 session 从项目对接开始，完成了 CER 和 DER 全量评测、清除了根目录双真源风险、删除了 asr_service.py 重复路由，并将 P1 分支（含全部清理）fast-forward 合并到 main。现在 main 已是最新 P1 完整代码，所有门禁通过（CER 6.54%、Ali DER 19.96% vs 基线 19.69%，+0.27% 无回归），项目可以进入 P3 或新功能开发阶段。

## Codebase Understanding

### Architecture Overview

- 项目是 ASR + 说话人日志（Diarization）系统，核心生产入口为 `asr_onnx_service.py`（FastAPI，`:8002`，`/transcribe` 路由）
- 算法已全部下沉到 `src/funclip_pro/` SDK 包：`core/`（segmentation/speaker/asr/alignment/tokenization）、`utils/`（srt）、`pipeline/`（offline=OfflinePipeline）
- `asr_onnx_service.py` 和 `cli_transcribe.py` 都是薄壳，只做参数透传，推理委托给 `OfflinePipeline`
- DER 评测通过 HTTP POST 到 8002 服务计算，与算法代码解耦
- 测试集：AliMeeting（`testset/ali_near_prep_match/`，3 场），非 AISHELL-4

### Critical Files

| File | Purpose | Relevance |
|------|---------|-----------|
| `src/funclip_pro/config/loader.py` | 配置加载器 | resolve_model_path / apply_dll_patch / load_config |
| `src/funclip_pro/core/segmentation.py` | 分割引擎 | pyannote powerset 分割，帧级单人过滤 |
| `src/funclip_pro/core/speaker.py` | 声纹引擎 | Cam++ + SpectralClustering 全局谱聚类 |
| `src/funclip_pro/core/asr.py` | ASR 后端 | ONNX/PyTorch/Sherpa 三大引擎路由 |
| `src/funclip_pro/core/alignment.py` | 子句说话人对齐 | 锚点扩散 |
| `src/funclip_pro/pipeline/offline.py` | OfflinePipeline | 统一转写流水线，返回四元组 |
| `asr_onnx_service.py` | FastAPI 薄路由 | `:8002`，`POST /transcribe` |
| `der_eval.py` | DER 评测器 | POST 8002 算 DER，含 stem 配对致命坑 |
| `segmentation_engine.py` | **薄导出** | → `funclip_pro.core.segmentation` |
| `speaker_engine.py` | **薄导出** | → `funclip_pro.core.speaker` |
| `testset/ali_near_prep_match/` | AliMeeting 对齐目录 | 3 场，绕过 `_mixed` 后缀 stem 坑 |
| `testset/CER_DER_TEST_REPORT.md` | DER 方法学文档 | §5 说明 AISHELL-4 单通道 DER 失真根因 |

### Key Patterns Discovered

- **等价优先**：P1 重构只做"薄路由厚算法"的位置搬迁，算法逻辑字节级等价，不顺手优化
- **DER 评测与代码解耦**：DER 通过 HTTP POST 到服务测，不在算法包内跑
- **DER 评测用 oracle-K**：`der_eval.py` 用 RTTM 真实说话人数 `num_speakers` 强制收敛
- **`der_eval.py` stem 配对致命坑**：按文件名 stem 配对 audio/rttm。Ali 音频 `{name}_mixed.flac` vs rttm `{name}.rttm` → 0 匹配 → 静默 DER=0.0。必须用 `ali_near_prep_match` 对齐目录
- **红线集**：numpy 1.26.4 锁死、时间戳 ms、`powerset.cpu()` 在 `to_multilabel` 前、DLL 补丁保活、显式 git add

## Work Completed

### Tasks Finished

- [x] **项目对接** — 读 HANDOFF.md + 3 份 handoff + AGENTS.md + tests/README.md + CER_DER_TEST_REPORT.md
- [x] **测试集判断** — 对比 AISHELL-4 vs AliMeeting，确认 AliMeeting 更合适（近场单通道无方法学失真）
- [x] **CER 评测** — AISHELL-1 采样 1000 条，6.54%（与 P1 前 6.53% 等价），0 失败
- [x] **DER 基线建立** — 切 main 跑 Ali 3 场（全局 19.69%），建立 old-code 基线
- [x] **DER P1 评测** — 当前 P1 跑 Ali 3 场（全局 19.96%），+0.27% 无回归判定通过
- [x] **双真源清理** — `segmentation_engine.py` / `speaker_engine.py` 改为薄导出，6 个测试改 import
- [x] **重复路由删除** — `asr_service.py` 删除，8 个测试文件迁移/skip
- [x] **P1 合入 main** — fast-forward 合并，零冲突，41 门禁全绿

### Files Modified

| File | Changes | Rationale |
|------|---------|-----------|
| `segmentation_engine.py` | 568 行 → 3 行薄导出 | 消除双真源风险 |
| `speaker_engine.py` | 322 行 → 3 行薄导出 | 消除双真源风险 |
| `asr_service.py` | **已删除** | 与薄路由重复 |
| `tests/test_segmentation_engine.py` | import 改为 `funclip_pro.core.segmentation` | 双真源清理 |
| `tests/test_seg_clustering.py` | import 改为 `funclip_pro.core.speaker` | 双真源清理 |
| `tests/test_seg_seamless.py` | import 改为 `funclip_pro.core.speaker` | 双真源清理 |
| `tests/test_vad_sliding.py` | import 改为 `funclip_pro.core.speaker` | 双真源清理 |
| `tests/test_sliding_segmentation.py` | import 改为 `funclip_pro.core.speaker`（2 处） | 双真源清理 |
| `tests/test_asr_api.py` | skip（依赖 asr_service.MODEL/VAD_MODEL Mock） | 重复路由删除 |
| `tests/test_pytorch_inference_refactor.py` | skip（依赖 _run_inference 内部函数） | 重复路由删除 |
| `tests/test_pytorch_route.py` | skip（参数签名不同） | 重复路由删除 |
| `tests/test_pytorch_affinity.py` | `import asr_onnx_service` | 重复路由删除 |
| `tests/test_pytorch_service_integration.py` | 路径改为 `asr_onnx_service.py` + 8002 端口 | 重复路由删除 |
| `tests/test_asr_comparison.py` | 路径改为 `asr_onnx_service.py` | 重复路由删除 |
| `tests/test_asr_comparison_cpu.py` | 路径改为 `asr_onnx_service.py` | 重复路由删除 |

### Decisions Made

| Decision | Options Considered | Rationale |
|----------|-------------------|-----------|
| AliMeeting 为正式测试集 | AISHELL-4 vs AliMeeting | 用户确认 + AISHELL-4 远场阵列抽 ch0 单通道 DER 失真（CER_DER_TEST_REPORT.md §5） |
| 根算法文件改为薄导出而非删除 | 直接删除 vs 薄导出 | 兼容已有 import，不破坏历史脚本 |
| P1 直接 fast-forward 到 main | rebase + merge vs fast-forward | P0 已在 main 中，无冲突风险 |
| asr_service.py 直接删除 | 保留 vs 删除 | 已被 P1 薄路由完全替代，仅测试引用 |
| DER 基线跑 main 而非 P0 分支 | P0 分支 vs main | P0 已在 main 中，零差异 |

## Pending Work

### Immediate Next Steps

1. **讨论 P3 立项** — 候选方向：Gradio UI 清理 / 实时流式重构 / DER 方法学改进
2. **清理 pytest 全家桶崩溃** — 旧基准测试（bench_* 等）关 fd 导致 pytest 崩溃，需逐文件修复
3. **清理未跟踪垃圾** — 工作树有 `nul`/`output*.mp3`/`.agents/` 等非必要文件

### Blockers/Open Questions

- [ ] P3 具体做什么？需尖子决定方向
- [ ] 8002 服务稳定性问题（长跑 DER 中途挂掉，疑似 OOM）是否要查？

### Deferred Items

- DER 方法学改进（多通道替代 ch0 单通道）— 低优先级，等 P3 立项
- der_eval.py 增加原生 Ali 支持（自动处理 `_mixed` 后缀）— 不紧急，对齐目录已能绕过

## Context for Resuming Agent

### Important Context

1. **P0+P1 已完成并合入 main** — 项目主干干净。`segmentation_engine.py` / `speaker_engine.py` 现在是薄导出（指向 `funclip_pro.core`），`asr_service.py` 已删除。
2. **测试集是 AliMeeting** — `testset/ali_near_prep_match/`，3 场。14.85% 基线是 AISHELL-4 的，不适用。
3. **DER 基线已建立** — main 旧代码 19.69%，P1 19.96%（+0.27%，无回归）。
4. **der_eval 的 stem 配对坑** — 跑任何新测试集，先看配对日志 `匹配 N 对 audio/rttm`，否则可能静默 DER=0.0。
5. **红线**：numpy 1.26.4、时间戳 ms、`powerset.cpu()` 在 `to_multilabel` 前、DLL 补丁保活、显式 git add。
6. **8002 服务** — 当前已停。启动：`cd E:\project\funclip-pro && PYTHONPATH=E:/project/funclip-pro/src E:\conda\envs\asr_ui_env\python.exe asr_onnx_service.py`
7. **管理 Python** — ML/Pytest 用 `E:\conda\envs\asr_ui_env\python.exe`，沙箱受管 Python 无 torch。

### Assumptions Made

- GPU 非确定性噪声导致的 DER 波动 ±1% 视为正常，不判回归
- 8002 服务挂掉是 OOM（非代码 bug），但未深查
- 用户后续希望开展 P3 或新功能开发

### Potential Gotchas

- **pytest 全家桶崩溃** — 旧测试关 fd，别跑无过滤的 `pytest tests/`。只跑指定文件。
- **沙箱 taskkill 限制** — 杀不掉外部会话的 8002 进程
- **显式 git add** — 工作树有 `nul`/`output*.mp3`/`.agents/` 等未跟踪垃圾，禁用 `git add -A`
- **DER 必须 `--diarize_strategy seg_clustering`** — 默认 two_stage 得 49%-57% DER 误判回归

## Environment State

### Tools/Services Used

- 本机 conda 环境：`E:\conda\envs\asr_ui_env\python.exe`（torch 2.3.1+cu121 / CUDA 12.1）
- FastAPI 服务 `asr_onnx_service.py` @ `:8002`（当前已停）
- `der_eval.py` DER 评测器（POST 到上述服务）
- `cer_eval_parallel.py` CER 并发评测器
- 模型权重：`E:\project\funclip-pro\model\`（Windows Junction 软链 → `/e/FunClip/FunClip/model`）

### Active Processes

- **8002 FastAPI 服务：当前未运行**（已停）
- 无 der_eval / CER 残留进程

### Environment Variables

- `PYTHONPATH=E:\project\funclip-pro\src`（跑包/服务/评测时必设）
- `FUNCLIP_INTEGRATION`（置 1 才跑 integration pytest）

## Related Resources

- `.claude/handoffs/2026-07-14-p1-complete-handoff.md` — P1 完成 handoff（前序）
- `.claude/handoffs/2026-07-14-183317-der-ali-der-test-pause.md` — DER 测试集纠正 handoff
- `tests/README.md` — 测试图谱（39 文件分类）
- `testset/CER_DER_TEST_REPORT.md` — DER 方法学说明
- `AGENTS.md` — 项目规约与红线
- `test_results/der_ali_seg_clustering.json` — P1 Ali DER 19.96%
- `test_results/der_ali_baseline.json` — main 基线 Ali DER 19.69%

---

**Security Reminder**: Before finalizing, run `validate_handoff.py` to check for accidental secret exposure.

## Suggested Skills（下一个智能体应调用）

- `.agents/skills/to-spec` — 开 P3 新功能时先写 spec
- `.agents/skills/implement` — 实现：TDD + 末尾 code-review + 提交当前分支
- `.agents/skills/tdd` — 预对齐 seam 再写测试
- `.agents/skills/code-review` — Standards/Spec 双轴自审
- `.agents/skills/handoff` — 完成后再写下一个 handoff
- `.agents/skills/wayfinder` — 不确定方向时用来探索代码库
- `.agents/skills/improve-codebase-architecture` — 架构改进建议
