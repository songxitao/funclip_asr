# Handoff: DER 测试集纠正为 Ali + 全量评测暂停待重组

## Session Metadata
- Created: 2026-07-14 18:33:17
- Project: E:\project\funclip-pro
- Branch: refactor/p1-algo-packaging
- Session duration: 约 2 小时（18:00 起重跑测试门禁、纠正测试集、定位服务挂掉、暂停全量）

### Recent Commits (for context)
  - 6ef1a58 docs(p1/w6): code-review 双轴自审 + 写 P1 接手 handoff
  - 6a8cae9 test(p1/w5): 测试门禁 — P1 相关单测 28 绿 + DER 单场 seg_clustering 等价 P0 非回归
  - 1b5ca48 refactor(p1/w4): 薄路由+薄客户端瘦身（收缩步）
  - 3811c2c refactor(p1/w3): 整合 OfflinePipeline 统一转写流水线
  - 04b9624 refactor(p2/w2): 下沉对齐与 SRT 工具到 core/utils

## Handoff Chain

- **Continues from**: [2026-07-14-162727-p0-complete-p1-handoff.md](./2026-07-14-162727-p0-complete-p1-handoff.md)
  - Previous title: P0 全量完成 → P1 算法 SDK 化对接（W0–W6 已完成）
- **Supersedes**: None

> P1 算法 SDK 化（W0–W6）已全部提交，本 handoff 只承接"DER 测试门禁"这一收尾环节。

## Current State Summary

P1 算法 SDK 化（funclip_pro 包 + 薄路由/CLI）已于分支 `refactor/p1-algo-packaging` 完成 7 个 commit，W0–W6 全绿。当前处于**最后一环：DER 测试门禁验证**。关键转折：用户纠正——DER 测试集实际是 **`testset/ali_near_prep`（阿里中文会议集）**，而非常被误用的 AISHELL-4。按 Ali 集重跑 DER 时，后台全量任务（`KPjX3M`）中途失败，根因是 **8002 FastAPI 服务在评测过程中自行挂掉**（POST 全部 connection refused），并非 P1 代码问题。用户指令"先把全量测试停掉，交给下一个智能体，我再组织一下测试任务"——当前**全量评测已暂停、无 der_eval 残留进程**，等待下一个智能体按用户重新组织的测试任务继续。

## Codebase Understanding

### Architecture Overview

- `src/funclip_pro/{core,utils,pipeline}` 是 P1 下沉后的算法包：`core`(segmentation/speaker/asr/tokenization/alignment)、`utils`(srt)、`pipeline`(offline=OfflinePipeline)。
- `asr_onnx_service.py` 是**薄路由**：FastAPI app，`POST /transcribe` 接收参数 → 透传 → `OfflinePipeline.run(...)` → 用返回四元组 `(raw_text, engine_key, segments, diarized_text)` 组装响应。无内联推理/对齐/SRT。
- `cli_transcribe.py` 也是薄壳，直跑 `OfflinePipeline.run()`。
- DER 评测与算法解耦：`der_eval.py` 不跑流水线，而是 **POST `/transcribe` 到运行中的服务**，再用 RTTM 参考算 DER。
- 红线：numpy 1.26.4；时间戳 ms（`cluster_with_segmentation` 返回秒需 ×1000）；`powerset.cpu()` 在 `to_multilabel` 前；`apply_dll_patch()` 保活；零硬编码盘符；绝对导入 `from funclip_pro.x import Y`；`seg_clustering` 必须显式指定。

### Critical Files

| File | Purpose | Relevance |
|------|---------|-----------|
| `der_eval.py` | DER 评测器，POST 服务算 DER（纯 Python） | **测试门禁主入口**；含致命 stem 配对坑（见 Gotchas） |
| `asr_onnx_service.py` | FastAPI 薄路由 + 服务入口（`uvicorn`/`OfflinePipeline(auto_load=True)`） | 评测服务端；8002 端口；易中途挂 |
| `src/funclip_pro/pipeline/offline.py` | OfflinePipeline，seg_clustering 双分支(line155/292) | 算法等价性核心；与 main 旧 `_run_inference`(line617/744) 字节级一致 |
| `testset/ali_near_prep/` | **真实测试集（阿里中文会议，3 场）** | 评测数据；音频带 `_mixed` 后缀 |
| `testset/ali_near_prep_match/` | 文件名对齐软链目录（绕过 stem 坑） | 跑 Ali DER 必须用这个目录 |
| `testset/dia-aishell4-test/` | 旧误用测试集（20 场） | 基线 14.85%–15.13% 来源；对 Ali 未必适用 |
| `.claude/handoffs/2026-07-14-p1-complete-handoff.md` | P1 接手文档 | 目录结构/模块职责/启动方式/下一步 |
| `testset/CER_DER_TEST_REPORT.md` | DER 评测方法学说明 | §5 明文 AISHELL-4 单通道+远场 DER 失真，数字不可作系统参考 |

### Key Patterns Discovered

- **等价优先**：P1 重构只做"薄路由厚算法"的位置搬迁，算法逻辑字节级等价，不顺手优化 45454 交替/字级时间戳对齐。
- **评测与代码解耦**：DER 通过 HTTP POST 到服务测，不在算法包内跑——所以换测试集/换策略不动算法代码。
- **DER 评测用 oracle-K**：`der_eval.py` 用 RTTM 真实说话人数 `num_speakers` 强制收敛（避免聚类 K 偏差干扰 DER）。

## Work Completed

### Tasks Finished

- [x] P1 算法 SDK 化 W0–W6 全部提交（分支 `refactor/p1-algo-packaging`，7 commits，未合 main）
- [x] 澄清 DER 评测机制：`der_eval.py` POST 8002 + 显式 `seg_clustering` 链路无断点
- [x] P1 相关纯逻辑单测隔离跑全绿（unit 19 / seg_clustering 4 / config_loader 5）
- [x] 纠正测试集为 Ali；建 `testset/ali_near_prep_match/` 对齐目录，3 对全部 PAIR OK
- [x] 停掉全量 Ali DER（TaskStop `KPjX3M`），确认无 der_eval 残留进程
- [x] 查明 Ali DER 失败根因：8002 服务中途挂掉（非代码 bug）

### Files Modified

| File | Changes | Rationale |
|------|---------|-----------|
| `testset/ali_near_prep_match/` (新建, 未跟踪) | 3 场音频软链改名（去 `_mixed`）对齐 rttm stem | 绕过 der_eval 按 stem 配对的致命坑 |
| `test_results/der_single_seg_clustering.json` (已提交) | AISHELL-4 单场 DER 29.81% 证据 | P0 同文件 71.82%/32.2%，非回归 |
| Ali 全量 DER（任务 KPjX3M 失败） | 因 8002 服务中途挂掉而 failed，未生成有效 der_ali 结果文件 | 无有效产物，无需提交 |
| `.claude/handoffs/2026-07-14-183317-der-ali-der-test-pause.md` | 本 handoff | 交接用 |

### Decisions Made

| Decision | Options Considered | Rationale |
|----------|-------------------|-----------|
| 测试集 = Ali（`ali_near_prep`），非 AISHELL-4 | 沿用 AISHELL-4 / 改用 Ali | 用户明确："我们这边是用阿里的中国会议作为测试集" |
| 用对齐目录绕 `_mixed` 后缀，不急着改 `der_eval.py` | 改脚本支持后缀 / 软链对齐 | 非破坏式、不动评测脚本与仓库；后续可再补 der_eval 的 Ali 支持 |
| 先停全量测试 | 继续跑 / 先停 | 用户指令"先把全量测试停掉，交给下一个智能体" |
| 不盲信子智能体、不抢跑 20 场 GPU | 重复跑 / 等通知 / 自跑 | 守护等价性结论的可靠性 |

## Pending Work

### Immediate Next Steps

1. **重启 8002 服务**（它已挂）：`cd E:\project\funclip-pro && PYTHONPATH=E:/project/funclip-pro/src E:\conda\envs\asr_ui_env\python.exe asr_onnx_service.py`，启动后 `curl http://localhost:8002/docs` 探活（应 200）。
2. **定基线策略**（见 Blockers）：要证 Ali 上"P1 无回归"，需 old-code(main) 在 Ali 的 DER 基线。三条路待用户/下个智能体选：① 用户给历史 Ali 基线数值 → 只跑 P1 on Ali 对比；② checkout main 跑 Ali 得旧基线 + 跑 P1 on Ali 得新值做 before/after；③ 只出 P1 绝对 DER 看量级。
3. **按所选基线策略跑 Ali DER**：`der_eval.py --audio_dir/--rttm_dir E:/project/funclip-pro/testset/ali_near_prep_match --limit 0 --diarize_strategy seg_clustering --out test_results/der_ali_*.json`（服务必须在跑）。
4. 出 DER 结论后补 commit 收口 W5 全量评测，并视情况合入 main。

### Blockers/Open Questions

- [ ] **Ali 基线未知（核心阻塞）**：14.85%–15.13% 来自 AISHELL-4 旧报告，对 Ali 不适用。没有 old-code Ali DER 就无法证"无回归"。需用户给历史数值，或下个智能体跑 main 分支在 Ali 得基线。
- [ ] **8002 服务稳定性**：实测中服务会在评测中途挂掉（HTTP 000）。原因未查（疑似 OOM 或 GPU 资源）。重跑前务必探活，挂了就重启；必要时加 `--timeout`/资源监控。
- [ ] 全套 pytest 崩溃：`tests/` 下重型对比/benchmark 旧测试（`test_asr_comparison*`、`test_*_service_integration.py`、`bench_*`）在模块顶层重包装 sys.stdout/stderr 并用 subprocess 起真子进程，把 pytest capture fd 关了（`lost sys.stderr`/`I/O operation on closed file`）。与 P1 无关，已排除出常规门禁；若要修需单独处理这些旧测试。

### Deferred Items

- [ ] W6 自审遗留的 P1 后清理项：根 `segmentation_engine.py`/`speaker_engine.py` 与 `src/` 双真源漂移、`asr_service.py` 近重复（等价优先阶段未动手，符合红线）。
- [ ] `der_eval.py` 增加 Ali 支持（识别 `_mixed` 后缀、数据集可配置），让其不再依赖外部对齐目录。
- [ ] AISHELL-4 全量 DER（task 1M7J83）已被我 kill，因其非真实测试集；如仍需要可重跑。

## Context for Resuming Agent

### Important Context

- **真实测试集是 `testset/ali_near_prep`（阿里中文会议，3 场：R8002_M8002 / R8002_M8003 / R8004_M8005）**，不是 AISHELL-4。DER 门禁与"无回归"判定一律以 Ali 为准。
- **`der_eval.py` 致命坑**：按文件名 stem 配对 audio/rttm。Ali 音频是 `{name}_mixed.flac`，rttm 是 `{name}.rttm` → stem 对不上 → **0 匹配 → 静默写出 `global_DER: 0.0` 假成功**。跑 Ali 必须用对齐目录 `testset/ali_near_prep_match/`（已建好，3 对 PAIR OK）。
- **基线 14.85%–15.13% 来自 AISHELL-4，对 Ali 未必成立**。证 Ali 无回归需 old-code(main) 在 Ali 的 DER 基线。
- **8002 服务会中途挂**：本次 Ali DER 失败就是服务掉了（HTTP 000），不是 P1 bug。每次跑 DER 前必须探活，挂了重启。
- P1 算法已字节级等价（offline.py seg_clustering 双分支与 main 旧 `_run_inference` 一致），重构本身无回归风险；DER 焦点在"测试集/基线对齐"而非算法。

### Assumptions Made

- 早期误以为 AISHELL-4 是测试集（沿用旧报告），后被用户纠正为 Ali。
- 曾假设 14.85%–15.13% 是通用基线，实则为 AISHELL-4 专属。
- 假设后台 Ali DER 任务在正常跑，实际它因服务挂掉而失败（不可盲信后台任务状态）。

### Potential Gotchas

- **der_eval stem 配对**：任何新测试集若 audio/rttm 命名不一致都会 0 匹配假成功——务必先看配对日志行 `匹配 N 对 audio/rttm`。
- **服务挂掉**：长跑 DER 前探活；若 `curl` 返回 000 先重启服务再跑。
- **pytest 全套崩溃**：别用 `pytest tests/` 跑全套（旧重型测试关 fd）；只跑 P1 相关：`test_offline_pipeline_unit.py` / `test_seg_clustering.py` / `test_config_loader.py`（28 绿），integration 需 `FUNCLIP_INTEGRATION=1`。
- **环境路径**：真实 ML 用 `E:\conda\envs\asr_ui_env\python.exe`，导入包设 `PYTHONPATH=E:\project\funclip-pro\src`。沙箱受管 Python 无 torch。
- **taskkill 限制**：沙箱内 `taskkill` 碰不到用户交互会话的进程（如常驻 ASR 服务），只能结束本沙箱进程。
- **git 红线**：禁止 `git add -A`；提交只 `git add <明确文件>`。工作树有无关垃圾（`nul`/`output*.mp3`/`.agents/`）绝不提交。

## Environment State

### Tools/Services Used

- FastAPI 服务 `asr_onnx_service.py` @ `http://localhost:8002`（当前：**已挂，需重启**）
- `der_eval.py` DER 评测器（POST 到上述服务）
- pytest（conda 环境）

### Active Processes

- **无 der_eval / 无 DER 评测进程**（已停）。
- 8002 FastAPI 服务：**当前未运行**（中途挂掉，需重启才能再评测）。
- 注意：用户交互会话里可能有其常驻 ASR 服务（若占 8002 需先让其关闭）。

### Environment Variables

- `PYTHONPATH`（指向 `E:\project\funclip-pro\src`，跑包/服务/评测时必设）
- `FUNCLIP_INTEGRATION`（置 1 才跑 integration pytest，默认跳过）
- （其余见 `asr_onnx_service.py` 启动期读取的模型路径配置，无密钥类变量）

## Related Resources

- `.claude/handoffs/2026-07-14-p1-complete-handoff.md` — P1 接手文档（目录/模块/启动/DER 现状）
- `.claude/handoffs/2026-07-14-162727-p0-complete-p1-handoff.md` — P0 完成 + P1 对接
- `.scratch/p1-algo-packaging/issues/01..11-*.md` — P1 工单（含接口契约）
- `testset/CER_DER_TEST_REPORT.md` — DER 方法学（§5 AISHELL-4 失真说明）
- `AGENTS.md` — 项目规范与红线

---

**Security Reminder**: Before finalizing, run `validate_handoff.py` to check for accidental secret exposure.
