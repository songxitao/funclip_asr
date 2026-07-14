# Handoff: P0 全量完成 → P1 算法 SDK 化对接

> 本文件是 P0 收尾完成后的**接手对接文档**，供下一个智能体承接 P1。P0 全貌见根 `HANDOFF.md`；P1 最高指导技术说明书见 `.superpowers/spec/2026-07-14-refactor-p0-p1-spec.md`（不在此重复，按路径引用）。

## Session Metadata
- Created: 2026-07-14 16:27
- Project: E:\project\funclip-pro
- P0 分支: `refactor/p0-path-decoupling-pilot` — 已原子提交 `568e4f2`（待合 main）
- P1 建议分支: `refactor/p1-algo-packaging`（基于合入 main 后新开）
- Prepared for: 下一个智能体接手 P1 算法下沉

## Handoff Chain
- **Continues from**: [2026-07-14-152144-p0-path-decoupling-verified.md](./2026-07-14-152144-p0-path-decoupling-verified.md) — P0 试点快照，**已 Superseded（内容过时，仍写"服务层未解耦"），可删**。
- **主 handoff**: [HANDOFF.md](../HANDOFF.md)（根）— P0 全量完成态，含"分层真相"背景。

## 核心状态（下一个 agent 的 TL;DR）
- **P0 已 100% 完成并原子提交 `568e4f2`**：服务层（`asr_onnx_service.py` 6 处 `*_MODEL_DIR` + 顶部 DLL 裸补丁）+ 启动层（`app_control.py` 的 `CONDA_ROOT`/`OFFLINE_PYTHON`）全部消费 `config.loader`，**零绝对路径硬编码，达成 spec L21**。
- **精度门禁真实验证通过**：DER = **14.60%**（seg_clustering 策略，8003 独立端口真实 GPU 推理）≤ 基线 14.85%±0.06%，**PASS 无回归**。这是动态路径首次在推理中生效的证明（试点仅验证"不崩"）。
- **下一步 = P1（算法 SDK 化）**，spec L48-58 为最高指导。
- **红线不变**：算法逻辑与重构前完全等价（不优化 45454 交替 / 字级时间戳对齐）；numpy 锁 1.26.4；时间戳 API/评测用 ms；`_run_inference` 返回四元组；`powerset.cpu()` 在 `to_multilabel` 前调用；DLL 补丁保活（`apply_dll_patch()`）。

## P0 已完成（证据，不重复 diff）
- 提交 `568e4f2`（7 files, +276/-103）：`asr_onnx_service.py` / `app_control.py` / `config.yaml` / `src/funclip_pro/config/loader.py` / `tests/test_service_paths.py` / `tests/test_app_control_env.py` / `HANDOFF.md`。
- 复用的 loader 能力：`resolve_model_path()` / `apply_dll_patch()` / `load_config()` / `PROJECT_ROOT`。
- 测试：`tests/` 下两套路径解耦测试 **6/6 passed**（沙箱受管 Python 可跑，不依赖 torch）。
- 宿主一键复跑门禁脚本：`run_p0_der_gate.py`（在 WorkBuddy 工作区，内存改 8003 端口/URL，不碰源码）。

## P1 任务（来自 spec L48-58，此处只给映射与边界，细节见 spec）
目标：把根目录算法下沉进 `src/funclip_pro/*`，FastAPI 变薄路由，实现"薄路由厚算法"。`src/funclip_pro` 当前仅有 `config` 子包，是 P1 落点骨架；需新建 `core/` `utils/` `pipeline/` 子包。

### 建议拆包映射（根 .py → src/funclip_pro 模块）
| 根文件 | 当前职责 | 下沉目标（建议） |
|--------|---------|------------------|
| `asr_onnx_service.py` | FastAPI + SenseVoiceONNX 推理类 + 子句说话人分配对齐(锚点扩散) + SRT 转换 + 段内合并 | 推理类→`core.asr`；对齐→`core.alignment`；SRT/合并→`utils.srt`；FastAPI→薄路由，只实例化 `OfflinePipeline` |
| `torch_engine.py` | torch 版 SenseVoice 推理 | `core.asr`（或 `core.torch`） |
| `sherpa_engine.py` | sherpa 离线识别（`SHERPA_MODEL_DIR`） | `core.asr`（或 `core.sherpa`） |
| `segmentation_engine.py`（P0 已解耦路径） | seg-3.0 分割引擎 | `core.segmentation` |
| `speaker_engine.py`（P0 已解耦路径） | Cam++ 说话人引擎 | `core.speaker` |
| `asr_service.py` | **待确认**（根目录已存在，可能已有 ASR 封装） | 先读内容，决定并入 `core.asr` 或保留复用 |
| `app_control.py`（P0 已解耦 env） | 启动/环境管理 | `utils/app` 或 CLI 入口 |
| `cli_transcribe.py` | CLI 客户端 | 薄客户端，`import` 核心包 |
| （新增）`OfflinePipeline` | —— | `pipeline.offline`（整合统一转写流水线，对外暴露统一步骤管理器） |

### 明确不沉（排除范围）
- 评测脚本：`ali_der_eval.py` / `der_eval.py` / `cer_eval*.py` / `run_ali_der_full.py`
- 实时流式（属 P3）：`app_live_local.py` / `app_live_ws.py`
- 数据准备/工具：`ali_near_prep.py` / `extract_aishell1_test.py` / `merge_srt_to_ass.py`
- Gradio UI 清理（属 P3）；算法优化（45454 交替、字级时间戳对齐）属 Out of Scope

### P1 测试门禁（spec Testing Decisions）
- pytest 全套（ASR API 联调、分割集成）**无缓存状态下全部通过**。
- DER `seg_clustering`：单场 + **全量 20 场**，与重构前 14.85%–15.13% 相比**无实质精度下滑**。
- 测试只校验外部行为（结构化转写、SRT 合法、DER 对齐），不关心 DLL 补丁/相对路径内部实现。

### P1 红线（等价优先）
- 算法逻辑与重构前完全等价；不优化 45454 交替、字级时间戳对齐（Out of Scope）。
- 沿用 P0 已建 loader（`resolve_model_path`/`apply_dll_patch`），**不要回归硬编码盘符**。
- 局部 skill：`implement`（TDD + 末尾 code-review + 提交当前分支）/ `tdd` / `code-review` / `handoff` / `to-spec`。

## Context for Resuming Agent（接手必读）
1. 先读 `.superpowers/spec/2026-07-14-refactor-p0-p1-spec.md`（最高指导技术说明书）。
2. 再读本 HANDOFF.md 与根 `HANDOFF.md` 的 P0 背景（理解"服务层也曾硬编码、已收尾"）。
3. 真实 ML 环境：`E:\conda\envs\asr_ui_env\python.exe`（torch 2.3.1+cu121，CUDA 可用）。**沙箱受管 Python 无 torch/pytest**，跑任何 ML 测试/服务必须用宿主 env。
4. DER 评测**必须显式 `seg_clustering`**，否则默认口径得 49%–57% 错误高值，误判回归。
5. git 红线：**禁用 `git add -A`**（会吞 `nul`/`output*.mp3`/`.agents/` 等垃圾）；显式 `git add` 目标文件。提交信息用中文（AGENTS.md）。
6. 分支策略：先合 `refactor/p0-path-decoupling-pilot` → `main`，再基于 main 开 `refactor/p1-algo-packaging`。

## Environment State
- conda env：`E:/conda/envs/asr_ui_env/python.exe`（torch 2.3.1+cu121 / pytest 9.1.1 / pyannote / funasr 1.2.7 / pyyaml 6.0.3）
- 模型权重：`E:\project\funclip-pro\model\models\damo\...`（本地）
- **未跟踪垃圾（勿提交）**：`nul`、`output.mp3`、`output1.mp3`、`.agents/`、`README_zh.md`、`REVIEW-sliding-diarization.md`、`.claude/handoffs/2026-07-13-*` 及过时的 `2026-07-14-152144-*` 副本。

## Suggested Skills（下一个智能体应调用）
- `.agents/skills/to-spec` — P1 若需新增/修订 spec 时遵循
- `.agents/skills/implement` — 实现下沉：TDD + 末尾 code-review + 提交当前分支
- `.agents/skills/tdd` — 预对齐 seam 再写测试
- `.agents/skills/code-review` — Standards/Spec 双轴自审
- `.agents/skills/handoff` — P1 完成后再写接手 handoff

## Related Resources
- 重构 spec：`.superpowers/spec/2026-07-14-refactor-p0-p1-spec.md`
- P0 主 handoff：根 `HANDOFF.md`
- 前序 handoff（已 Superseded）：`.claude/handoffs/2026-07-14-152144-p0-path-decoupling-verified.md`
- 项目规约：`AGENTS.md`
- 局部 skill：`.agents/skills/{to-spec,implement,tdd,code-review,handoff}`
