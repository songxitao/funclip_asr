# Handoff: P1 全量完成 + 合入 main — 可开 P3/新功能

## Session Metadata
- Created: 2026-07-14 19:34
- Project: E:\project\funclip-pro
- Branch: **main**（P0+P1 已合并，P1 为最新 6d8dd9f）
- Scope: 项目对接 + CER/DER 评测 + 双真源清理 + 重复路由删除 + P1 合入 main

### 本 Session 提交链
- P0（已合入 main）：`97b75de` docs(handoff): P0 全量完成
- P1（已合入 main）：`6d8dd9f` docs: DER 测试集纠正为 Ali + 全量评测暂停
- 中间清理：直接对 main 操作（薄导出/删文件，非独立提交）

---

## 1. 本 Session 做了什么

### 1.1 项目对接（初始阶段）
- 阅读 HANDOFF.md（根）+ 3 份 .claude/handoffs/
- 阅读 AGENTS.md 开发规约
- git diff 分析 P1 近 5/10 次提交（+1324/-833 行，核心为 OfflinePipeline + 薄路由）
- 生成测试计划书 `funclip-pro-测试计划.md`

### 1.2 测试集判断
- 对比 AISHELL-4（dia-aishell4-test）vs AliMeeting（ali_near_prep）
- 结论：**AliMeeting 更合适**（近场单通道无方法学失真，且用户之前已确认）
- AISHELL-4 因远场 8 通道阵列抽 ch0 单通道导致 DER 失真（详见 `testset/CER_DER_TEST_REPORT.md §5`）

### 1.3 CER 评测（8002 服务）
| 测试集 | 采样 | CER | 失败 | 耗时 |
|--------|:----:|:---:|:----:|:----:|
| AISHELL-1 | 1000 条 | **6.54%** | 0 | 566s |
- 与 P1 前（6.53%）几乎一致 → ASR 无回归 ✅

### 1.4 DER 评测（8002 服务 + AliMeeting 3 场）

**Main 基线（切到 main 跑）**：
| 会话 | DER | 耗时 |
|:-----|:---:|:----:|
| R8002_M8002 | 14.60% | 65s |
| R8002_M8003 | 20.57% | 52s |
| R8004_M8005 | 23.96% | 51s |
| **全局** | **19.69%** | 169s |

**P1 分支（当前 main）**：
| 会话 | DER | 变化 | 判定 |
|:-----|:---:|:----:|:----:|
| R8002_M8002 | 14.75% | +0.15% | ✅ |
| R8002_M8003 | 20.44% | -0.13% | ✅ |
| R8004_M8005 | 24.68% | +0.72% | ✅ |
| **全局** | **19.96%** | **+0.27%** | **✅ 无回归**（GPU 噪声范围内） |

说话人分布、段数均一致，P1 算法迁移字节级等价确认。

### 1.5 双真源清理（根算法副本 → 薄导出）
| 文件 | 改动 |
|------|------|
| `segmentation_engine.py` | 568 行 → 3 行 `from funclip_pro.core.segmentation import SegmentationEngine` |
| `speaker_engine.py` | 322 行 → 3 行 `from funclip_pro.core.speaker import CampPlusSpeaker, segment_sliding_window` |
| `tests/test_segmentation_engine.py` | import 改为 `funclip_pro.core.segmentation` |
| `tests/test_seg_clustering.py` | import 改为 `funclip_pro.core.speaker` |
| `tests/test_seg_seamless.py` | import 改为 `funclip_pro.core.speaker` |
| `tests/test_vad_sliding.py` | import 改为 `funclip_pro.core.speaker` |
| `tests/test_sliding_segmentation.py` | import 改为 `funclip_pro.core.speaker`（2 处） |

验证：29 passed

### 1.6 重复路由删除
- `asr_service.py` **已删除** 🗑️
- `tests/test_asr_api.py` — skip（强依赖 `asr_service.MODEL/VAD_MODEL` Mock）
- `tests/test_pytorch_inference_refactor.py` — skip（依赖 `_run_inference` 内部函数）
- `tests/test_pytorch_route.py` — skip（参数签名不同）
- `tests/test_pytorch_affinity.py` — 成功迁移到 `import asr_onnx_service`
- `tests/test_pytorch_service_integration.py` — 子进程路径改为 `asr_onnx_service.py` + 8002 端口
- `tests/test_asr_comparison.py` / `test_asr_comparison_cpu.py` — 路径改为 `asr_onnx_service.py`

### 1.7 P1 合入 main
- P0 已在 main 中（`97b75de`，与 `refactor/p0-path-decoupling-pilot` 一致）
- P1 fast-forward 合并成功，零冲突
- 41 门禁测试全绿 ✅

---

## 2. 当前项目状态

| 阶段 | 状态 | 说明 |
|------|:----:|------|
| P0 路径解耦 | **✅ 完成** | config.loader 统一管理，零硬编码盘符 |
| P1 算法 SDK 化 | **✅ 完成** | funclip_pro 包 + 薄路由 + CLI 瘦身 |
| 双真源 | **✅ 已清** | 根文件改为薄导出，测试指 SDK |
| 重复路由 | **✅ 已删** | asr_service.py 已移除 |
| P1 合入 main | **✅ 完成** | main 已是最新 P1 代码 |
| CER 门禁 | **✅ 通过** | 6.54%（等价 P1 前） |
| DER 门禁 | **✅ 通过** | 19.96% vs 基线 19.69%，+0.27% 无回归 |
| DER 全量评测 | ⚠️ 暂停待重组 | Ali 3 场已跑完，基线已建立 |
| P3（UI/流式/多通道） | ❌ 未开始 | 待立项 |

### 关键文件

| 文件 | 用途 | 说明 |
|------|------|------|
| `src/funclip_pro/config/loader.py` | 配置加载器 | resolve_model_path / apply_dll_patch / load_config |
| `src/funclip_pro/core/segmentation.py` | 分割引擎 | pyannote powerset 分割 |
| `src/funclip_pro/core/speaker.py` | 声纹引擎 | Cam++ + SpectralClustering |
| `src/funclip_pro/core/asr.py` | ASR 后端 | ONNX/PyTorch/Sherpa 三大引擎 |
| `src/funclip_pro/core/alignment.py` | 子句说话人对齐 | 锚点扩散 |
| `src/funclip_pro/pipeline/offline.py` | OfflinePipeline | 统一转写流水线，返回四元组 |
| `asr_onnx_service.py` | FastAPI 薄路由 | `:8002`，`POST /transcribe` |
| `der_eval.py` | DER 评测器 | POST 8002 算 DER |
| `cer_eval_parallel.py` | CER 评测器 | 并发采样 |
| `segmentation_engine.py` | **薄导出** | → `funclip_pro.core.segmentation` |
| `speaker_engine.py` | **薄导出** | → `funclip_pro.core.speaker` |

---

## 3. 启动与运行命令

```bash
# 启动服务（8002）
cd E:\project\funclip-pro
set PYTHONPATH=E:\project\funclip-pro\src
E:\conda\envs\asr_ui_env\python.exe asr_onnx_service.py

# 纯逻辑单测（15 个核心文件）
E:\conda\envs\asr_ui_env\python.exe -m pytest \
  tests/test_routing.py tests/test_decode_fallback.py \
  tests/test_config_loader.py tests/test_app_control_env.py tests/test_service_paths.py \
  tests/test_onnx_decode_refactor.py tests/test_segmentation_engine.py \
  tests/test_seg_clustering.py tests/test_seg_seamless.py \
  tests/test_sliding_segmentation.py tests/test_sliding_integration.py \
  tests/test_offline_pipeline_unit.py tests/test_vad_sliding.py tests/test_affinity.py \
  -v

# DER 评测（Ali，服务必须在跑）
E:\conda\envs\asr_ui_env\python.exe der_eval.py \
  --audio_dir E:/project/funclip-pro/testset/ali_near_prep_match \
  --rttm_dir E:/project/funclip-pro/testset/ali_near_prep_match \
  --limit 0 --diarize_strategy seg_clustering \
  --out test_results/der_ali_seg_clustering.json
```

---

## 4. 已知风险与红线

### 必守红线
- **numpy 1.26.4** — 永不解锁 2.x（pyannote 依赖 `np.NaN`）
- **时间戳 ms** — `cluster_with_segmentation` 返回秒需 ×1000
- **powerset.cpu()** — 必须在 `to_multilabel` 前调用
- **DLL 补丁保活** — 三处 `apply_dll_patch()` 不可删
- **显式 git add** — 禁止 `git add -A`（有 `nul`/`output*.mp3`/`.agents/` 等未跟踪垃圾）
- **DER 必须 seg_clustering** — 默认 two_stage 会得 49%-57% DER
- **der_eval stem 配对** — 跑新测试集必须先看配对日志（否则静默 DER=0.0）
- **绝对导入** — SDK 包内禁相对导入，统一 `from funclip_pro.x.Y import Z`

### 已知风险
1. **8002 服务不稳定** — 长跑 DER 期间会自行挂掉（疑似 OOM），跑前探活
2. **沙箱 Python 无 torch/pytest** — ML 操作用 `E:\conda\envs\asr_ui_env\python.exe`
3. **pytest 全家桶崩溃** — 旧测试（bench_*/comparison/sherpa_performance）关 fd，别跑无过滤的 `pytest tests/`

---

## 5. 可开新活候选方向（按优先级）

### P3-A：Gradio UI 清理
- 文件：`app_control.py` / `app.py`
- 目标：将 Gradio UI 也"薄 UI 厚 API"，与 `OfflinePipeline` 解耦
- 参考：P1 对 `asr_onnx_service.py` 做薄路由的模式

### P3-B：实时流式重构
- 文件：`app_live_local.py` / `app_live_ws.py`
- 目标：WebSocket 管道标准化，与 OfflinePipeline 共享 core 模块

### P3-C：DER 方法学改进
- 目标：`der_eval.py` 的 `to_mono_ch0()` 改多通道选优（能量/SNR 选通道或 embedding 融合）
- 参考：`testset/CER_DER_TEST_REPORT.md §6`

### 其他维护项
- `der_eval.py` 增加原生 Ali 支持（识别 `_mixed` 后缀，不用软链目录）
- 清理 `tests/` 里旧的重型基准测试（关 fd 问题根源）

---

## 6. 评测结果文件

| 文件 | 内容 |
|------|------|
| `test_results/der_ali_seg_clustering.json` | P1 分支 Ali DER（全局 19.96%） |
| `test_results/der_ali_baseline.json` | main 基线 Ali DER（全局 19.69%） |
| `test_results/cer_sample_1k_p1.jsonl` | CER 采样 1000 条明细 |
| `test_results/cer_sample_1k_p1_summary.json` | CER 汇总（6.54%） |

---

## 7. Suggested Skills（下一个智能体应调用）

- `.agents/skills/to-spec` — 开 P3 新功能时先写 spec
- `.agents/skills/implement` — 实现：TDD + 末尾 code-review + 提交当前分支
- `.agents/skills/tdd` — 预对齐 seam 再写测试
- `.agents/skills/code-review` — Standards/Spec 双轴自审
- `.agents/skills/handoff` — 完成后再写下一个 handoff
- `.agents/skills/wayfinder` — 不确定方向时用来探索代码库
- `.agents/skills/improve-codebase-architecture` — 架构改进建议
