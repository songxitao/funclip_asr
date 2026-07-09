# Handoff: funclip-pro ASR 推理服务 — Sherpa-ONNX 接入与自适应管线设计

## 0. 速读 (TL;DR)
- **项目**：`E:\project\funclip-pro` —— FastAPI 实时字幕 ASR 服务（SoundVoiceSmall 后端）。
- **已完成**：Sherpa-ONNX INT8 引擎已接入 `asr_onnx_service.py`，单测 + 集成测试 3 项全绿；双引擎（Sherpa-CPU vs PyTorch-GPU）评测已完成，结论有数据支撑。
- **本次会话目标（用户已确认「执行一下吧」）**：把 `vad_split` 布尔升级为三态 `vad_strategy=auto|always|never` + 廉价 trim + 引擎自动路由 + 生成 Dify 自定义工具 OpenAPI schema。
- **两个关键纠正**（防止接手人再踩）：① `top_db` 越小越激进（之前说反过）；② 标点不能跳过 PUNC，必须「剥原生标点 → 拼全文 → PUNC 一次」。

---

## 1. 环境与运行
- **Python 环境**：`E:\conda\envs\asr_ui_env`
- **GPU**：RTX 4080 Laptop 12GB，`torch 2.3.1+cu121`，`torch.cuda.is_available() == True`
- **依赖**：`sherpa-onnx==1.13.4`、`funasr`、`librosa`、`fastapi`、`uvicorn`
- **服务**：端口 `8002`，端点 `POST /transcribe`（multipart 音频 + 可选 `vad_split`/`engine` 参数），返回 `application/json` `{"text": "..."}`
- **测试音频**：`E:\下载\下载\李雪花2.wav`（424.5s，85 段，真实长音频）

---

## 2. ASR 设计（架构与决策）

### 2.1 总体管线
```
音频 ──► [FSMN VAD 切分 | 或 短音频直接解码]
        ──► ASR 引擎: Sherpa-ONNX(CPU) 或 PyTorch(GPU)
        ──► 剥 <|...|> 标签 + 剥原生标点(保留 ITN 数字)
        ──► "\n".join 拼成全文
        ──► PUNC_MODEL.generate(全文)  ← 对全文只跑一次
        ──► JSON {"text": 带标点文本}
```

### 2.2 引擎选择（CPU 部署 vs GPU 可用）
- **CPU 部署主选 Sherpa-ONNX INT8**：RTF 0.026、零显存。理由：ORT/OpenVINO 在动态 shape + 零散量化下严重退化（OpenVINO 集成已判为死路，见 Gotchas）。
- **GPU 可用时 PyTorch FP32 更快**：总耗时 5.34s vs Sherpa-CPU 10.92s（快 2.045x），但独占 ~945 MiB 显存。
- **决策**：引擎选择是「资源调度决策」，放服务端自动路由，**不要让 Dify 选**。Dify 节点只传音频 + 可选 `engine=gpu|cpu` 覆盖。

### 2.3 VAD 策略（核心设计点，用户已确认方向）
- **FSMN VAD 现状**：`max_gap_ms=300`、`max_duration_ms=8000`，合并 ≤8s 段后批量 `decode_streams`。
- **短音频痛点（用户洞察）**：3s 短语音跑完整 VAD 流程额外 ~0.2–0.5s，端到端延迟差 ~1.6x；但整段本来就是一句话，切分无收益。**短音频跳过 VAD 切分能明显降延迟、提体感。**
- **廉价 trim（关键概念）**：`librosa.effects.trim` 用能量阈值法，**不加载任何模型**，复杂度 O(n)，对短音频 ~0 成本。它只切「首尾静音」（防幻觉），VAD 还会切「句内停顿」——短音频不需要后者。
- **决定**：把 `vad_split`(bool) 升级为 `vad_strategy=auto|always|never`（详见 §4）。

### 2.4 标点策略（关键纠正，已验证落地）
- ❌ **旧错误想法**：「Sherpa 原生带标点 → PUNC 冗余可跳过」。质量层面不成立——逐段标点在 VAD 短片段边界不可靠。
- ✅ **正确做法（已写入代码）**：每条 chunk → 剥 `<|...|>` 标签 → 剥原生标点（**保留 ITN 数字**）→ `join` 拼全文 → `PUNC_MODEL.generate` 一次。三条路径（Sherpa / 旧 OpenVINO / PyTorch）统一为「剥标点 → 拼全文 → PUNC 一次」。
- **验证数据**：Sherpa 逐段原生标点 174 个 vs 全文 PUNC 170 个，仅差 4；短音频/被切断句子上差异更大，故保留全文 PUNC 最稳。

---

## 3. 已完成（含证据）

| 工作 | 证据 |
|---|---|
| `SherpaSenseVoice` 引擎类 | `sherpa_engine.py`（封装 `from_sense_voice` + `decode_streams`，CPU `num_threads=6`）|
| 接入 `asr_onnx_service.py` | `MODEL` 改为 `SherpaSenseVoice` 指向 `SenseVoiceSmallOnnx`；`_run_inference` 两分支统一「标签清洗 + 剥标点 → 拼全文 → PUNC 一次」|
| 测试全绿 | `tests/test_sherpa_engine.py`（单元）+ `tests/test_onnx_service_integration.py`（端口 8002，POST 李雪花2.wav 断言带标点 JSON）→ **3 passed** |
| 双引擎评测 | `tests/bench_sherpa_vs_pytorch.py` + `tests/bench_report.json` |

**评测数字（李雪花2.wav，424.5s，85 段）**：

| 指标 | Sherpa-ONNX (CPU) | PyTorch (GPU) |
|---|---|---|
| ASR 耗时 | 10.36s | 4.92s |
| PUNC 耗时 | 0.56s | 0.41s |
| **总计** | **10.92s** | **5.34s** |
| RTF | 0.0257 | 0.0126 |
| 显存峰值 | 0 MiB | 945 MiB |
| 去标点文本长度 | 1691 字 | 1691 字 |
| **CER（去标点对比）** | **3.02%** | （差 ~51 字符，典型 INT8 vs FP32）|
| 最终标点数 | 170 | 168 |
| 加速比 | — | **2.045x** |

- **意外发现**：Sherpa-INT8 个别词反而更准（如「愿意**显摆**」vs PyTorch GPU 幻觉「愿意**嫌买**」），量化没让它变笨。
- 方案文档：`docs/superpowers/plans/2026-07-09-sherpa-onnx-service-integration.md`（已含 Step 8 标点更正）。

---

## 4. 待办（下次会话目标 — 用户已确认「执行一下吧」）

详细 spec 见上方方案文档。要点：
1. ✅ **`vad_split`(bool) → `vad_strategy=auto|always|never`**（已完成）
   - `auto`（默认）：`duration <= SHORT_AUDIO_MS` → 廉价 trim → 直接整段解码；否则 → 完整 FSMN VAD 切分。
   - `always`：永远完整 FSMN VAD。
   - `never`：不做 VAD 切分，但仍做廉价 trim（防幻觉最低保障）。
2. ✅ **新增 `_cheap_trim(audio_path, top_db=40, pad_ms=100)`**：用 `librosa.effects.trim`，**不加载任何模型**；返回 `(trimmed_waveform, full_duration_ms)`。
3. ✅ **引擎自动路由**：`CUDA 可用` 且 `音频够长(>SHORT_AUDIO_MS)` → PyTorch-GPU（惰性加载，锁内构建）；否则 → Sherpa-CPU。保留 `engine=gpu|cpu` 覆盖参数；torch 分支失败**自动回退 Sherpa**，保证端点始终返回文本。
4. ✅ **Dify 自定义工具 OpenAPI schema 文件**：`dify_openapi.yaml`（单端点 `POST /transcribe`，multipart 音频 + 可选 `engine`/`vad_strategy`，response `{text}`），可直接导入 Dify。
5. ✅ **默认值（用户确认）**：`SHORT_AUDIO_MS = 5000`，`SHORT_TRIM_TOP_DB = 40`，`pad_ms = 100`。
6. ✅ 已完成 `git commit`。

---

## 5. top_db 概念纠正（重要，防再犯）
- `top_db` 是**相对峰值的分贝阈值**（不是 topk 那种「取前 k 个」计数）。判定：`保留帧 ⟺ RMS(frame) ≥ 峰值 / 10^(top_db/10)`。
- **越小 → 分母越小 → 门槛越高 → 砍得越多 = 越激进**（之前说反过，已更正）。
- 参考：`10`=很激进（轻语音被切）；`40`=推荐中间值；`60`=librosa 默认（保守）。
- 分贝是对数比值：`30dB ≈ 差 1000 倍`，`60dB ≈ 差 100 万倍`。

---

## 6. Gotchas（必读）
- ⚠️ **Sherpa `model.int8.onnx` 是 CPU-ONLY**。永远别送 CUDA/ORT-GPU：动态量化反量化拖垮 GPU（32.33s vs PyTorch FP32 GPU 7.74s，慢 4.17x）。
- ⚠️ `accept_waveform` 要求切片 ≥ **1600 采样点(0.1s)**，否则报错；`sherpa_engine.py` 已硬性过滤。
- ⚠️ **OpenVINO 集成是死路**：CPU 跑评 30.96s，比 PyTorch FP32 基准 16.19s 还慢，已 commit 但判废。
- ⚠️ **`PUNC_MODEL` 绝不能移除**——它是全文标点质量的把关，三条路径共用。
- Sherpa 输出是**干净文本、无 `<|...|>` 标签**；PyTorch 原始输出含标签需 `re.sub(r"<|.*?|>", "", t)` 清洗。
- GPU 显存峰值约 945 MiB，部署时预留。

---

## 7. 文件地图

| 文件 | 作用 |
|---|---|
| `sherpa_engine.py` | `SherpaSenseVoice` 引擎封装（CPU INT8）|
| `asr_onnx_service.py` | FastAPI 服务（三态 vad_strategy + 引擎自动路由 + _cheap_trim）|
| `torch_engine.py` | `PyTorchSenseVoice` 引擎封装（funasr AutoModel，GPU 惰性加载）|
| `dify_openapi.yaml` | Dify 自定义工具 OpenAPI schema（POST /transcribe）|
| `tests/test_routing.py` | 路由/VAD 策略/cheap_trim/strip_punctuation 单测 |
| `tests/test_torch_engine.py` | PyTorch 引擎接口测试（模型目录缺失时跳过）|
| `tests/test_sherpa_engine.py` | 引擎单测 |
| `tests/test_onnx_service_integration.py` | 端口 8002 集成测试（李雪花2.wav，断言带标点 JSON）|
| `tests/test_decode_fallback.py` | `_decode` 引擎路由 + PyTorch→Sherpa 失败回退测试（monkeypatch 假引擎，不加载模型，3 passed）|
| `tests/bench_sherpa_vs_pytorch.py` | 双引擎评测脚本 |
| `tests/bench_report.json` | 评测结果（§3 数字来源）|
| `tests/verify_punc_sources.py` | 标点来源验证（Sherpa vs PyTorch 原生输出对比）|
| `docs/superpowers/plans/2026-07-09-sherpa-onnx-service-integration.md` | 实施计划（含 Step 8 标点更正）|
| `model/models/iic/SenseVoiceSmallOnnx` | Sherpa 模型目录（`model.int8.onnx` + `tokens.txt`）|

---

## 8. 已验证 API（直接复用，勿改签名）
```python
# Sherpa 引擎（已验证可跑，不要传 language 参数）
sherpa_onnx.OfflineRecognizer.from_sense_voice(
    model=".../SenseVoiceSmallOnnx/model.int8.onnx",
    tokens=".../SenseVoiceSmallOnnx/tokens.txt",
    num_threads=6, use_itn=True,
)
# 封装类接口
engine = SherpaSenseVoice(model_dir=".../SenseVoiceSmallOnnx", num_threads=6, use_itn=True)
texts: list[str] = engine(audio_path_or_list_of_np_arrays)  # <1600 采样点自动跳过
```
- **复跑测试**：`E:\conda\envs\asr_ui_env\python.exe -m pytest tests/test_sherpa_engine.py tests/test_onnx_service_integration.py -v`

---

## 9. 恢复会话的验收标准
- [x] `auto` 策略：短音频(<5s)走 trim 直解，长音频走 FSMN VAD。
- [x] CUDA 可用时自动走 PyTorch-GPU，否则 Sherpa-CPU；`engine` 参数可强制覆盖。
- [x] `POST /transcribe` 仍返回带标点 JSON（并新增 `engine` 字段）。
- [x] 现有 3 项测试 + 新增测试全部全绿（共 13 项：test_routing 6 + test_sherpa_engine 2 + test_torch_engine 1 + test_decode_fallback 3 + test_onnx_service_integration 1，详见 §10.2）。
- [x] 生成 Dify OpenAPI schema 文件，可直接导入 Dify 自定义工具。

---
## 10. 晚场收尾工作（2026-07-09 22:08 追加 — 验证 + 提交拆分 + 回退测试）

本段由 agent 在子智能体完成 §4 后，独立「trust but verify」补做，目的是把之前未坐实的环节闭合。

### 10.1 关键发现：测试环境不是系统 Python
- 之前一度以为「子智能体声称的 9 passed 复现不了」——根因是**找错了解释器**。
- **能跑测试的唯一环境是 conda venv**：`E:\conda\envs\asr_ui_env\python.exe`（Python 3.11.14，`funasr==1.2.7`、`sherpa-onnx==1.13.4`、`pytest==9.1.1` 都已装）。
- 系统 Python `D:\program files\python\python.exe` **缺 `funasr` / `sherpa_onnx`**，`import asr_onnx_service` 会直接因 `from funasr import AutoModel` 失败，collect 不起来。
- **唯一正确复跑命令**：
  ```bash
  E:\conda\envs\asr_ui_env\python.exe -m pytest tests/test_routing.py tests/test_sherpa_engine.py tests/test_torch_engine.py tests/test_decode_fallback.py -q
  ```
- ⚠️ 不要直接 Read 源码（用户硬约束）；读码一律走 `codegraph node <file>`。

### 10.2 独立验证结果（已坐实）
- 单测：**`test_routing` 6 + `test_sherpa_engine` 2 + `test_torch_engine` 1 + `test_decode_fallback` 3 = 12 passed**（约 10.4s）。
- 集成：`test_onnx_service_integration.py`（起 8002 服务，POST 李雪花2.wav，断言带标点 JSON）→ **1 passed**（约 24.6s）。
- `codegraph node asr_onnx_service.py` 逐行确认 §4 核心逻辑均在：
  - `_select_engine()`：cpu→sherpa / gpu→torch / auto→CUDA 可用且长音频走 torch 否则 sherpa。
  - `_use_vad(vad_strategy, duration_ms)`：always / never / auto 三态。
  - `_cheap_trim(audio_path, top_db=SHORT_TRIM_TOP_DB, pad_ms=TRIM_PAD_MS)`。
  - `_decode()` 第 327–337 行：`except Exception → 回退 Sherpa-CPU`，PyTorch→Sherpa 兜底分支**确实存在**（并在 10.3 被强制练到）。
  - `torch_engine.PyTorchSenseVoice` 惰性加载（锁内，防集成测试超时）。

### 10.3 补齐唯一缺口：PyTorch→Sherpa 回退测试
- `_decode` 的 `except → 回退 Sherpa` 分支此前**从没被强制触发过验证**。新增 `tests/test_decode_fallback.py`：
  - 用 `monkeypatch` 注入假引擎，**强制 torch 分支抛错**，断言仍回退 Sherpa 返回文本——`except` 兜底分支第一次被真正练到。
  - 覆盖三态：torch 成功 / torch 失败回退 / sherpa 直连。**3 passed**，**不加载任何模型**。

### 10.4 提交拆分（git 历史已重写，纯本地、安全）
- 原提交 `5ac6852` 用 `git add -A` 把**前会话交付物**（sherpa_engine.py / bench 脚本与报告 / verify_punc_sources.py / 集成计划 doc）也误并入同一提交。
- 已 `git reset --soft` 拆成两个独立提交，当前链：
  - `c7ca931` test: 新增 `_decode` 引擎路由与 PyTorch→Sherpa 回退测试
  - `ff5d193` feat: §4 vad_strategy 三态 + 引擎自动路由 + _cheap_trim + Dify OpenAPI schema
  - `9a3c432` chore: 前会话交付物（sherpa 引擎 / bench / punc 校验 / 集成计划）
  - `3242e0a` 父提交（原封不动）
- 无远程、无 upstream，改写历史不涉及 force push，安全。

### 10.5 已知小问题（非阻塞）
- `pytest tests/` 全目录一起跑时，集成测试起服务线程会触发 pytest 输出捕获冲突（`I/O on closed file`）——属 harness 隔离问题，**非代码缺陷**。拆开跑（单元集合 / 集成单独）即可全绿。

### 10.6 验收标准（更新）
- [x] 全部测试独立复跑全绿：单测 12 passed + 集成 1 passed。
- [x] PyTorch→Sherpa 回退兜底分支被测试强制练到（不再是「写了没验」）。
- [x] 提交历史干净：§4 功能、前会话交付物、回退测试三笔独立提交，互不混杂。
- [x] 测试必须走 conda 环境 `asr_ui_env`，已写入 §1/§10.1，接手人勿踩系统 Python 的坑。

---
*本 handoff 覆盖至 2026-07-09 22:08。对接窗口建议用 project-docking 读取本文件 + codegraph 同步符号，再用 writing-plans / executing-plans 推进后续（如：修复 §10.5 capture 冲突、给 Dify schema 配实际调用样例）。*
