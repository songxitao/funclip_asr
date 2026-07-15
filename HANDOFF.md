# Handoff: 重构遗留 Bug 整改全部完成 — P3.2 体验闭环

## Session Metadata
- Created: 2026-07-14 23:07
- Project: E:\project\funclip-pro
- Branch: main (工作区未提交)
- Session focus: P3.2 重构遗留的 5 个 Bug 修复与功能补全
- Worked by: WorkBuddy AI Agent (会话 ID: 2026-07-14-22-37-08)

### Recent Commits (for context)
  - `5316acd` refactor(p1.5/p2/p3.1): app_control 离线归口、冗余文件清理、pyproject.toml 构建标准化
  - `6d8dd9f` docs: DER 测试集纠正为 Ali + 全量评测暂停，写 handoff 交接下一智能体
  - `6ef1a58` docs(p1/w6): code-review 双轴自审 + 写 P1 接手 handoff
  - `6a8cae9` test(p1/w5): 测试门禁 — P1 相关单测 28 绿 + DER 单场 seg_clustering 等价 P0 非回归

---

## Handoff Chain
- **Continues from**: HANDOFF.md (2026-07-14 22:05, P3.2 流式重构完成)
- **Supersedes**: 上一个 HANDOFF.md（仅含架构重构，未含后续 Bug 修复）
- **Current session**: 2026-07-14 22:37–23:07 完成 5 个整改 Ticket（修复 Bug + 功能补全 + Qwen3 集成）

---

## Current State Summary

已完成 **P0 / P1 / P1.5 / P2 / P3.1 / P3.3 / P3.2** 架构重构，以及 **P3.2 后续 5 个整改 Ticket**。

### 本次会话完成状态

| # | Ticket | 文件变更 | 测试结果 |
|:-:|--------|---------|:-------:|
| 🅰️ | **01 — 音频泄露 + 混音增益** | `core/audio.py` + `tests/test_audio_stream.py` | ✅ 16 passed, 2 skp |
| 🅱️ | **02 — VAD 多会话污染 + 硬编码** | `core/streaming_asr.py` | ✅ 14 passed |
| 🅲 | **03 — 打字预览恢复** | `core/streaming_asr.py` + `app_live_local.py` | ✅ 与 🅰️ 共存 |
| 🅳 | **04 — Qwen3 Docker 引擎集成** | `core/asr.py` + `pipeline/offline.py` + `app_control.py` + `core/__init__.py` | ✅ 路由通过 |
| 🅴 | **05 — WebSocket 重塑 + 清理** | `app_live_ws.py` + **删除** `sherpa_engine.py` / `torch_engine.py` | ✅ 语法通过 |

### 全量回归测试: **30 passed, 2 skipped** 🟢
- `tests/test_audio_stream.py` — 16 passed / 2 skipped
- `tests/test_streaming_engine.py` — 14 passed

### GPU 测试环境
- 环境: `E:/conda/envs/asr_ui_env/python.exe` (torch 2.3.1+cu121, RTX 4080 Laptop GPU)
- 测试必须显式列出测试文件分批跑（`pytest tests/` 收集阶段卡死）

---

## 本次工作详情（WorkBuddy 2026-07-14 22:37–23:07）

### 🅰️ Ticket 01 — 修复音频流资源泄露与 MixedStream 混音增益丢失

**Spec 来源**: `.scratch/refactor-alignment/issues/01-fix-audio-leaks-and-gain.md`

**修改文件**:
- `src/funclip_pro/core/audio.py` — 两处变更

**变更 1 — PyAudio 句柄泄露修复**:
在 `LoopbackStream.stop()` 和 `MicStream.stop()` 中补充了 `self._pyaudio_instance.terminate()` + `del self._pyaudio_instance`，确保流销毁时 PortAudio 硬件句柄完全释放。

```python
def stop(self):
    try:
        self.stream.stop_stream()
        self.stream.close()
        if hasattr(self, "_pyaudio_instance"):
            self._pyaudio_instance.terminate()
            del self._pyaudio_instance
    except Exception:
        pass
```

**变更 2 — MixedStream 混音增益修复**:
`MixedStream.read()` 在 mic+loop 双源混音分支漏乘 `VOLUME_BOOST(3.0)`，修复为：
```python
if chunk_mic is not None and chunk_loop is not None:
    min_len = min(len(chunk_mic), len(chunk_loop))
    mixed = (chunk_mic[:min_len] + chunk_loop[:min_len]) / 2
    return np.clip(mixed * VOLUME_BOOST, -1.0, 1.0)
```

**测试修复**:
- `tests/test_audio_stream.py` — 更新 `test_mixed_stream_both_sources_average` 预期值以匹配修复后带 VOLUME_BOOST 的正确行为

---

### 🅱️ Ticket 02 — 修复 streaming_asr.py 多会话 VAD 污染与参数硬编码

**Spec 来源**: `.scratch/refactor-alignment/issues/02-fix-streaming-vad-concurrency.md`

**修改文件**: `src/funclip_pro/core/streaming_asr.py`

**变更 1 — Silero VAD Session 隔离**:
`_process_silero()` 改为每个 session 延迟初始化独立的 `SileroVAD` 实例，不再共享引擎级的 `self._vad_model`：
```python
if "_silero_vad" not in session:
    vad_path = str(resolve_model_path("models/silero_vad.onnx"))
    session["_silero_vad"] = SileroVAD(vad_path)

silero_vad = session["_silero_vad"]
speech_prob = silero_vad(chunk)
```
- 所有 `self._vad_model(...)` 替换为 `silero_vad(...)`
- 所有 `self._vad_model.reset_states()` 替换为 `silero_vad.reset_states()`

**变更 2 — language 硬编码移除**:
- `create_session()` 增加 `language: str = "auto"` 参数
- `_run_asr()` 接收 `session` 参数，从 `session.get("language", "auto")` 读取
- 三处 `_run_asr` 调用点全部传入 session

**变更 3 — FSMN VAD 灵敏度可配置**:
- `FsmnVadStreaming.__init__()` 增加 `silence_chunks_to_flush: int = 10` 构造参数

---

### 🅲 Ticket 03 — 恢复流式打字预览交互体验

**Spec 来源**: `.scratch/refactor-alignment/issues/03-restore-typing-preview.md`

**修改文件**:
- `src/funclip_pro/core/streaming_asr.py`
- `app_live_local.py`

**变更 1 — streaming_asr.py 新增预览逻辑**（`_process_silero` 说话分支内）:
- 新增 `session["_last_preview_time"]` 追踪
- 当 `is_speaking=True` 且 `time.time() - _last_preview_time > PREVIEW_MIN_INTERVAL` 时，对 buffer 拼接音频调用 `_run_asr()` 返回 `is_final=False` 的预览结果
- 预览不清空 buffer、不追加 history

**变更 2 — app_live_local.py GUI 对接**:
- `run_engine()` 主循环处理结果时检查 `seg.get("is_final", True)`
- `is_final=False` → `gui_queue.put({"hist": history, "curr": text + " 🟢", "typing": True})`

---

### 🅳 Ticket 04 — 包内集成 Qwen3 (Docker) 引擎与路由

**Spec 来源**: `.scratch/refactor-alignment/issues/04-integrate-qwen3-docker.md`

**修改文件**:
- `src/funclip_pro/core/asr.py` — **新增 QwenEngine 类 + parse_qwen_timestamps()**
- `src/funclip_pro/pipeline/offline.py` — **打通 qwen 引擎路由**
- `app_control.py` — **解禁 Qwen3/SeACo/Nano 硬编码阻断**
- `src/funclip_pro/core/__init__.py` — **导出 QwenEngine**

**变更 1 — QwenEngine 类**（参考旧版 `core/asr/qwen.py` 重塑）:
- 从 `config.json` 的 `qwen_server.host` 读取 API 端点，默认 `http://127.0.0.1:28000`
- 实现 `__call__(audio_path)` → 文本 + `transcribe()` → 完整结果（含 SRT 时间戳）
- 支持 Shared Volume 优化（复制音频到 Docker 共享卷）

**变更 2 — _select_engine 路由**:
- 新增 `"qwen"` / `"Qwen3"` / `"Qwen3 (Docker)"` → 引擎 key `"qwen"`

**变更 3 — OfflinePipeline run() 新增 qwen 分支**:
- 直接实例化 QwenEngine，调用 transcribe()，解析时间戳
- 提前 return 跳过 VAD/说话人分离流程（不适用于 Docker 引擎）

**变更 4 — app_control.py 解禁**:
- 删除了第 300-305 行 Qwen3 和 FunASR 非 SenseVoice 模式的 `yield "❌ 错误..."` 阻断
- 透传 `engine=engine` 参数给 `pipeline.run()`

---

### 🅴 Ticket 05 — 重塑 WebSocket 实时字幕客户端并物理清理残留

**Spec 来源**: `.scratch/refactor-alignment/issues/05-rebuild-websocket-client.md`

**修改文件**:
- `app_live_ws.py` — **重塑，-35% 体积（38958→25248 字节）**
- **物理删除**: `sherpa_engine.py` ✅ / `torch_engine.py` ✅

**变更 1 — app_live_ws.py 去重**:
删除以下与 `funclip_pro.core` 重复的代码：
- 自有 `SileroVAD` 类（54 行）
- 自有 `BaseStream / LoopbackStream / MicStream / MixedStream` 类（136 行）
- 自有 PyAudio 导入块 + `HAS_LOOPBACK` 标记
- `import onnxruntime`（仅被已删除 SileroVAD 使用）

替换为：
```python
from funclip_pro.core.audio import LoopbackStream, MicStream, MixedStream
from funclip_pro.core.streaming_asr import SileroVAD
```

**变更 2 — `_capture_thread` 简化**:
从手动累积 500ms + 重采样/声道转换改为调用 `self.stream.read()`（由 `core.audio` 统一处理），外部累积到 `CHUNK_SAMPLES` 后送入 ASR 队列。

**保留不变**: `SubtitleSegmenter`、`SubtitleOverlay` GUI、`ASRClient` WebSocket 通信 + VAD 状态机、入口 `__main__`

---

## 文件变更汇总（本次会话）

### 新增
| 文件 | 说明 |
|------|------|
| `.scratch/refactor-alignment/issues/01-fix-audio-leaks-and-gain.md` | Ticket 01 Spec |
| `.scratch/refactor-alignment/issues/02-fix-streaming-vad-concurrency.md` | Ticket 02 Spec |
| `.scratch/refactor-alignment/issues/03-restore-typing-preview.md` | Ticket 03 Spec |
| `.scratch/refactor-alignment/issues/04-integrate-qwen3-docker.md` | Ticket 04 Spec |
| `.scratch/refactor-alignment/issues/05-rebuild-websocket-client.md` | Ticket 05 Spec |

### 修改
| 文件 | 变更摘要 |
|------|---------|
| `src/funclip_pro/core/audio.py` | stop() 补 terminate() + MixedStream 混音加 VOLUME_BOOST |
| `src/funclip_pro/core/streaming_asr.py` | VAD session 隔离 + language 参数化 + 打字预览 + FSMN 灵敏度可配 |
| `app_live_local.py` | 对接 typing preview GUI 显示 |
| `app_live_ws.py` | 剥离手写音频采集，对接 core.audio |
| `src/funclip_pro/core/asr.py` | 新增 QwenEngine + parse_qwen_timestamps + _select_engine 路由 |
| `src/funclip_pro/pipeline/offline.py` | OfflinePipeline.run() 新增 qwen 引擎分支 |
| `src/funclip_pro/core/__init__.py` | 导出 QwenEngine + parse_qwen_timestamps |
| `app_control.py` | 删除 Qwen3/SeACo/Nano 阻断，透传 engine 参数 |
| `tests/test_audio_stream.py` | 更新混音测试预期值匹配修复后行为 |

### 删除
| 文件 | 说明 |
|------|------|
| `sherpa_engine.py` | 已下沉至 funclip_pro.core，真相源已唯一 |
| `torch_engine.py` | 已下沉至 funclip_pro.core，真相源已唯一 |

---

## Codebase Understanding（更新版）

### 架构全景
```
funclip_pro/
├── core/                          # 算法与硬件抽象 SDK（8 模块）
│   ├── asr.py                     # 离线 ASR: SenseVoiceSmall / PyTorchSenseVoice / 
│   │                              #   SherpaSenseVoice / QwenEngine [NEW]
│   ├── speaker.py                 # 说话人识别: CampPlusSpeaker
│   ├── segmentation.py            # 帧级活性检测: SegmentationEngine
│   ├── alignment.py               # 说话人-文本对齐
│   ├── tokenization.py            # 字符分词
│   ├── audio.py                   # 音频采集层 [P3.2] — 含泄露修复 [FIXED ✅]
│   └── streaming_asr.py           # 流式 ASR [P3.2] — 含 VAD 隔离 + 打字预览 [FIXED ✅]
├── config/
│   └── loader.py                  # 动态路径寻址 + DLL 补丁
├── pipeline/
│   └── offline.py                 # OfflinePipeline — 含 qwen 引擎路由 [NEW ✅]
├── utils/
│   ├── loader.py / srt.py / dll_patch.py
└── 外壳应用（薄壳）:
    ├── app_live_local.py          # 桌面实时字幕 [打字预览已恢复 ✅]
    ├── app_live_ws.py             # WebSocket 实时字幕 [已对接 core.audio ✅]
    ├── app_control.py             # Gradio 离线界面 [Qwen3/SeACo/Nano 已解禁 ✅]
    ├── asr_onnx_service.py        # FastAPI 离线服务
    └── cli_transcribe.py          # CLI 客户端
```

### Critical Files

| File | Purpose | Relevance |
|------|---------|-----------|
| `src/funclip_pro/core/audio.py` | 音频采集层（Base/Loopback/Mic/MixedStream）— **泄露已修复** | 🅰️ |
| `src/funclip_pro/core/streaming_asr.py` | 流式 ASR 引擎（SileroVAD/FsmnVad/FunAsrStreamingEngine）— **VAD 隔离 + 打字预览** | 🅱️🅲 |
| `src/funclip_pro/core/asr.py` | 离线 ASR 引擎集 — **QwenEngine 已集成** | 🅳 |
| `src/funclip_pro/pipeline/offline.py` | OfflinePipeline — **qwen 路由已打通** | 🅳 |
| `app_live_local.py` | 桌面实时字幕薄壳 — **打字预览已恢复** | 🅲 |
| `app_live_ws.py` | WebSocket 实时客户端 — **已对接 core.audio** | 🅴 |
| `app_control.py` | Gradio 离线界面 — **Qwen3/SeACo/Nano 阻断已解禁** | 🅳 |
| `tests/test_audio_stream.py` | 音频流测试 — **16 passed / 2 skipped** | 🅰️ |
| `tests/test_streaming_engine.py` | 流式引擎测试 — **14 passed** | 🅱️🅲 |

---

## 待审查项（给接手智能体）

### 建议的审查顺序

| 优先级 | 文件 | 审核要点 |
|:------:|------|---------|
| 🔴 P0 | `core/streaming_asr.py` | VAD session 隔离逻辑是否完备？预览是否有边界泄漏？ |
| 🔴 P0 | `core/audio.py` | stop() terminate 是否安全？混音 clip 是否正确？ |
| 🟡 P1 | `core/asr.py` (QwenEngine) | 接口对齐是否正确？config.json 读取路径是否正确？DOCKER_HOST 常量是否存在？ |
| 🟡 P1 | `pipeline/offline.py` | qwen 分支的提前 return 是否影响其他路径？ |
| 🟢 P2 | `app_control.py` | Qwen3/SeACo/Nano 阻断是否正确删除？ |
| 🟢 P2 | `app_live_ws.py` | 导入替换后 ASRClient 的流引用是否一致？ |
| 🟢 P2 | `tests/test_audio_stream.py` | 测试预期值与 VOLUME_BOOST 逻辑一致？ |

## 后审计修复（2026-07-14 23:07–23:19）

在安全审计报告 `.scratch/refactor-alignment/refactor-audit-report.md` 和整改规范 `.scratch/refactor-alignment/spec-refactor-remediation.md` 指导下完成的补充修复：

| # | 缺陷 | 文件 | 严重度 |
|:-:|------|------|:-----:|
| ① | 单路音频溢出(缺clip) + 空帧死循环 + get_queue_size 特性依恋 | `core/audio.py` / `tests/test_audio_stream.py` | 🔴 |
| ② | ASR 模型路径硬编码，`--model_dir` 失效 | `core/streaming_asr.py` | 🔴 |
| ③ | Qwen HTTP timeout=1200s 假死 + 无容错 | `core/asr.py` | 🔴 |
| ④ | SileroVAD 析构未释放 ONNX session 句柄 | `core/streaming_asr.py` | 🟡 |

### ① 音频物理流截断与低开销控频
**文件**: `core/audio.py`
- 单路分支（仅 mic / 仅 loop）加 `np.clip(..., -1.0, 1.0)` 防爆音
- 空帧分支加 `time.sleep(0.032)` 消除 CPU 100% 自旋
- `get_queue_size()` 改为调用子流公开方法，卸除特性依恋
- 同步更新测试预期值

### ② 流式模型传参解耦
**文件**: `core/streaming_asr.py`
- `_ensure_asr()` 模型路径改为 `self.config.get("model_dir") or "models/iic/SenseVoiceSmall"`
- `--model_dir` 命令行参数现在可以正确生效

### ③ Qwen 网络交互超时与容错
**文件**: `core/asr.py`
- `requests.post` timeout 从 `1200` 改为 `(2.0, 15.0)`（2s 连接超时，15s 读取超时）
- 分 `ConnectTimeout` / `ReadTimeout` / `ConnectionError` 三类异常捕获，打印友好日志后 raise

### ④ SileroVAD ONNX session 析构
**文件**: `core/streaming_asr.py`
- `SileroVAD` 新增 `__del__` 方法，显式 `del self.session` 触发 ONNX InferenceSession 资源释放

### 测试结果
```
tests/test_audio_stream.py    16 passed / 2 skipped
tests/test_streaming_engine.py 14 passed
```
**全量: 30 passed, 2 skipped** ✅

---

### 已知遗留问题
- **13 个已有测试失败**：`test_asr_onnx_service*.py` 中引用了已删除的 `MODEL`/`_get_torch_model`/`_get_spk_model` 属性，非本次引入
- **`tests/archive/` 死锁**：归档旧测试关闭 stdout 句柄导致 pytest 崩溃，不要碰
- **`app_live_ws.py`**：虽然已对接 core.audio，但尚未经过端到端硬件测试（需要真实麦克风/回路设备）

### 推荐的下一步
1. **code-review**：用 `code-review` skill 对新修改做双轴审查（Standard + Spec）
2. **端到端测试**：在有硬件的 Windows 机器上启动 `app_live_local.py --overlay` 验证打字预览效果
3. **提交**：确认审查通过后 commit 当前所有修改

### Suggested Skills
- `code-review` — 双轴审查（Standards + Spec）— 接手后**首要动作**
- `diagnosing-bugs` — 如果审查中发现回归
- `implement` — 修复审查发现的问题
- `tdd` — 为新逻辑补充测试

### Potential Gotchas
- **测试不能用 `pytest tests/` 目录收集**：必须显式列出测试文件
- **GPU 环境**：`E:/conda/envs/asr_ui_env/python.exe`（RTX 4080, torch 2.3.1+cu121）
- **`tests/archive/` 死锁**：不要碰
- **Qwen3 Docker**：需要 `start_qwen_backend.bat` 先运行才能端到端验证
- **TYPO 注意**：`core/asr/qwen.py` 中引用了未定义的 `DOCKER_HOST` 常量（原始代码自带的 Bug）——新版 `QwenEngine` 已改为从 `config.json` 读取，修复了这一历史遗留

---

## 下一步：GitHub 上线 (v0.8.0 Release)

**决策**：走作品集路线（选择 A），1 小时内完成最小上线，作为个人作品展示。

### Spec 文件
`.scratch/refactor-alignment/spec-github-release.md` — 按 to-spec 格式编写，包含完整的 Problem Statement、User Stories、Implementation Decisions。

### 任务清单

| # | 任务 | 文件/操作 | 估算 |
|:-:|------|---------|:---:|
| ① | 添加 Apache 2.0 LICENSE | 新建 `LICENSE` 文件 | 2 min |
| ② | 补全 pyproject.toml | authors / license / readme / urls / scripts / version→0.8.0 | 5 min |
| ③ | 创建 CHANGELOG.md | Keep a Changelog 格式，v0.1→v0.8 | 10 min |
| ④ | 修复 13 个 pre-existing 测试失败 | `tests/test_sherpa_engine.py` 等旧测试 | 20 min |
| ⑤ | 全量回归验证 | 30 passed + 全量测试通过 | 10 min |
| ⑥ | 提交 + 推送 | `git commit -m "chore: v0.8.0 release"` + `git push` | 5 min |

### CLI Entry Point 设计
```toml
[project.scripts]
funclip-pro = "funclip_pro.cli:main"
```
提供一个 `funclip-pro transcribe audio.wav --diarize` 的 CLI 命令。

### 版本计划
- **当前**: v0.8.0 (待发布)
- 之后无承诺，随缘维护
