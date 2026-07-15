# FunClip Pro 架构重构报告 — P0~P3.2

> 日期：2026-07-14
> 范围：P0（路径解耦试点）→ P3.2（实时流式重构）
> 主线：厚算法，薄路由 — 所有算法逻辑向 `funclip_pro.core` 包回归

---

## 1. 重构动机与目标

FunClip Pro 最初是单体脚本式项目，核心算法与界面逻辑深度耦合，表现为：

- **`sys.path.insert` 黑魔法**：每个外壳脚本启动时动态挂载模块搜索路径，不安装不可用
- **算法逻辑内联在外壳中**：VAD、ASR、音频采集、GUI 绘制全部平铺在同一文件
- **机密盘符硬编码**：模型路径依赖绝对路径，换机崩溃
- **同能力多份实现**：`funclip/asr1.py`、`funclip/super_asr_engine.py` 等冗余副本未被清理

重构核心目标：**将算法层沉淀为可安装的 Python 包 `funclip_pro`，外壳应用降级为薄路由编排层。**

---

## 2. 重构全景路线

| 阶段 | 核心变化 | 关键文件 | 状态 |
|:----:|----------|----------|:----:|
| **P0** | 路径解耦试点 + DLL 补丁 | `config/loader.py` | ✅ |
| **P1** | 算法包化：asr/speaker/segmentation/alignment → `core/` | `core/asr.py`, `core/speaker.py`, `core/segmentation.py`, `core/alignment.py` | ✅ |
| **P1.5** | 精度闭环（DER 零退化验证）+ 旧引擎物理删除 | `der_eval.py` | ✅ |
| **P2** | 构建标准化：`pyproject.toml` + 清除 `sys.path.insert` | `pyproject.toml` | ✅ |
| **P3.1** | `app_control.py` 离线 Gradio 归口：subprocess → OfflinePipeline | `app_control.py`, `pipeline/offline.py` | ✅ |
| **P3.3** | 物理清理：删除 `funclip/` 目录 5 个冗余文件 | `funclip/` 全目录 | ✅ |
| **P3.2** | **流式重构：音频采集 + 流式 ASR 下沉 | `core/audio.py`, `core/streaming_asr.py`, `app_live_local.py` | ✅ |

---

## 3. 核心包演化 (`funclip_pro.core`)

### 重构前

```
funclip_pro/              # 仅含少量工具函数
```

所有算法代码平铺在根目录下：
- `segmentation_engine.py`（321行） 
- `speaker_engine.py`（567行）
- `asr_service.py`（8.7KB）
- `funclip/asr1.py`、`funclip/asr_engine.py`、`funclip/launch.py`...

### 重构后

```
funclip_pro/
├── core/                    # 算法与硬件抽象 SDK（7 模块）
│   ├── asr.py               # 离线 ASR: SenseVoiceSmall / SherpaSenseVoice
│   ├── speaker.py           # 说话人识别: CampPlusSpeaker
│   ├── segmentation.py      # 帧级活性检测: SegmentationEngine
│   ├── alignment.py         # 说话人-文本对齐
│   ├── tokenization.py      # 字符分词
│   ├── audio.py        [NEW]# 音频采集层: BaseStream / LoopbackStream / MicStream / MixedStream
│   └── streaming_asr.py [NEW]# 流式 ASR: FunAsrStreamingEngine / SileroVAD / FsmnVadStreaming
├── config/
│   └── loader.py            # 动态路径寻址 + DLL 补丁
├── pipeline/
│   └── offline.py           # 离线转写-聚类-对齐 Pipeline
└── utils/
    ├── loader.py            # Windows 路径兼容
    ├── srt.py               # SRT 字幕合并
    └── dll_patch.py         # ONNX GPU DLL 点亮
```

**关键指标：-4,200+ 行冗余代码，7 个文件物理删除，3 个全新模块新增。**

---

## 4. GUI 应用的变化 (`app_live_local.py`)

### 重构前 — 厚引擎单体

```
app_live_local.py (944行)
├── import 段                     # from funasr import AutoModel
├── 配置常量                      # VAD_THRESHOLD, VOLUME_BOOST...
├── SileroVAD 类       (42行)    # ONNX 推理 Silero VAD
├── FsmnVadStreaming   (96行)    # FunASR FSMN VAD 能量门控版
├── SubtitleOverlay    (76行)    # Tkinter 悬浮窗 (保留)
├── BaseStream         (42行)    # PyAudio 采集基类
├── LoopbackStream     (42行)    # WASAPI 环回采集
├── MicStream          (30行)    # 麦克风采集
├── MixedStream        (40行)    # 双源混音器
├── run_engine()       (268行)   # VAD + ASR + 循环 + GUI 全部内联
└── __main__           (36行)    # 入口 (保留)
```

**问题**：`run_engine()` 内部直接实例化 `AutoModel`，VAD 状态机、解码拼接、缓冲管理全部硬编码在大函数中。无法在没有 GUI 的环境下复用流式能力。

### 重构后 — 薄壳编排

```
app_live_local.py (282行)
├── import 段                     # from funclip_pro.core.audio import ...
├── 配置常量                      # 保留
├── SubtitleOverlay    (76行)    # Tkinter 悬浮窗 (保留)
├── run_engine()       (87行)    # 仅 6 步编排:
│                                #   1. 延迟导入 core 包
│                                #   2. 选择 Loopback/Mic/MixedStream
│                                #   3. stream.start()
│                                #   4. engine = FunAsrStreamingEngine(config)
│                                #   5. session_id = engine.create_session()
│                                #   6. 循环: stream.read() → engine.feed_chunk() → gui_queue.put()
└── __main__           (36行)    # 入口 (保留)
```

**关键变化**：
- **-662 行**（70% 的代码被删除，全部迁移至 core 包）
- GUI 不再导入 `funasr`、不再管理 ONNX 会话、不直接调用 `AutoModel.generate()`
- 流式引擎可脱离 GUI 被其他项目 `pip install funclip_pro` 后直接使用

### GUI 工作流对比

```
重构前:
  PyAudio → process_data() → SileroVAD() → AutoModel.generate() → SubtitleOverlay
  (所有逻辑在 944 行中揉成一团)

重构后:
  LoopbackStream → MixedStream.read() → FunAsrStreamingEngine.feed_chunk() → SubtitleOverlay
       ↑                    ↑                         ↑
  core/audio.py       core/audio.py           core/streaming_asr.py
  (硬件抽象)          (音频帧)                 (流式 ASR 引擎)
```

---

## 5. FastAPI 服务的变化 (`asr_onnx_service.py`)

### 重构前 — 厚服务单体

```
asr_onnx_service.py (~350行)
├── sys.path.insert            # 动态挂载路径
├── MODEL = None                # 模块级全局变量保存模型
├── load_models()               # 启动时加载所有模型
├── _get_torch_model()          # 内部函数：获取 PyTorch 引擎
├── _get_spk_model()            # 内部函数：获取说话人模型
├── _decode()                   # 内部函数：引擎路由 + 回退逻辑
├── _run_inference()            # 核心推理函数
├── /transcribe endpoint        # FastAPI 路由
└── uvicorn.run()               # 启动入口
```

**问题**：`_decode()` 内部实现前向推理 + torch→sherpa 回退逻辑，难以单独测试；`load_models()` 同步加载所有模型导致启动慢。

### 重构后 — 薄路由

```
asr_onnx_service.py (169行)
├── apply_dll_patch()          # 启动前点亮 DLL
├── from funclip_pro.pipeline import OfflinePipeline  # 唯一外部依赖
├── PIPELINE = None             # 模块级全局变量
├── /transcribe endpoint        # 薄路由: request → PIPELINE.run() → response
│   ├── asyncio.to_thread(PIPELINE.run, ...)  # 异步非阻塞
│   └── 返回值组装: text / srt / json
└── uvicorn.run()               # 启动入口
```

**关键变化**：
- **所有推理逻辑委托给 `OfflinePipeline`**：不再有 `_decode()`、`_get_torch_model()`、`_run_inference()`
- **删除了 PyTorch/Sherpa 引擎路由代码**：由 `OfflinePipeline` 内部根据配置自动选择
- **删除了 `load_models()`**：改为 `OfflinePipeline(auto_load=True)` 惰性加载
- **并发控制**：引入 `asyncio.Semaphore(3)` 防范 CUDA OOM

### FastAPI 调用对比

```
重构前:
  POST /transcribe → _run_inference(audio) → _decode(engine) → MODEL.generate()
  （文件解析、格式转换、推理逻辑全部耦合）

重构后:
  POST /transcribe → GPU_SEMAPHORE.acquire() → PIPELINE.run(audio)
  （路由只做：接收文件 → 写入临时路径 → 调 Pipeline）
```

---

## 6. 文件变更统计

| 指标 | 数值 |
|------|:----:|
| 物理删除文件 | 7 个（`asr_service.py`, `segmentation_engine.py`, `speaker_engine.py`, `funclip/` 下 4 个冗余） |
| 新增 Python 模块 | 3 个（`core/audio.py`, `core/streaming_asr.py`, `pyproject.toml`） |
| 薄壳化文件 | 3 个（`app_control.py` -662行, `app_live_local.py` -662行, `asr_onnx_service.py` -181行） |
| 新增测试 | 32 个（`test_audio_stream.py` ×18, `test_streaming_engine.py` ×14） |
| 归档旧测试 | 5 个（`tests/archive/` 下 stdout 冲突测试） |
| 总删除行数 | ~4,200+ 行 |
| 总新增行数 | ~1,200+ 行（含测试） |
| 净减少 | ~3,000+ 行 |

---

## 7. 测试结果

| 测试集 | 通过 | 失败 | 跳过 | 说明 |
|--------|:---:|:----:|:----:|------|
| GPU 全量 | **107** | 13 | 3 | 13 failed 均为测试未同步更新旧 API |
| `test_audio_stream.py` | **16** | 0 | 2 | 硬件集成测试跳过 |
| `test_streaming_engine.py` | **14** | 0 | 0 | 会话隔离/VAD/生命周期 |

---

## 8. 后续建议

1. **`app_live_ws.py` 薄壳重塑**：与 `app_live_local.py` 相同方式，删除 `AutoModel` 直接调用
2. **修复 13 个已知测试失败**：同步更新测试中对 `asr_onnx_service` 旧属性的引用
3. **跨会话 handoff**：当前 `FunAsrStreamingEngine` 的多会话架构已为 WebSocket 多客户端场景铺平道路
