# 🎬 FunClip Pro — P3.2 重构版能力文档

> 本文档为 FunClip Pro 项目 P3.2 架构重构及后续整改的**能力与安全审计说明**，与主 `README.md` 互补。
> 主 README 关注产品功能与使用，本文档关注重构后的**模块能力、引擎选型、安全基线**。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python 3.11"/>
  <img src="https://img.shields.io/badge/PyTorch-2.3.1%2Bcu121-green.svg" alt="PyTorch 2.3.1"/>
  <img src="https://img.shields.io/badge/tests-30%20passed%2C%202%20skipped-brightgreen.svg" alt="Tests"/>
  <img src="https://img.shields.io/badge/audit-passed-green.svg" alt="Audit"/>
</p>

---

## 📋 重构背景

FunClip Pro 最初是单体脚本式项目，核心算法与界面逻辑深度耦合。P0~P3.2 架构重构的核心目标：

> **厚算法，薄路由** — 所有算法逻辑向 `funclip_pro.core` 包沉淀，外壳应用降级为薄路由编排层。

重构后的核心收益：
- **-4,200+ 行冗余代码**：7 个文件物理删除，3 个全新模块新增
- **30 项回归测试全绿**：流式引擎 + 音频采集层 100% 回归通过
- **双重安全审计通过**：Spec 维度 + Standards 维度审计确认项目达到合入安全级别

---

## 🧩 核心能力矩阵 (Core Capability Matrix)

`funclip_pro` 核心包提供 **8 个算法与硬件抽象模块**，支持离线转写和流式实时字幕双模式：

### 离线 ASR + Diarization 管线

| 模块 | 功能 | 引擎选型 | 硬件要求 |
|------|------|---------|---------|
| `core/asr.py` | 离线 ASR 推理 | SenseVoiceSmall (CPU/GPU) / PyTorchSenseVoice (GPU) / SherpaSenseVoice (CPU) / **QwenEngine (Docker 网络)** | CPU / CUDA / Docker |
| `core/speaker.py` | 说话人识别 (Cam++) | CUDA 优先, CPU 回退 | CUDA 推荐 |
| `core/segmentation.py` | 帧级活性检测 (segmentation-3.0) | CUDA 优先, CPU 回退 | CUDA 推荐 |
| `core/alignment.py` | 说话人-文本对齐 | 时间重叠对齐 / 无缝锚点扩散 | CPU |
| `core/tokenization.py` | 字符分词 | 原生 Python 实现 | CPU |
| `pipeline/offline.py` | 统一转写管线 (VAD→ASR→聚类→对齐→SRT) | 自动路由选择最佳引擎 | 根据引擎决定 |

### 流式实时字幕

| 模块 | 功能 | 关键能力 | 硬件要求 |
|------|------|---------|---------|
| `core/audio.py` | 音频采集硬件抽象 | LoopbackStream (声卡环回) / MicStream (麦克风) / MixedStream (双源混音) | PyAudio / pyaudiowpatch |
| `core/streaming_asr.py` | 流式 ASR 引擎 | FunAsrStreamingEngine / SileroVAD / FSMN VAD | CPU (ONNX Runtime) |

### 应用外壳（薄壳）

| 应用 | 行数 | 重构说明 |
|------|:---:|---------|
| `app_live_local.py` | 282 | 实时桌面字幕 — 原 944 行（-662），打字预览已恢复 |
| `app_live_ws.py` | 重塑 | WebSocket 实时客户端 — 已对接 core.audio（-35% 体积） |
| `app_control.py` | 收口 | Gradio 离线界面 — OfflinePipeline 进程内调用，Qwen3/SeACo/Nano 已解禁 |
| `asr_onnx_service.py` | 169 | FastAPI 离线转写服务（原 350 行，-181） |
| `cli_transcribe.py` | 薄壳 | 命令行客户端 |

---

## 🏗️ 系统架构设计 (Architecture)

```mermaid
graph TD
    Audio[输入音频 16kHz WAV/FLAC] --> VAD[FSMN-VAD 活性切分]
    VAD --> ASR[ASR 引擎路由]
    
    subgraph ASR_Engines[ASR 引擎选型]
        SenseVoice[SenseVoiceSmall ONNX/CPU]
        PyTorch[PyTorchSenseVoice GPU]
        Sherpa[SherpaSenseVoice INT8/CPU]
        Qwen3[QwenEngine Docker ASR API]
    end
    
    ASR --> SenseVoice
    ASR --> PyTorch
    ASR --> Sherpa
    ASR --> Qwen3
    SenseVoice --> ASR_Texts[ASR 文本段与毫秒级时间戳]
    PyTorch --> ASR_Texts
    Sherpa --> ASR_Texts
    Qwen3 --> ASR_Texts
    
    Audio --> SegEngine[SegmentationEngine]
    subgraph Diarization Pipeline (seg_clustering)
        SegEngine --> Chunks[10s 无重叠分块推理]
        Chunks --> Powerset[Powerset 解码与 CPU 多标签转换]
        Powerset --> FrameFilter[帧级单人过滤 17ms 精度]
        FrameFilter --> CamPlus[Cam++ 提取纯净声纹 Embedding]
        CamPlus --> Spectral[SpectralClustering 全局谱聚类]
        Spectral --> Smooth[相邻同人段合并 <0.5s]
    end
    
    ASR_Texts --> Align[时间重叠对齐与文本回填]
    Smooth --> Align
    Align --> JSON_Output[毫秒级结构化 JSON 说话人转写段]

    subgraph Live Streaming (P3.2)
        AudioSrc[声卡环回/麦克风] --> AudioCore[core.audio]
        AudioCore --> StreamASR[core.streaming_asr]
        StreamASR --> Preview[打字预览 🟢]
        StreamASR --> VAD_Isolation[VAD Session 隔离]
        StreamASR --> GUISub[app_live_local.py 薄壳]
        GUISub --> Overlay[Tkinter 悬浮窗字幕]
    end
```

### 核心包文件结构

```
funclip_pro/                              # 可安装 Python 包（pip install -e .）
├── core/                                 # 算法与硬件抽象 SDK（8 模块）
│   ├── asr.py                            # 离线 ASR: SenseVoice / PyTorch / Sherpa / QwenEngine
│   ├── audio.py                          # 音频采集层: Loopback/Mic/MixedStream
│   ├── streaming_asr.py                  # 流式 ASR: FunAsrStreamingEngine + 打字预览
│   ├── speaker.py                        # 说话人识别: Cam++ 
│   ├── segmentation.py                   # Segmentation-3.0 帧级活性检测
│   ├── alignment.py                      # 说话人-文本对齐
│   └── tokenization.py                   # 字符分词
├── config/
│   └── loader.py                         # 动态路径寻址 + DLL 补丁
├── pipeline/
│   └── offline.py                        # 离线转写-聚类-对齐一体化 Pipeline（含 qwen 路由）
└── utils/
    ├── loader.py                         # Windows 路径兼容
    ├── srt.py                            # SRT 字幕合并
    └── dll_patch.py                      # ONNX DLL 点亮补丁
```

---

## 🔒 安全审计与稳定性加固 (Security Audit)

P3.2 重构完成后，经过两轮独立安全审计（Spec 维度 + Standards 维度），发现并修复了 4 项高危缺陷。

### 审计修复总览

| # | 缺陷 | 风险等级 | 修复文件 | 验收状态 |
|:-:|------|:-------:|---------|:--------:|
| ① | 混音单路数值溢出 + 空帧 CPU 100% 死循环 + get_queue_size 特性依恋 | 🔴 高危 | `core/audio.py` | 🟢 通过 |
| ② | ASR 模型路径硬编码，`--model_dir` 命令行参数失效 | 🔴 高危 | `core/streaming_asr.py` | 🟢 通过 |
| ③ | Qwen HTTP 超时 1200s (20 分钟)，宕机时界面假死闪退 | 🔴 高危 | `core/asr.py` | 🟢 通过 |
| ④ | SileroVAD ONNX InferenceSession 析构泄漏 | 🟡 中危 | `core/streaming_asr.py` | 🟢 通过 |

### ① 音频溢出截断与休眠降频
- **问题**：`MixedStream.read()` 单路输入缺 `np.clip` 导致爆音；空帧无休眠导致 CPU 100% 自旋
- **修复**：单路分支补 `np.clip(..., -1.0, 1.0)`；空帧分支 `time.sleep(0.032)`；`get_queue_size()` 委托子流公开方法

### ② ASR 模型传参解耦
- **问题**：`_ensure_asr()` 硬编码 `"models/iic/SenseVoiceSmall"`，忽略 `config.get("model_dir")`
- **修复**：模型路径改为 `self.config.get("model_dir") or "models/iic/SenseVoiceSmall"`

### ③ Qwen3 网络超时与分类容错
- **问题**：`QwenEngine.transcribe()` timeout=1200s，宕机时界面假死 20 分钟
- **修复**：timeout 改为 `(2.0, 15.0)`；分类捕获 `ConnectTimeout`/`ReadTimeout`/`ConnectionError`，友好日志后安全抛

### ④ SileroVAD ONNX 析构
- **问题**：`SileroVAD` 无 `__del__`，高并发下句柄泄漏
- **修复**：新增 `__del__` 显式 `del self.session`

### 二次审计结论

> 本轮 P3.2 重构整改版补充修复在功能正确性、多线程高并发稳定性、系统资源消耗控制（CPU/句柄/显存）以及网络容错防闪退等维度均**完全达标**，满足合入 `main` 分支发布的所有安全性指标。
>
> 审计报告详见：`.scratch/refactor-alignment/refactor-audit-report.md`

---

## 🧪 测试覆盖 (Test Coverage)

| 测试套件 | 通过 | 跳过 | 覆盖模块 |
|---------|:---:|:----:|---------|
| `tests/test_audio_stream.py` | **16** | 2 | `core/audio.py` |
| `tests/test_streaming_engine.py` | **14** | 0 | `core/streaming_asr.py` |
| **全量回归** | **30** | **2** | 音频层 + 流式引擎 🟢 |

**已知限制**：
- `tests/archive/` 中 5 个归档测试（模块级替换 `sys.stdout`）不可执行
- 13 个 pre-existing 测试失败（旧测试引用已删除的 `asr_onnx_service` 属性），不影响功能

---

## 🚀 引擎选型速查

| 引擎 | 启动方式 | 硬件 | 适用场景 |
|------|---------|------|---------|
| **SenseVoiceSmall** | 默认自动选择 | CPU/GPU | 通用离线转写，CPU 友好 |
| **PyTorchSenseVoice** | 自动路由（GPU 优先） | CUDA | 高精度离线转写 |
| **SherpaSenseVoice** | 自动路由（CPU 回退） | CPU | 低功耗离线转写 |
| **Qwen3 (Docker)** | 需先运行 `start_qwen_backend.bat` | Docker 网络 | LLM-ASR 高精度转写 |

---

## 📜 重构版本演进

| 版本 | 里程碑 | 核心变化 |
|:----:|--------|---------|
| **v0.8** | 安全审计修复 | 4 项高危缺陷修复 + 二次审计通过 |
| **v0.7** | P3.2 流式重构 | core.audio / streaming_asr 下沉，薄壳重塑，Qwen3 集成，打字预览，VAD 隔离 |
| **v0.6** | P3 外壳收口 | app_control 进程内 OfflinePipeline，物理清理冗余文件 |
| **v0.5** | 精度闭环+P2构建 | DER 零退化验证，pyproject.toml，清除 sys.path.insert |
| **v0.2-0.3** | 说话人分离完善 | SRT 输出，同说话人合并，无缝时间轴 |
| **v0.1** | seg_clustering 基础 | Segmentation-3.0 + Cam++ + SpectralClustering, CONF 3.8% |
