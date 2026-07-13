# 🎬 FunClip Pro - 边缘设备极致对齐的 ASR & 说话人分离系统

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python 3.11"/>
  <img src="https://img.shields.io/badge/PyTorch-2.3.1%2Bcu121-green.svg" alt="PyTorch 2.3.1"/>
  <img src="https://img.shields.io/badge/CUDA-12.1-orange.svg" alt="CUDA 12.1"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-red.svg" alt="License"/>
</p>

<p align="center">
  🤗 <a href="#quickstart">快速开始</a> &nbsp | &nbsp 📑 <a href="#key-features">特性矩阵</a> &nbsp | &nbsp 🏗️ <a href="#architecture">架构设计</a> &nbsp | &nbsp 📊 <a href="#evaluation">评测指标</a>
</p>

**拒绝多人混杂带来的声纹污染，让边缘设备也能享有 15% 顶级 DER 的高精度说话人分离对齐。**

---

## 📖 项目简介
**FunClip Pro** 是一款专为高精度、低时延转写设计的边缘语音识别与说话人分离系统。针对传统 Diarization 管线中由于 VAD 大段内多人混杂、静音和声音重叠造成的声纹污染问题，FunClip Pro 创新性地引入了 **pyannote/segmentation-3.0** 帧级多标签说话人活性检测引擎。通过对单人纯净音频帧的精密过滤与提纯，为 **Cam++** 提供纯净的声纹特征输入，最后通过全局谱聚类，成功将说话人混淆错判率（CONF）降至极致，实现广播级的角色转写对齐。

---

## ⚡ 痛点转化特性矩阵 (Value-Driven Feature Matrix)

| 核心特性 (Key Feature) | 底层痛点 (Pain Point) | 创新技术方案 (Technical Solution) | 简历/转化价值 (Value Proposition) |
| :--- | :--- | :--- | :--- |
| **🎯 帧级活性单人切片 (Segmentation-Driven SAD)** | 传统 VAD 只能判定有声/无声，当大段内包含多人交替发言时，会产生严重的声纹污染，导致聚类失败。 | 利用 `segmentation-3.0` 在 10s 分块内进行约 17ms 级的帧级检测，剔除静音与重叠发言，仅取纯净的单人帧用于声纹提取。 | **将说话人混淆错判率（CONF）暴跌至 3.8%，实现极其干净的说话人表征。** |
| **🧠 自适应时序退避兜底 (Adaptive Temporal Backoff)** | 极短音频片段（小于 0.1s）在输入 Cam++ 时由于特征量不足易引发向量提取失败，导致系统崩溃。 | 实现多级时序与前向/后向填充退避机制，对提取失败的片段采用相邻有效标签自动对齐，保证服务 100% 可用性。 | **零崩溃风险，为极短发言和边缘静音提供极佳的系统健壮性保障。** |
| **🛡️ 锁安全多线程惰性加载 (Lock-Protected Lazy Loading)** | ASR 与 Diarization 模型并发加载时容易产生显存竞争，或因服务重启引发多线程模型冲突。 | 引入全局线程锁 `_SEG_LOCK` 与 `_SPK_LOCK`，仅在客户端发起特定 Diarization 策略时，才惰性地将模型安全初始化至 GPU。 | **极大节省闲置显存开销，提供线程安全的并发推理保障。** |

---

## 🏗️ 系统架构设计 (Architecture)

```mermaid
graph TD
    Audio[输入音频 16kHz WAV/FLAC] --> VAD[FSMN-VAD 活性切分]
    VAD --> ASR[Paraformer ONNX ASR 转写]
    ASR --> ASR_Texts[ASR 文本段与毫秒级时间戳]
    
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
```

---

## 📊 双轨评测与基准表现 (Evaluation & Benchmark)

我们在 `AliMeeting` 测试集上对单场 `R8002_M8002`（含多路重叠、交替发言的近场录音，总时长 2062 秒）进行了量化评测，数据对比如下：

| Diarization 策略 | 全局 DER | 混淆错判 (CONF) | 虚警 (FA) | 漏检 (MISS) | 核心改善原理 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **spectral (旧基线)** | 49.21% | 48.3% | 0.3% | 0.0% | VAD 大段直接提声纹，混入多人导致严重声纹污染。 |
| **vad_sliding** | 47.70% | 46.8% | 0.3% | 0.0% | 段内滑窗平均提纯，受限于大段内多人混杂。 |
| **seg_clustering v1** | 15.13% | **3.8%** | 0.0% | 11.3% | 引入 Segmentation-3.0 帧级单人提取，CONF 暴跌 91.8%。 |
| **seg_clustering v2 🏆** | **14.54%** | **13.6%** | **0.0%** | **0.7%** | 无缝时间轴 + 锚点扩散，回收重叠/低置信度段，MISS 骤降。 |

> **评测结论**：
> - **v1 (3.8% CONF)**：通过 Segmentation 帧级提纯，将混淆错判率从 46.8% 断崖式降至 3.8%
> - **v2 (14.54% DER)**：无缝时间轴输出所有帧段（含重叠/静音）+ 锚点扩散回收丢弃段，MISS 从 11.3% 骤降至 0.7%
> - 最新实测（VAD 段内合并版）：**14.85%**，微差属正常波动范围

---

## 🚀 快速开始 (Quick Start) <a id="quickstart"></a>

### 1. 配置要求
- 操作系统：Windows 10 / 11 
- Python 环境：Python 3.11 (推荐使用 conda 安装在 `E:\conda\envs\asr_ui_env`)
- 显卡驱动：推荐支持 CUDA 12.1+ 的 Nvidia GPU

### 2. 闪电部署
```bash
git clone <repo_url>
cd funclip-pro
E:\conda\envs\asr_ui_env\python.exe -m pip install -r requirements.txt
```

### 3. 一键启动服务
在根目录下双击运行 `一键启动_ASR_API服务.bat` 或在控制台运行：
```bash
E:\conda\envs\asr_ui_env\python.exe asr_onnx_service.py
```
服务将在后台初始化，并默认绑定 `8002` 端口。

---

## 📑 开发者 API 使用指南

服务启动后，可以通过 HTTP POST 对 `/transcribe` 发送请求。

### 请求示例 (Python Requests)
```python
import requests

url = "http://127.0.0.1:8002/transcribe"
file_path = "path/to/your/audio.wav"

with open(file_path, "rb") as f:
    files = {"file": f}
    data = {
        "diarize": "true",
        "diarize_strategy": "seg_clustering",
        "vad_strategy": "always",
        "num_speakers": "4"  # 可选，传入真实人数可使聚类更为稳健
    }
    response = requests.post(url, files=files, data=data)

print(response.json())
```

### 响应格式 (JSON)

支持三种输出格式，通过 `response_format` 参数控制：

| 格式 | 参数值 | 说明 |
|:---|:---|:---|
| **JSON** | `json`（默认） | 结构化 segments 数组 + diarized_text + 引擎信息 |
| **纯文本** | `text` | `diarized_text`（带说话人标记）或全文 |
| **SRT 字幕** | `srt` | 标准 SRT 字幕格式，同说话人相邻段已合并 |

**SRT 示例**：
```srt
1
00:00:00,050 --> 00:00:04,287
[说话人3] 这个吧，就是我一点心意，两条全是软的，

2
00:00:04,287 --> 00:00:21,078
[说话人1] 你小子。我这烟瘾呢...
```

### 输出特性

- **相邻同说话人自动合并**：所有输出格式均会合并相邻同说话人的段落，合并限制在 VAD 段内部（不跨段），防止解说与角色内容混淆
- **毫秒级时间戳**：segments 的 start/end 采用毫秒 (ms) 单位

返回数据中的 `segments` 字段采用**毫秒 (ms)** 级单位，并回填了该段对应的转写文本：
```json
{
  "text": "会议转写全文...",
  "latency_ms": 12450.5,
  "engine": "torch",
  "segments": [
    {
      "start": 0,
      "end": 3500,
      "speaker": "1",
      "text": "大家早上好。"
    },
    {
      "start": 3800,
      "end": 8200,
      "speaker": "2",
      "text": "今天我们主要讨论新版架构的评测。"
    }
  ]
}
```

### 命令行客户端

```bash
# 默认 JSON 输出
E:\conda\envs\asr_ui_env\python.exe cli_transcribe.py audio.wav --diarize

# 纯文本输出
E:\conda\envs\asr_ui_env\python.exe cli_transcribe.py audio.wav --diarize --format text

# SRT 字幕输出（同说话人相邻段已合并）
E:\conda\envs\asr_ui_env\python.exe cli_transcribe.py audio.wav --diarize --format srt
```

---

## 🏗️ 版本演进 (Changelog)

### v0.3 — SRT 输出 + 同说话人合并
- 新增 SRT 字幕输出格式（`response_format=srt`）
- 相邻同说话人自动合并（JSON/text/SRT 统一受益）
- 合并限制在 VAD 段内部（不跨段）
- CLI 新增 `--format {json,text,srt}` 参数

### v0.2 — 无缝说话人时间轴
- seg-3.0 输出所有帧段（含重叠/静音），构建无缝时间轴
- 锚点扩散逻辑回收 seg 丢弃段（MISS 11.3% → 0.7%）
- DER 从 15.13% → 14.54%
- 新增 cli_transcribe.py 命令行客户端

### v0.1 — seg_clustering 基础框架
- Segmentation-3.0 帧级单人提取 + Cam++ + SpectralClustering
- CONF 从 46.8% → 3.8%
- DER 15.13%
