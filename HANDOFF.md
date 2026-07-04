# Handoff: SenseVoice ASR FastAPI 微服务解耦与双模 GPU 加速调试

## Session Metadata
- **Created**: 2026-07-05T02:32:00+08:00
- **Project**: E:\project\funclip-pro
- **Target User**: 尖子
- **Branch**: `main` (Git status clean, latest commit: `b7c7c0a`)

---

## Current State Summary
我们完成了 ASR 独立微服务（`asr_service.py`）的设计与双模 TDD 开发，并在本地环境上成功完成了 GPU 加速性能调试：

1.  **FastAPI 微服务解耦**：
    - 新增了 [asr_service.py](file:///E:/project/funclip-pro/asr_service.py)。它不破坏项目原有的 Gradio 运行逻辑，单独启动并监听本地 `8001` 端口，提供 `/transcribe` 转写服务。
    - 引入了 `asyncio.to_thread` 调度 PyTorch GPU，避免了重型推理卡死主事件循环；并采用 `asyncio.Semaphore(3)` 限制最大并发推理数以防范 CUDA OOM。
    - 所有临时文件的写入和清理逻辑全置于 `try...finally` 中，消除了异常分支下的磁盘泄漏隐患。
2.  **GPU 驱动与依赖故障根治**：
    - 成功排查并解决了 pip 自动下载国内镜像源 CPU 伪高版本（`torch-2.12.1`）导致的 API 静默回滚至 CPU 慢速运行的 Bug。
    - 解决了 NumPy 2.x 与 PyTorch 1.x 之间破坏性升级导致的 C-API 二进制不兼容冲突（降级安装为 `numpy<2`，目前稳定在 `1.26.4`）。
    - 目前 GPU 推理已完全被点亮，转写速度从 184 秒缩减至 4.1 秒（提速 44 倍），RTF 实时率达到 GPU 级别的 0.01~0.05。
3.  **双模 API 管道设计**：
    - 接口支持 Form 参数 `vad_split: bool`。
    - **`vad_split=false`** (极速单句模式)：适合 Dify 聊天机器人提问场景，跳过 VAD，几百毫秒秒回。
    - **`vad_split=true`** (VAD 切句分段模式)：适合 Dify 知识库索引建库场景。调用本地 VAD 模型切句，输出以 `\n` 分段的结构化文本，大幅提升了 RAG 句子分界质量和 ASR 长文识别精度。

---

## Pending Work

### Immediate Next Steps (新会话第一步)
尖子计划明天在本地的 **Dify** 中实际配置和接入该 ASR 微服务。接管的 Agent 应该配合执行以下步骤：
1.  **启动 API 中台服务**：
    在 `asr_ui_env` 激活状态下，启动：`python asr_service.py`。
2.  **在 Dify 中创建两个不同的应用/流程图**：
    - **对话应用**：拖入 HTTP 节点，URL 指向 `http://127.0.0.1:8001/transcribe`，Form 参数 `vad_split` 设为 `false`。
    - **建库流水线应用**：HTTP 节点的 `vad_split` 设为 `true`，用于对管理员上传的长音视频切句分割。
3.  **调试联调**：
    如果在 Docker 容器内部调用宿主机的 API 端口，可能需要替换 `127.0.0.1` 为局域网 IP（如 `192.168.x.x`）或 Docker 网桥 IP（`host.docker.internal`），协助尖子调试连通性。

---

## Context for Resuming Agent (Gotchas & Environment)

### 🛠️ 环境状态
- **操作系统**：Windows 11
- **推荐 Python 虚拟环境**：`asr_ui_env`（物理路径 `E:\conda\envs\asr_ui_env`）。
- **已安装测试依赖**：在 `asr_ui_env` 中已经部署了 `pytest` 和 `httpx`，接管的 Agent 可在项目根目录下通过运行：
  `& "E:\conda\envs\asr_ui_env\python.exe" -m pytest tests/test_asr_api.py -v`
  一键通过所有的 Mock 单元测试和本地 CUDA 硬件集成测试（2 Passed）。
- **模型本地路径**：
  - ASR (SenseVoiceSmall): `E:\project\funclip-pro\model\models\iic\SenseVoiceSmall`
  - VAD (FSMN-VAD): `E:\project\funclip-pro\model\models\damo\speech_fsmn_vad_zh-cn-16k-common-pytorch`
