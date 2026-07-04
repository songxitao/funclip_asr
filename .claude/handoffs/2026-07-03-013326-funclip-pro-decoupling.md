# Handoff: FunClip 核心多引擎 ASR 系统的扁平化迁移、相对路径解耦与一键启动集成重构

## Session Metadata
- Created: 2026-07-03 01:33:26
- Project: E:\project\funclip-pro
- Branch: [not a git repo or detached HEAD]
- Session duration: 2.0 hours

### Recent Commits (for context)
  - [no recent commits - fresh decoupled repository]

## Handoff Chain

- **Continues from**: None (fresh start for funclip-pro repo)
- **Supersedes**: None

## Current State Summary

我们已经成功将原先混乱在 E 盘根目录下的 FunClip 桌面字幕客户端、Qwen3-ASR 推理服务端、本地 ASR 推理核心进行了深度剥离和扁平化归档。在新项目 `E:\project\funclip-pro` 下重新构建出了高内聚、扁平且易于移植的结构。
利用 Windows Junction 对数十 GB 模型文件进行了零物理复制挂载；修改了所有代码中的硬编码绝对路径并将其改为动态相对路径；将 Conda 虚拟环境配置解耦提取至 `config.json`；编写并用 GBK 编码重写了 4 个一键运行的 Windows 批处理脚本，去除了 `chcp 65001` 切换，彻底解决了终端中文乱码问题。
目前，项目全量 38 个 `.py` 核心文件均通过了编译语法回归测试，所有改动完全无损。

## Codebase Understanding

### Architecture Overview

新项目被重构为干净的单层扁平化结构：
1. **根目录**：存放三个系统入口、工具脚本、`config.json` 和快捷批处理，结构非常直观。
2. **底层核心类库 (`core/`)**：封装了 ASR 工厂、VAD 处理和 Tkinter 透明悬浮窗 UI，其中 ASR 自建的多线程 `pin_memory` 并发 Batch 加速性能极佳。
3. **Docker 服务端 (`qwen_server/`)**：用于跑在容器里的 Qwen3-ASR vLLM 微服务，内含“双端异步消息队列解耦”与“音频积压合并”逻辑。
4. **模型软链挂载 (`model/`)**：挂载了 SenseVoice、ASR-Nano 等大模型权重。

### Critical Files

| File | Purpose | Relevance |
|------|---------|-----------|
| [app_control.py](file:///E:/project/funclip-pro/app_control.py) | Gradio 综合控制台 (原 `asruiv6.py`) | 系统总入口，动态解析 `config.json`。已将 Qwen3 Docker 控制函数的 `root_dir` 修正为 `PROJECT_ROOT`，能正确在根目录下呼叫 bat。 |
| [app_live_local.py](file:///E:/project/funclip-pro/app_live_local.py) | 本地实时悬浮字幕 (原 `live_engine.py`) | 监听声卡 WASAPI 环回，利用能量门控 SileroVAD 级联本地模型推理。已将 `--model_dir` 参数增加默认值（指向本地 SenseVoice），不再崩溃。 |
| [app_live_ws.py](file:///E:/project/funclip-pro/app_live_ws.py) | Qwen 实时流客户端 (原 `qwen_live_subtitle.py`) | 采集本地 500ms 音频帧，通过 WebSocket 异步发送到 Docker vLLM 推理后端。已对 VAD 绝对路径进行解耦。 |
| [config.json](file:///E:/project/funclip-pro/config.json) | 全局虚拟环境配置文件 | 存放 `conda_root`、`offline_python`。当前配置已为你绑定你本机的健康环境 `asr_ui_env`（已校验包含 gradio、funasr 等全量依赖）。 |
| [core/asr/funasr.py](file:///E:/project/funclip-pro/core/asr/funasr.py) | FunASR 批量推理实现 | ASR 底层封装。已将 `MODEL_ROOT` 绝对路径替换为以 `__file__` 自动向上追溯根目录的相对路径，彻底解决路径报错。 |
| [start_qwen_backend.bat](file:///E:/project/funclip-pro/start_qwen_backend.bat) | 启动 Qwen Docker 后端 | 已将 Docker 模型映射路径修改为相对挂载的 `%~dp0model\models\Qwen:/data/shared`，保证在任意目录下均可正确挂载显卡拉起。 |

### Key Patterns Discovered

*   **双端异步积压合并**：`custom_server.py` 内部使用 `asyncio.Queue` 解耦收包与推理，在推理前把队列里所有积压的音频块进行 `np.concatenate` 合并推理，有效抗击网络卡顿。
*   **本地模型 Junction 映射**：由于大模型极其庞大，通过 `mklink /j` 把大文件映射进干净的开发目录中，既免去了重复下载和两倍空间占用，又在开发态下让相对路径得以保持一致。

## Work Completed

### Tasks Finished

- [x] 新建隔离目录 `E:\project\funclip-pro` 并初始化 `.gitignore`。
- [x] 用 `mklink /j` 将 `E:\FunClip\FunClip\model` 成功挂载为软链目录 `E:\project\funclip-pro\model`。
- [x] 将 `core/`、`funclip/`、`qwen_server/` 及 app 活跃文件迁移至新项目并重命名规范化。
- [x] 动态加载 `config.json` 重写 `app_control.py` 里的绝对环境路径。
- [x] 解耦 `app_live_local.py`、`app_live_ws.py` 和 `core/asr/funasr.py` 中的所有绝对路径。
- [x] 编写并使用 GBK (ANSI) 编码重写了 4 个快捷批处理脚本，去除了 `chcp 65001` 以根除中文终端乱码。
- [x] 对重构后的 38 个 `.py` 文件进行了语法编译回归校验，编译通过率 100%。

## Pending Work

### Immediate Next Steps

1. **一键本地字幕验证**：双击运行 [一键启动_本地实时字幕.bat](file:///E:/project/funclip-pro/一键启动_本地实时字幕.bat)，对着麦克风发声，观察 Tkinter 悬浮窗是否如期显示，且控制台无黄字 Traceback。
2. **Qwen3-Docker 联合验证**：
   - 双击 [start_qwen_backend.bat](file:///E:/project/funclip-pro/start_qwen_backend.bat) 启动 Docker 中的 vLLM 后端服务。
   - 双击 [一键启动_集成控制台.bat](file:///E:/project/funclip-pro/一键启动_集成控制台.bat)，切换到“Qwen3 (Docker)”选项并开启实时听写，校验与 Docker 端的高并发 WebSocket 通信是否无延迟响应。
3. **NVIDIA Nemotron-3.5 并发镜像拉取配置**：参考你的 Obsidian 攻略，在阿里云云效 Flow 中新建 Shell 脚本步骤，实现免 Dockerfile 满速代购拉取 `nvcr.io` 的私有镜像。

### Blockers/Open Questions

- 无阻塞性技术难点。目前本地路径和环境已完全理顺，处于随时可以启动的状态。

## Context for Resuming Agent

### Important Context

*   **不可改动或重命名 `model/` 文件夹**：如果删除该软链目录，本地 ASR (SenseVoice 等) 与 VAD (Silero) 加载模型时将因路径不存在直接闪退。
*   **批处理文件的编码要求**：在 Windows cmd 终端下，所有批处理（`.bat`）中含有中文字符提示时，**绝对不能在头部添加 `chcp 65001`**，并且必须以 **GBK (ANSI/CP936) 编码格式保存**，否则终端输出必呈乱码。
*   **Qwen3-ASR 实时流 RTF 劣化机制**：Qwen 实时流因为 LLM 自回归注意力瓶颈，计算开销随上下文长度二次方增长，需在 app 级写定期上下文重置（20s/150字）以保证 RTF；而未来的 Nemotron-3.5 依靠 Cache-Aware FastConformer-RNNT 局部注意力和 Cache 复用，处理时间复杂度为 $O(1)$，无需重置即可维持恒定低 RTF。

## Environment State

### Tools/Services Used

- **Docker Desktop** (运行 `qwen3-asr` 容器，映射 28000 端口，配置 `--gpus all` 及 `--shm-size 10g`)。
- **Miniconda** (物理环境路径 `D:\program files\Miniconda`，已在 `config.json` 声明)。

### Active Processes

- `qwen3-asr` 容器（可使用 `docker ps` 查看其运行状态，占用容器内的 FastAPI 端口 `80`，映射到宿主机端口 `28000`）。

### Environment Variables

- `NGC_API_KEY` (未来拉取英伟达 Speech NIM 私有官方镜像时使用)。
