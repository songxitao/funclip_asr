# Handoff: FunClip-Pro 首次 Git 初始化与简历级重构设计对齐

## Session Metadata
- Created: 2026-07-04 20:49:29
- Project: E:\project\funclip-pro
- Branch: `main` (Git repository initialized)
- Session duration: 1.0 hours

### Recent Commits (for context)
- `8075c08` (HEAD -> main) - init: import funclip-pro decoupled codebase (2026-07-04)

## Handoff Chain
- **Continues from**: [2026-07-03-013326-funclip-pro-decoupling.md](file:///E:/project/funclip-pro/.claude/handoffs/2026-07-03-013326-funclip-pro-decoupling.md)
- **Supersedes**: None

---

## Current State Summary

我们完成了本项目的快速对接，并根据尖子的指示，重点开展了 Git 初始化与简历级工程化改造的痛点头脑风暴。

1. **Git 本地建库完成**：
   - 优化了项目根目录下的 [.gitignore](file:///E:/project/funclip-pro/.gitignore)，增加了大权重模型软链目录 `model/`、音频录音目录 `recordings/` 以及临时的任务文本文件，确保提交轻量化。
   - 在项目根目录执行了 `git init`，添加并完成了首次规范化提交（Commit ID: `8075c08`），分支命名为 `main`。
2. **简历级痛点分析与共识对齐**：
   - 确定了 Demo 距离生产级项目的硬伤所在：
     - **进程管理粗糙**：离线和实时转写深度绑定 Windows 批处理和 `cmd.exe /c` 黑框弹窗，无法跨平台且无法监控子进程生命周期。
     - **可观测性缺失**：Gradio 界面无法获取后台 ASR 的具体进度百分比和吞吐指标。
     - **代码拷贝冗余**：通过 CodeGraph 读图探索，发现 `app_live_local.py` 与 `app_live_ws.py` 均直接硬编码复制了完全一样的 Tkinter `SubtitleOverlay` 字幕悬浮窗类，违反 DRY 工程原则。
   - 确立了重构路线：优先进行**【第一阶段：核心解耦与多进程异步管道架构（消除弹窗与 UI 冗余）】**。

---

## Codebase Understanding

### Critical Files
- [app_control.py](file:///E:/project/funclip-pro/app_control.py)：WebUI 总入口，当前使用 `subprocess.Popen` 调用 cmd.exe 和 bat 脚本。
- [app_live_local.py](file:///E:/project/funclip-pro/app_live_local.py) & [app_live_ws.py](file:///E:/project/funclip-pro/app_live_ws.py)：本地实时和 WS 实时听写客户端，均硬编码了各自的 `SubtitleOverlay`。
- [core/](file:///E:/project/funclip-pro/core)：底层逻辑库，未来将封装统一的 `core/ui/overlay.py` 和 ASR 接口。
- [implementation_plan.md](file:///C:/Users/song/.gemini/antigravity/brain/306f4782-de9b-452f-97a7-418549f2f1aa/implementation_plan.md)：保存在 AI 脑区的重构路线图，详细记录了各阶段重构子任务。

---

## Decisions Made

- **安全过滤 Git**：在初始化 Git 前，将物理软链 `model/` 写入了 `.gitignore`，避免了添加数十 GB 模型的事故。
- **重构架构设计**：保留 Gradio 作为操作界面，在代码底层引入 `JobRunner`（多线程/多进程管道）和 `Queue` 回调，将外部批处理和 cmd 黑框全部重构在 Python 进程内完成。此改造方案经对齐后，能为简历提供强有力的**“可观测性设计”**与**“跨平台多进程异步 IPC 控制”**两大硬核闪光点。

---

## Pending Work & Immediate Next Steps

在下一个会话中，新接管的 AI 代理需要从以下步骤开始：

1. **第一步：消灭 UI 重复代码，抽取 overlay 模块**：
   - 提取 `app_live_local.py` 和 `app_live_ws.py` 中的 `SubtitleOverlay` 代码。
   - 在 `core/` 下新建 `core/ui/overlay.py`，并将此类迁移进去，使两端通过标准的模块导入（`from core.ui.overlay import SubtitleOverlay`）复用 UI，消除冗余拷贝。
2. **第二步：封装离线 ASR 驱动，去除黑框与临时 txt**：
   - 重构 `funclip/asr1.py` 为通用的 `ASROfflineEngine`。
   - 在 `app_control.py` 内部使用多线程/协程后台 Worker 执行转写，淘汰 `temp_task_list.txt`，并通过管道/队列捕获 stdout 进度，通过 `gr.Progress()` 将转写进度实时回传渲染到 WebUI。
3. **第三步：将启动脚本统一为 Python 入口**：
   - 将现有 Windows 的 bat 脚本重构为跨平台的 `run.py` 或进程启动类，实现 Windows/Linux 自适应环境唤醒。
