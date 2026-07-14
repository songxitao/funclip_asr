# 01 — 搭建 `src/funclip_pro` 包骨架

**What to build:** 在 `src/funclip_pro` 下建立 `core/` `utils/` `pipeline/` 三个子包与各自的 `__init__.py`，使其可被 `import funclip_pro.core` / `funclip_pro.utils` / `funclip_pro.pipeline` 成功导入（即便暂时为空）。为后续 T02–T07 下沉模块提供落点，并建立统一的包导入约定（绝对导入 `funclip_pro.x`）。同步在 `AGENTS.md` 顶部补一节"包结构"说明新 SDK 布局。

**Blocked by:** None — can start immediately

**Status:** ready-for-agent

- [ ] `E:\conda\envs\asr_ui_env\python.exe -c "import funclip_pro.core, funclip_pro.utils, funclip_pro.pipeline"` 不报错（空包亦可）
- [ ] 三个子包目录与 `__init__.py` 均已创建
- [ ] `AGENTS.md` 增加"包结构"小节，注明 core / utils / pipeline 职责
- [ ] 确立绝对导入约定：模块间用 `from funclip_pro.core.x import Y`，禁止相对导入越级

**Notes:**
- 不移动任何根文件代码；仅搭骨架。
- 本票是扩展-收缩策略的"扩展"前置，所有下沉模块都只新增 `src/` 下文件、不碰根文件。
- 最高指导：`.superpowers/spec/2026-07-14-refactor-p0-p1-spec.md` L48-58。
