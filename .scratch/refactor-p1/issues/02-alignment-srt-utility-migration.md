# 02 — 提取时序对齐与 SRT/同人合并工具

**What to build:**
完成说话人分配对齐算法以及字幕文件（SRT）格式化与合并功能的包化抽离，支持在不同外壳应用间统一调用。

**Blocked by:**
01 — 建立 ASR 封装与引擎库下沉

**Status:** ready-for-agent

- [ ] 在 `src/funclip_pro/core/` 目录下新建 `alignment.py`，将 `_assign_clauses_to_speakers_seamless` 核心锚点分配对齐算法迁移进来并暴露为模块函数。
- [ ] 在 `src/funclip_pro/utils/` 目录下新建 `srt.py`，将 `_merge_same_speaker_segments` 合并说话人函数、以及所有的 ms 级别时间戳转换为 SRT 格式的工具函数（如 `_ms_to_srt`、`_segments_to_srt` 等）迁移进来。
- [ ] 编写测试 `tests/test_p1_alignment_srt.py`，利用 Mock 数据跑测对齐与合并函数，确保输出的段落结构在 VAD 段内正常合并，无越界或乱序 Bug。
