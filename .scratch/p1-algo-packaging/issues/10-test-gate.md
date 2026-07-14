# 10 — 测试门禁：pytest 全套 + DER 全量 20 场

**What to build:** 更新/新增 `tests/` 覆盖 core 各模块 + pipeline；无缓存跑全套 pytest；DER `seg_clustering` 单场 + 全量 20 场与重构前 14.85%–15.13% 无实质下滑。

**Blocked by:** 08 (FastAPI 薄路由), 09 (CLI 薄客户端)

**Status:** ready-for-agent

- [ ] pytest 全套无缓存全绿（含新增 core / pipeline 测试）
- [ ] DER `seg_clustering` 单场 + 全量 20 场 PASS，无回归
- [ ] 门禁脚本可复跑（沿用 `run_p0_der_gate.py` 思路，扩展为 P1）

**Notes:**
- 测试只校验外部行为（结构化转写、SRT 合法、DER 对齐），不关心 DLL 补丁/相对路径内部实现（spec Testing Decisions）。
- DER 评测**必须显式 `seg_clustering`**，基线口径 14.85%–15.13%。
- 真实 ML 跑 `E:\conda\envs\asr_ui_env\python.exe`；沙箱受管 Python 无 torch，单测部分需不依赖 GPU。
- 最高指导：`.superpowers/spec/2026-07-14-refactor-p0-p1-spec.md` Testing Decisions。
