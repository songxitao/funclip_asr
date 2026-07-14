# 11 — code-review + 接手 handoff

**What to build:** 按 `code-review` skill 双轴自审（Standards / Spec）；写 P1 接手 handoff 到 `.claude/handoffs/`；更新根 `HANDOFF.md` 记录等价性证据 + DER 门禁结果 + 派发收口说明。

**Blocked by:** 10 (测试门禁)

**Status:** ready-for-agent

- [ ] code-review 通过（无红线违反、无硬编码回归）
- [ ] handoff 记录等价性证据 + DER 门禁结果
- [ ] 旧根引擎文件（segmentation_engine.py 等）已处理为薄再导出或删除，且外部 import 不破

**Notes:**
- 双轴：Standards（项目规约 / 红线）+ Spec（`.superpowers/spec/2026-07-14-refactor-p0-p1-spec.md` 是否达成）。
- DER 门禁结果须明确：单场 + 全量 20 场 `seg_clustering` 与 14.85%–15.13% 对比。
- 红线复查清单：numpy 1.26.4、`powerset.cpu()` 在 `to_multilabel` 前、时间戳 ms、`_run_inference` 四元组、`apply_dll_patch()` 保活、零硬编码盘符。
- 最高指导：`.superpowers/spec/2026-07-14-refactor-p0-p1-spec.md`；局部 skill：`code-review` / `handoff`。
