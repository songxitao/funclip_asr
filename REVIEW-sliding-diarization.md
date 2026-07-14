# 代码审查报告：funclip-pro 滑窗说话人分离（sliding diarization）

> 审查者：WorkBuddy | 日期：2026-07-12 | 审查对象：commit `48fc641`→`b53d672`（7 笔）
> 方法：阅读 3 份 handoff + codegraph 符号查询 + 逐文件 Read + 复跑单测（7 passed 复现）

---

## 一、审查范围

| 项 | 内容 |
|---|---|
| Handoff | `HANDOFF.md`（与归档版 `HANDOFF-2026-07-12-sliding-review.md` 内容一致）、`.claude/handoffs/2026-07-12-014154-*.md`（88/100 READY） |
| 核心代码 | `speaker_engine.py`（`segment_sliding_window` + `cluster_sliding`）、`asr_onnx_service.py`（`_run_inference` sliding 分支） |
| 测试 | `tests/test_sliding_segmentation.py`（6 测）、`tests/test_sliding_integration.py`（1 测） |
| 评测 | `ali_der_eval.py`、`run_ali_der_full.py`、`der_eval.py`、`ali_near_prep.py` |
| codegraph | `cluster_sliding` callers 确认唯一生产调用者是 `_run_inference`（asr_onnx_service.py:388） |

---

## 二、代码审查发现

### speaker_engine.py

#### P1-1 None 填充导致尾部/中间静音被标成人（FA 上升根因）
`cluster_sliding` L286-292：无效窗（`extract_embedding` 返回 None）用 `last_valid`（前一个有效标签）前向填充，首窗无效填硬编码 `1`。
- **后果**：音频中的静音/停顿段（Cam++ 因 `<_MIN_SAMPLES` 或能量不足返回 None）会被标成前一个说话人，hyp 把静音也算成有人说话 → FA。
- **印证**：handoff 记录 FA 从 0.3%→3.4% 上升，正是此机制。
- **建议**：无效窗改用"静音/未知"标签（如 0 或 None），不输出为 hyp 段；或叠加轻量能量/VAD 判静音，静音窗跳过。

#### P1-2 合并相邻同人窗丢失边界精度
L293-307：合并后边界是窗边界（0.5s 步长粒度）。真实说话人切换点可能在窗中间，合并后边界最大误差 0.5s。DER collar=0.25s 只能吸收 0.25s，剩余算 CONF/FA/MISS。
- **定性**：滑窗固有局限，design spec 已声明，可接受，但应在全量评测中量化边界误差占比。

#### P1-3 自动估 K 过分割，29.76% 是 oracle-K 上限
L264-268：`n_clusters = max(2, min(20, n // 10))`。单场 34min 约 2000 窗 → 估 K=20，但实际会议 3-6 人，严重过分割。
- **关键**：评测时 `ali_der_eval.py` 传了 `num_speakers`（RTTM 真实人数，oracle-K），L38 `data={..."num_speakers": str(n_spk)}`，所以**自动估 K 没触发**。
- **结论**：handoff 的 29.76% 是 oracle-K 结果，是**上限**，不代表零样本部署效果。handoff Assumptions 未明确这点对结论的影响。
- **建议**：补一组不传 num_speakers 的零样本评测，量化自动估 K 的退化程度。

### asr_onnx_service.py

#### P1-4 sliding 分支隐式依赖 VAD chunks 非空（设计与实现矛盾）
L445 `if diarize and chunks:` —— sliding 分支在 `chunks`（VAD 切出）非空时才触发。
- **矛盾**：sliding 名义"不依赖 VAD 段做分人"（handoff 决策表），但实现上**依赖 VAD 段非空作为触发条件**。
- **后果**：若某场 VAD 全判静音或失败（`chunks` 空），sliding 不执行，`segments` 空 → DER 100% MISS，且静默无报错。
- **建议**：sliding 分支改为 `if diarize:` 独立判断，不依赖 `chunks`（sliding 用 `y` 整段音频，本就不需要 chunks）。

#### P1-5 docstring 与实现不一致
L395 docstring `"single" | "two_stage"(默认) | "spectral"` —— **漏了 sliding**。默认值仍是 `"two_stage"`。新增策略后 docstring 未同步。

#### P2-1 sliding 分支 segments 的 text 为空
L459 `"text": ""` —— sliding 段不回填 ASR 文字。`diarized_text`（L474-476）会拼成空串。design 如此，但下游若依赖 text 会功能退化，需明确告知。

### 评测脚本

#### P1-6 der_eval.py 的 argparse choices 不含 sliding
`der_eval.py` L163-165：`--diarize_strategy` choices=`["single","two_stage","spectral"]`，**AISHELL-4 评测脚本走不了 sliding**。
- **影响**：两套评测脚本割裂（der_eval.py 给 AISHELL-4，ali_der_eval.py 给 AliMeeting），sliding 只在后者可用。维护性差，易误用。

#### P1-7 DER 的 greedy_map 多对一映射偏乐观
`der_eval.py` L90-106：每个 hyp 说话人独立映射到重叠最多的 ref，**允许多对一**（多个 hyp 说话人→同一 ref）。
- **印证**：`test_results/der_spectral.json` 的 mapping `1→006-M, 3→006-M`，hyp 把 006-M 拆成 speaker 1 和 3，多对一"洗白"了过分割。
- **后果**：sliding 的 29.76% 可能因多对一偏乐观。标准 DER 用最优一对一映射（匈牙利）或 pyannote 联合映射。
- **建议**：补 `pyannote.metrics` 交叉校验。

#### P2-2 collar 仅打 ref 边界
`der_eval.py` L120-123：只对 ref 的 start/end 打 collar ignore，hyp 边界不打。接近 pyannote 标准（collar 包围 ref 边界），但结合多对一 greedy 仍有偏差，需联合校验。

#### P2-3 ali_der_eval.py 单场结果无持久化
`eval_one` L73 只 return，不写文件。单场 29.76% 的数字仅在 handoff 文字里，无 JSON 佐证。`run_ali_der_full.py` 才写 JSON。建议单场也落盘。

### 测试代码

#### P2-4 单测 mock 未覆盖真实 Cam++ 行为
`test_sliding_segmentation.py` 用 `patch extract_embedding` 返回固定向量，不加载真实模型。合并/填充逻辑验证了，但**真实 Cam++ 在 1.5s 短窗上的 embedding 质量未验证**（Cam++ 训练用更长片段，1.5s 窗可能不稳定）。需端到端验证（全量 Task 6）。

#### P2-5 单测未覆盖 None 填充致 FA 场景
`test_cluster_sliding_none_embedding_filled` 只验证"不崩"（`assert len(merged) >= 1`），没验证 None 填充是否错误延长段、产生 FA。应加断言：中间窗 None，验证填充后段时长是否超出有效窗范围。

#### P2-6 集成测试未断言服务端传参
`test_sliding_integration.py` L25-26 `FakeSpk.cluster_sliding` 固定返回 2 段，没验证服务端是否正确透传 `win_sec/step_sec/n_speakers`。可用 `MagicMock` + `assert_called_once_with` 补强。

#### P2-7 集成测试未覆盖 sliding 依赖 chunks 的边界
未测 VAD 返回空时 sliding 是否被跳过（对应 P1-4）。

---

## 三、测试结果意见

### 复跑验证
- **单测 7 passed 复现成功**：segmentation 6 测（2.94s）+ integration 1 测（7.60s），与 handoff "7 passed" 一致。
- **codegraph 调用链确认**：`cluster_sliding` 唯一生产调用者是 `_run_inference`，参数透传链 Form(489)→to_thread(519)→_run_inference(388)→cluster_sliding(450) 完整无断点。

### 对单场 29.76% DER 的评估

| 维度 | 评估 |
|---|---|
| 方向性 | ✅ 有效——CONF 从 48.3%→26.3%，证明"VAD 段粒度是 DER 虚高主因"的核心假设成立 |
| 可推广性 | ❌ 不足——仅 1 场 R8002_M8002，全量 20 场未跑，单场可能特例 |
| 部署代表性 | ❌ 偏乐观——评测传了 oracle-K（真实人数），自动估 K 未触发；29.76% 是上限，非零样本部署预期 |
| DER 口径 | ⚠️ 非标准——greedy 多对一映射 + 纯 Python 0.25s collar，与 pyannote 标准可能有差异，需交叉校验 |
| FA 副作用 | ⚠️ 需权衡——FA 从 0.3%→3.4%，是 None 填充 + 短窗切细的副作用，净 DER 仍降但错误模式变了 |
| 跨测试集 | ⚠️ 不可混——TEST_REPORT 的 32.18% 是 AISHELL-4 远场 spectral，29.76% 是 AliMeeting 近场 sliding，不同集不同策略。handoff 内部同场对比（sliding 29.76% vs spectral 49.21%）才公平 |

### 综合结论
**单场 29.76% 有方向性意义（CONF 显著下降验证了滑窗思路正确），但不能作为最终结论**，理由：①单场特例未全量验证；②oracle-K 上限非部署预期；③DER 口径非标准；④FA 上升副作用未量化。

---

## 四、建议优先级

| 优先级 | 行动 | 依据 |
|---|---|---|
| P0 | 跑全量 20 场（Task 6），同测试集对比 sliding vs spectral | 单场不可推广 |
| P0 | 补一组零样本评测（不传 num_speakers），量化自动估 K 退化 | 29.76% 是 oracle-K 上限 |
| P1 | 补 `pyannote.metrics` DER 交叉校验 | greedy 多对一偏乐观 |
| P1 | 修 P1-1：None 填充改静音标签，降 FA | FA 0.3%→3.4% 根因 |
| P1 | 修 P1-4：sliding 分支独立判断，不依赖 chunks | 设计实现矛盾，静默失效风险 |
| P2 | 修 P1-5/P1-6：docstring 补 sliding、der_eval choices 补 sliding | 维护性 |
| P2 | 补单测：None 填充 FA 断言、服务端传参断言、VAD 空边界 | 测试覆盖 |
| P2 | 全量后看 SPK4（19.7s）等少数说话人是否被漏检 | 短说话人召回 |

---

## 五、Handoff 质量评价

- **交接清晰度高**：来龙去脉、改动文件、设计决策、已知问题、复现命令齐全，codegraph 索引状态、conda 环境、端口、6GB 数据集 gitignore 均交代。
- **诚实标注局限**：自标 5 个 blockers（全量未跑、SPK4 漏检、FA 上升、抢话天花板、参数透传链待审），未夸大单场结果。
- **遗漏点**：①未明确 29.76% 是 oracle-K 上限（P1-3）；②未提 sliding 隐式依赖 chunks（P1-4）；③未提 der_eval choices 不含 sliding（P1-6）；④未提 greedy 多对一偏乐观（P1-7）。本审查补充。

> 总体：实现思路正确、单测可复现、handoff 交接质量高。主要风险在"单场 oracle-K 结果被误读为部署预期"以及 4 个 P1 代码问题。建议先跑全量 + 零样本评测再下最终结论。
