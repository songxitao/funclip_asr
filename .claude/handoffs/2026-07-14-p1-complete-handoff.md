# Handoff: P1 算法 SDK 化 — 已完成（双轴自审 + 接手文档）

## Session Metadata
- Created: 2026-07-14 16:36
- Project: E:\project\funclip-pro
- Branch: refactor/p1-algo-packaging（W1-W5 已提交，**未**合入 main）
- Scope: 只读代码审查 + 写文档；**未**改 src/ 算法、**未** git commit/add、**未**启动服务/GPU

### P1 提交链（已确认）
- W1 `d2c0749` core 下沉（segmentation/speaker/asr/tokenization）
- W2 `04b9624` alignment/utils.srt 下沉
- W3 `3811c2c` OfflinePipeline 整合（返回四元组）
- W4 `1b5ca48` 薄路由（FastAPI）+ CLI 瘦身
- W5 `6a8cae9` 测试门禁（P1 相关单测 28 绿；DER 单场 seg_clustering=29.81%，与 P0 同文件 71.82%/32.2% 量级相当、非回归）

---

## 1. P1 做了什么

把根目录三大算法引擎与编排下沉为统一 SDK 包 `src/funclip_pro/`，`asr_onnx_service.py` 收缩为只做 FastAPI 薄路由 + 启动保活，推理编排全部委托 `OfflinePipeline`。根引擎逻辑**字节级等价**迁移，未做算法改动。

---

## 2. 目录结构与各模块职责

```
src/funclip_pro/
  config/loader.py        P0 已建：resolve_model_path / apply_dll_patch / load_config
  core/
    segmentation.py       SegmentationEngine（pyannote powerset 分割，10s 无重叠分块）
    speaker.py            CampPlusSpeaker（Cam++ 向量 + single/two_stage/spectral/vad_sliding 聚类
                           + cluster_with_segmentation / cluster_with_seamless_segmentation）
    asr.py                SenseVoiceSmall(ONNX)/PyTorchSenseVoice/SherpaSenseVoice + _decode/_clean/
                           _post_punc/_select_engine/_use_vad/_cheap_trim/_merge_vad_segments/load_models
    tokenization.py       CharTokenizer（字符级 token 还原）
    alignment.py          _assign_clauses_to_speakers / _seamless（子句→说话人锚点扩散对齐，单位 ms）
  utils/
    srt.py                _ms_to_srt / _merge_same_speaker_segments（VAD 段内不跨段）/ _segments_to_srt
  pipeline/
    offline.py            OfflinePipeline.run() → 四元组 (raw_text, engine_key, segments, diarized_text)
```

根薄壳：
- `asr_onnx_service.py`：FastAPI app + `/transcribe` 路由 + startup 保活；委托 `OfflinePipeline`，**不 import asr_service.py**，绝对导入 `from funclip_pro.x`。
- `cli_transcribe.py`：命令行客户端（已瘦身）。

---

## 3. 如何启动

```bash
# 服务（薄路由，:8002），等价原 main 启动方式
E:/conda/envs/asr_ui_env/python.exe asr_onnx_service.py

# CLI
E:/conda/envs/asr_ui_env/python.exe cli_transcribe.py <audio> --diarize --diarize_strategy seg_clustering

# 单测（P1 相关）
E:/conda/envs/asr_ui_env/python.exe -m pytest tests/test_offline_pipeline_unit.py tests/test_offline_pipeline_integration.py tests/test_routing.py -v
```

---

## 4. DER 评测现状与已知方法学缺陷

- **P1 门禁结果**：单场 AISHELL-4 `L_R003S01C02`（seg_clustering）= **29.81%**（test_results/der_single_seg_clustering.json）。与 P0 同文件 two_stage 71.82% / 阈值 32.2% 量级相当，**非回归**（同口径、单场 GPU 噪声）。
- **方法学缺陷（不可作系统水平参考）**：`der_eval.py` 抽 **ch0 单通道**，而 AISHELL-4 是**远场 8 通道阵列会议**——相当于丢弃阵列带来的说话人区分增益，Cam++ embedding 区分度极差。详见 `testset/CER_DER_TEST_REPORT.md §5`：
  - §5.2 真因：单通道输入 + 远场场景 + 聚类策略不匹配 → DER 失真，**非 ASR/说话人模型崩溃**。
  - 已排除：int('?') bug、Cam++ CUDA .numpy() 崩溃、VAD 漏检、参考分布极端。
- **结论**：当前 DER 数字只能证明"服务不崩、流程等价"，不能证明 diarization 真实水平。

---

## 5. 下一步建议（给后续高级模型/工程师）

1. **多通道 DER 评测**（首选，`CER_DER_TEST_REPORT.md §6`）：`der_eval.py` 的 `to_mono_ch0()` 改多通道——A 选最强/最佳 SNR 通道，B 通道 embedding 融合，C beamforming；保证 hyp/ref 时间轴对齐。
2. **换更鲁棒聚类后端**：用 pyannote AHC / SpectralClustering 替换 `two_stage`；或调 `two_stage` 阈值与 oracle-K 配合。
3. **embedding 融合**：远场下多通道 embedding 均值/拼接后聚类，提升区分度。

---

## 6. 回归风险点 / 待修项（双轴自审发现）

**轴A 正确性/等价性（高优先级）：无红线违反。** 已逐文件核对：
- seg_clustering 双分支保留（offline.py L155 seamless 分支 + L292 经典分支），与 main `_run_inference` L617/L744 字节级一致 ✓
- `powerset.cpu()` 均在 `to_multilabel` 前（segmentation.py L97、L260）✓
- `apply_dll_patch()` 保活三处：speaker.py L39、offline.py L26/L103、asr_onnx_service.py L34 ✓
- 时间戳秒×1000→ms：offline.py L168/287/298，alignment 输入即 ms ✓
- `_assign_clauses_to_speakers(_seamless)` 锚点扩散逻辑与 main L394/470 一致 ✓
- core/speaker.py 与 main speaker_engine.py 算法体 diff 仅 docstring/类型注解/dll-patch 差异，聚类逻辑等价 ✓

**轴B 规范/红线（高优先级）：无违反。** numpy==1.26.4 锁守（requirements.txt:6）；src/funclip_pro 下 grep 零硬编码盘符；模块绝对导入 `from funclip_pro.x`；薄路由只依赖 funclip_pro 不依赖 asr_service.py。

**中优先级（待修，未动手）：**
1. **双真源风险（未达成 issue#11 清单项）**：根 `segmentation_engine.py` / `speaker_engine.py` 仍是**完整算法副本**（非薄再导出），且 6+ 个测试（`test_segmentation_engine.py`、`test_seg_clustering.py`、`test_seg_seamless.py`、`test_sliding_segmentation.py`、`test_vad_sliding.py` 等）仍直接 import 根副本。P1 SDK 与根副本并存，若后续只改其一会分叉。等价性证据仅覆盖"服务路径"，未覆盖"核心测试路径"。→ 建议将这些测试迁移到 `funclip_pro.core`，并把根引擎文件改为薄再导出或删除。
2. **asr_service.py 为重复路由**：是与 asr_onnx_service.py 的近重复（第二个 FastAPI 服务），含**旧式 `_run_inference` 返回 str（非四元组）**，仅被 `tests/` 引用。→ 建议收敛为薄壳或删除，避免歧义。

**低优先级：**
3. `core/asr.py` import 期加载 torch/onnxruntime，独立 import 该模块的脚本须先调用 `apply_dll_patch()`（offline.py / asr_onnx_service.py 已在 import 前点亮，正常路径安全）。
4. 测试门禁仍绑定根引擎副本，建议后续随待修项①一并迁移，使门禁直接验证 SDK。

---

## 7. 接手提示
- 分支未合入 main，合入前建议先处理中优先级待修项①②（或至少确认根副本不改动）。
- 跑 DER 必须显式 `--diarize_strategy seg_clustering`，否则默认口径会得到错误高 DER 误判回归。
- 当前 DER 数字失真源于方法学（单通道+远场），不应据此判定 P1 是否回归。
