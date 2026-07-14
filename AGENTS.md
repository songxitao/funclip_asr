# 🤖 AGENTS.md (面向协同 Agent 的机器可读开发规约)

本规约旨在为新进入本项目的 AI 协同 Agent 提供物理接管与开发规范指导，使其在 3 秒内进入状态，零 Token 浪费。

---

## 1. ⚙️ 项目运行上下文 (Environment Context)
- **物理项目路径**：`E:\project\funclip-pro`
- **指定的 Python 路径**：`E:\conda\envs\asr_ui_env\python.exe`
- **DLL 点亮补丁**：服务在加载 CUDA 模型和 `onnxruntime` 推理时，依赖于 `asr_onnx_service.py` 顶部的动态 `os.add_dll_directory` 修改。请保留这些顶层设置，切勿在重构中将其遗漏。

---

## 2. 🏗️ 代码骨架与模块映射 (Symbol Map)

- [segmentation_engine.py](file:///E:/project/funclip-pro/segmentation_engine.py)
  - `SegmentationEngine` 类：加载本地 `segmentation-3.0` 模型，以 10s 无重叠分块推理，提供帧级 SAD 并过滤重叠帧，返回单人纯净段。
- [speaker_engine.py](file:///E:/project/funclip-pro/speaker_engine.py)
  - `CampPlusSpeaker` 类：
    - `cluster_with_segmentation` 方法：接收 `segment_engine` 获取分割段，逐段提 Cam++ Embedding，通过 `SpectralClustering` 全局聚类，并进行相邻段平滑合并。
- [asr_onnx_service.py](file:///E:/project/funclip-pro/asr_onnx_service.py)
  - `_get_seg_model()`：线程安全惰性加载 `SegmentationEngine`（优先 CUDA，失败回退 CPU）。
  - `_run_inference` 里的 `seg_clustering` 分支：处理 Diarization 并在毫秒级单位下回填转写 `text`，生成最终 segments。
- [der_eval.py](file:///E:/project/funclip-pro/der_eval.py)
  - 全局 DER 评测逻辑。
- [ali_der_eval.py](file:///E:/project/funclip-pro/ali_der_eval.py)
  - 针对 `AliMeeting` 近场单场 R8002_M8002 评测驱动。

---

## 3. 🚨 开发红线与 Gotchas

1. **设备 mismatch 防范**：在 `SegmentationEngine` 内部进行 powerset 多标签映射转换时，由于 `Powerset.mapping` 是 CPU 张量，必须在调用 `to_multilabel` 前将 powerset 矩阵转换为 CPU：`powerset.cpu()`。
2. **本地路径加载规避**：`pyannote.audio` 的 `from_pretrained` 传入目录会触发 `hf_hub_download` 解析 RepoID 崩溃。若路径为目录，必须拼接 `pytorch_model.bin` 传参。
3. **NumPy 2.x 拦截**：由于 `pyannote.audio` 底层调用 `np.NaN`，项目环境必须强力锁死在 `numpy==1.26.4`，请勿将其升级。
4. **时间戳单位防崩溃**：API 返回给前端以及 `ali_der_eval.py` 的时间单位必须是**毫秒 (ms)**。`cluster_with_segmentation` 返回的单位是**秒 (s)**，回填时必须乘以 1000。
5. **返回值解包匹配**：`_run_inference` 在任何分支中，都必须返回包含 `(raw_text, engine_key, segments, diarized_text)` 的四元组，请勿在分支内直接提前返回列表。

---

## 🧪 自动化测试与重启校验

- **单元测试**：
  ```bash
  chcp 65001 >$null; [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; E:\conda\envs\asr_ui_env\python.exe -m pytest tests/test_segmentation_engine.py tests/test_seg_clustering.py -v
  ```
- **服务状态与清理**：
  在启动或进行 API 测试前，可通过 `netstat -ano | findstr :8002` 查找 8002 的占用进程 PID 并使用 `taskkill /PID <PID> /F` 清理，保障端口可用。
- **服务启动**：
  ```bash
  E:\conda\envs\asr_ui_env\python.exe asr_onnx_service.py
  ```

## 4. 📦 SDK 包结构（P1 算法下沉落点）

- `src/funclip_pro/` 为统一算法 SDK 包，P1 起根目录算法按职责下沉：
  - `core.segmentation` ← `SegmentationEngine`（segmentation_engine.py）
  - `core.speaker` ← `CampPlusSpeaker`（speaker_engine.py）
  - `core.asr` ← `SenseVoiceSmall`(ONNX) / `PyTorchSenseVoice` / `SherpaSenseVoice` + 解码工具（asr_onnx_service.py / torch_engine.py / sherpa_engine.py）
  - `core.alignment` ← 子句说话人分配对齐（锚点扩散）
  - `utils.srt` ← SRT 转换 + VAD 段内合并
  - `pipeline.offline` ← `OfflinePipeline` 统一转写流水线
- 模块间绝对导入：`from funclip_pro.core.x import Y`，禁止相对导入越级。
- 红线（派发子智能体必带）：numpy 锁 1.26.4；时间戳 ms；`_run_inference` 返回四元组；`powerset.cpu()` 在 `to_multilabel` 前；`apply_dll_patch()` 保活；零硬编码盘符。
