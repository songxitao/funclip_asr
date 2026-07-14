# 01 — 建立 ASR 封装与引擎库下沉

**What to build:**
完成 ASR 推理核心类与说话人引擎底层向主本地包 `src/funclip_pro` 的平移。外部无需感知内部物理路径变化，调用接口保持一致。

**Blocked by:**
None — 可以立即开始。

**Status:** ready-for-agent

- [ ] 在 `src/funclip_pro/core/` 目录下新建 `asr.py`，将 `SenseVoiceSmall` 推理类及其依赖类从 `asr_onnx_service.py` 中完整剥离移入。
- [ ] 将原有的 `speaker_engine.py` 移动至 `src/funclip_pro/core/speaker.py`；将原有的 `segmentation_engine.py` 移动至 `src/funclip_pro/core/segmentation.py`。
- [ ] 移除这些引擎文件中所有硬编码盘符路径，改由导入 `funclip_pro.config.loader` 调用 `resolve_model_path()` 动态获取模型权重路径。
- [ ] 编写测试 `tests/test_p1_engines.py`，利用 Mock 或测试配置实例化这三个核心引擎类，确保能够正常加载，没有任何 D:\ 或 E:\ 的路径硬编码报错。
