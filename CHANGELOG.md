# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] - 2026-07-15

### Added
- Apache 2.0 LICENSE 开源许可证文件
- CHANGELOG.md 版本记录（Keep a Changelog 格式）
- CLI 控制台入口：`funclip-pro transcribe audio.wav --diarize`
- 配置文件模板 `config.json.example`，新用户克隆后可直接参考配置
- GitHub 发布 Ticket 管理体系：`.scratch/refactor-alignment/issues/` 下 13 项 tracer-bullet 任务拆解

### Fixed
- 混音单路数值溢出导致爆音，空帧无休眠导致 CPU 100% 自旋死循环（`core/audio.py`）
- ASR 模型路径硬编码，`--model_dir` 命令行参数失效（`core/streaming_asr.py`）
- Qwen HTTP 超时 1200s（20 分钟），宕机时界面假死闪退（`core/asr.py`）
- SileroVAD ONNX InferenceSession 析构泄漏（`core/streaming_asr.py`）
- pytest 收集阶段因 `TestClient(app)` 顶层实例化导致的联网卡死（改为 fixture 惰性加载）
- 测试代码中所有开发者物理路径硬编码，替换为动态路径或相对寻址

### Changed
- `MixedStream.read()` 单路分支补 `np.clip(..., -1.0, 1.0)`，空帧分支 `time.sleep(0.032)` 降频
- `QwenEngine.transcribe()` 超时改为 `(2.0, 15.0)`，分类捕获网络异常
- `SileroVAD` 新增 `__del__` 显式释放 `self.session`
- `pyproject.toml` 补全元数据（authors/license/readme/urls/scripts），补充缺失依赖（websockets/requests/pyaudio/pyaudiowpatch）
- `tests/` 重组为三级目录：`unit/`（默认执行）、`integration/`（标记 `@slow`）、`archive/`（完全跳过）
- 默认 `pytest` 仅执行 10 个单元测试文件，3 秒内完成
- `app_control.py` 物理机回退路径降级为系统 PATH 调用，消除其他用户闪退隐患
- 一键启动批处理脚本（`.bat`）移除硬编码盘符，改用 `%~dp0` 相对路径
- `.gitignore` 补充 `.scratch/`、`.pilot_venv/`、打包产物排除规则

### Removed
- 废弃的 `sherpa_engine.py` / `torch_engine.py` 对应测试归档至 `tests/archive/`
- 因架构重构失效的 13 个旧测试文件归档隔离

## [0.7.0] - 2026-07-14

### Added
- `core/audio.py` 音频采集硬件抽象层，支持 LoopbackStream（声卡环回）、MicStream（麦克风）、MixedStream（双源混音）
- `core/streaming_asr.py` 流式 ASR 引擎，集成 FunAsrStreamingEngine + SileroVAD + FSMN VAD
- QwenEngine Docker ASR 网络集成，支持 LLM-ASR 高精度转写
- 打字预览功能（实时桌面字幕逐字显示）
- VAD Session 隔离机制，提升流式场景稳定性

### Changed
- `app_live_local.py` 实时桌面字幕应用重构，从 944 行精简至 282 行（-662 行）
- `app_live_ws.py` WebSocket 实时客户端重塑，对接 `core.audio`，体积减少 35%
- `app_control.py` Gradio 离线界面集成 OfflinePipeline，Qwen3/SeACo/Nano 引擎解禁
- `asr_onnx_service.py` FastAPI 离线转写服务从 350 行精简至 169 行（-181 行）

## [0.6.0] - 2026-07-12

### Changed
- Gradio 离线应用进程内归口：重构 `app_control.py` 离线转写，废除 `subprocess.Popen` CMD 弹窗调用
- 改为 Python 进程内惰性加载延迟实例化，直接调用核心 `OfflinePipeline.run()`，进程内完成文本与 SRT 物理写入
- 完全兼容原有输出结果

### Removed
- 物理删除孤立冗余的 `funclip/asr1.py`（119KB） 及 `funclip/launch.py` 及其依赖
- 物理清除 `funclip/asr1 copy.py` 等 5 个 0 引用的历史备份文件

### Fixed
- 同步修复 `test_seg_seamless.py` 等测试文件的下沉导入问题，56 个核心用例 55 PASSED、1 SKIPPED

## [0.5.0] - 2026-07-09

### Added
- PEP-517 构建系统：新增 `pyproject.toml`，支持 `pip install -e .` 一键安装
- 双分支 DER 精度验证门禁：Git 分支回退跑测旧代码 vs P1 算法，验证零精度退化
- `numpy==1.26.4` 红线锁死依赖

### Changed
- 根目录"真相源唯一"清理：物理删除已被包化的 `segmentation_engine.py`（321 行）、`speaker_engine.py`（567 行）、`asr_service.py`（8.7KB）
- 11 处测试 import 同步更新至 `funclip_pro.core.*`
- 3 个外壳脚本移除 `sys.path.insert` 动态挂载

### Fixed
- `der_eval.py` 新增 `_normalize_stem()` 自动剥离 Ali 数据集 `_mixed`/`_near`/`_far` 后缀
- 0 匹配时 `exit(1)` 终止，杜绝静默 DER=0.0% 假绿

### Removed
- 5 个在模块顶层替换 `sys.stdout`/`sys.stderr` 的致命测试转移至 `tests/archive/`

## [0.3.0] - 2026-07-06

### Added
- SRT 字幕输出格式（`response_format=srt`）
- 相邻同说话人自动合并（JSON/text/SRT 统一受益）
- CLI 新增 `--format {json,text,srt}` 参数

### Changed
- 合并限制在 VAD 段内部（不跨段），防止解说与角色内容混淆

## [0.2.0] - 2026-07-04

### Added
- seg-3.0 输出所有帧段（含重叠/静音），构建无缝说话人时间轴
- 锚点扩散逻辑回收 seg 丢弃段
- `cli_transcribe.py` 命令行客户端

### Changed
- DER 从 15.13% 降至 14.54%
- MISS 从 11.3% 骤降至 0.7%

## [0.1.0] - 2026-07-02

### Added
- Segmentation-3.0 帧级单人提取引擎（10s 分块、约 17ms 帧级检测）
- Cam++ 纯净声纹 Embedding 提取
- SpectralClustering 全局谱聚类
- 说话人混淆错判率（CONF）从 46.8% 断崖式降至 3.8%
- DER 达到 15.13% 基准表现
