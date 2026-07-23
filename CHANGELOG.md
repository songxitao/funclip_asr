# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.1] - 2026-07-23

### Changed
- **项目结构整理**：根目录 8 个散落脚本归入 `scripts/{web,live,cli,tools,bench}/`，同步修正内部路径引用
- **`.gitignore` 加固**：追加 ML 权重兜底（`*.safetensors, *.bin, *.onnx, *.pt, *.pth, *.h5, *.pkl`）、凭据文件（`.env, *.pem, *.key`）、数据文件（`*.csv, *.parquet, *.arrow`）三个安全区块
- **ruff lint 修复**：scripts/ 下 6 个文件的 14 处代码风格问题自动修复
- **pre-commit 配置**：ruff 检查限于 `src/`，排除工具脚本目录

### Removed
- **废弃测试清理**：删除 `tests/archive/`（17 个文件，1761 行废弃代码）
- **HANDOFF 文档迁移**：`HANDOFF.md`、`HANDOFF-REVIEW.md`、`docs/` 下 2 份 REVIEW 文档移入 `.handoffs/` 并加入 `.gitignore`

## [0.9.0] - 2026-07-18

### Added
- **SeACo Paraformer 引擎** — 四合一加载 SeACo+VAD+PUNC+SPK，内置句子级时间戳。`core/asr.py` 新增 `SeACoParaformer` 类
- **核心数据模型** — `core/models.py` 新增 `WordTimestamp` / `Segment` / `TranscriptionResult` dataclass。字级时间戳有结构化的落脚点，不再靠 dict 约定
- **Qwen VAD 直出** — Qwen 分支去掉 `_split_timestamps_to_segments` 二次拆分，VAD 段边界直接作为 SRT 时间戳。Qwen 词级 timestamps 落进 `Segment.words`
- **ASS 卡拉 OK 支持** — `utils/ass.py` 支持 word timestamps → 逐字高亮渲染
- **标准化 API 端点** — `asr_onnx_service.py` 新增 `POST /v1/transcribe`（Pydantic 请求/响应）、`GET /v1/health`
- **单元测试** — `tests/unit/test_models.py` 14 个测试用例

### Changed
- **架构变更**：`pipeline.run()` 返回值从 4-tuple 改为 `TranscriptionResult` dataclass。所有调用方同步适配
- **引擎路由**：`_select_engine()` 新增 `"seaco"` key。`funasr_mode`（SeACo/SenseVoice/Nano）从死选项变为实际生效
- **热词支持**：SeACo 模式支持 hotwords 传参（SenseVoice/Nano 暂不支持）

### Fixed
- **SeACo 说话人标注不受控** — 不勾 SPK 时 SeACo 分支仍返回 `[说话人X]` 标注（内置 spk_model 一直返回 speaker 信息）。修复：`diarize=False` 时剥离 speaker 字段

## [0.8.4] - 2026-07-18

### Fixed
- **P0 修复：Qwen 非VAD路径 SRT 全是单字碎片** — 非VAD路径直接用 `parse_qwen_timestamps()` 做字级映射，每个字单独一条 SRT（如"对"80ms、"一"160ms）。改为调用 `_split_timestamps_to_segments()` 聚合成句级 segments
- **P0 修复：标点全部丢失** — Qwen ForcedAligner 不返回标点时间戳，`_char_ts` 不含标点，SRT/ASS 全部无标点。改为从 `full_text` token 间隙检测标点字符，作为合成 token 注入 `_char_ts`
- **软标点过度切碎** — 逗号等软标点触发切分导致字幕过碎。对齐 FunClip 行为：软标点仅 Phase 1 检测，Phase 2 合并回去，只有硬标点（。？！!?；;）和 5s 超时才切
- **尾标点丢失** — 最后一段 `char_end` 不包含句末标点。收尾阶段扫描 `full_text` 末尾纯标点并追加
- **ASS 双层字幕** — `_segments_to_ass()` 每段输出 Default + Karaoke 两行导致重叠。去掉 Default 层，只保留 Karaoke
- **FunASR/SeACo 输出空 SRT** — `diarize=False` 时 `segments=[]`，`_segments_to_srt([])` 输出空文件。加 fallback 从 VAD chunks + ASR 文本构建无标注 segments
- **重复标点** — Qwen VAD 分支未 `_clean()` 就调用 `_post_punc()`，导致标点模型在已有标点的文本上又加一层。去掉 `_post_punc()`，直接用 Qwen ITN 标点
- **变量作用域 Bug** — `seg_meta`/`chunks` 只在 VAD 分支定义，`diarize=False` + 非VAD 路径访问未定义变量。提前初始化到两个分支

### Added
- `_split_timestamps_to_segments()` 两阶段智能切句函数（原子短语→智能凑句），供 `_build_srt()` 和 `OfflinePipeline.run` 复用
- Phase 1 原子短语：检测 token 末字符标点 + `full_text` token 间隙标点
- Phase 2 智能凑句：硬标点结算 / 5s 超时切半 / 软标点合并
- ASS 注入标点：在 Phase 1 将 between_text 标点作为合成 token 注入 `_char_ts`
- 单元测试 `test_split_timestamps.py`（10个 UT）：覆盖标点切分、gap 切分、5s 超时、空输入、offset、`_char_ts` 保留
- 完全重写 `test_qwen_vad_batch.py`（12个 UT + 3个 mock VAD 测试 + 3个集成测试）：覆盖三路径

### Performance
- 22 个单元测试全部通过，6 秒内完成

### Added
- 新增 3 个 WSL2 内存与性能诊断优先级 Ticket（`04-wsl2-memory-lock` P0、`05-cpu-hotspot-baseline` P1、`06-vllm-scheduler-overhead` P2）
- 新增 `test_qwen_engine_transcribe_batch_numpy` 单元测试，覆盖客户端 NumPy→Base64 单次批量路径

### Changed
- **架构变更**：客户端 `QwenEngine.transcribe_batch` 的 NumPy 入参路径从"异步 aiohttp 并发流式（每 chunk 一条 HTTP 到 `/v1/audio/transcribe_stream`）"彻底重写为"单次同步 `requests.post` 批量 Base64 到 `/v1/audio/batch_transcriptions`"
- 移除 `_transcribe_single_chunk_async` 及 `aiohttp`/`asyncio` 依赖（不再使用）
- `micro_batch_size` 经验锁定为 `64`（A/B 实验验证：8 导致端到端 Pipeline 超时，批量性能减半）
- `HANDOFF.md` 同步最新进度与架构决策

### Fixed
- **P0 修复**：配置 `.wslconfig memory=16GB` 锁死 WSL2 内存上限，消除宿主机 98% RAM 触发的 Windows 内核 Swap 抖动（原 System 进程占用 57.4% CPU），端到端 Pipeline 恢复稳定
- `return_timestamps=True` 硬编码确保客户端批量调用始终返回时间戳对齐片段

### Performance
| 指标 | v0.8.2（旧异步流式） | v0.8.3（新单次批量） | 提升 |
|------|---------------------|---------------------|------|
| 34min 端到端 Pipeline | 112.9s / 18.3x（旧） | **70.2s / 29.4x** | **+60.8%** |
| 单文件 34min 直推 | 85.4s / 24.2x（v0.8.1基线） | 55.0s / 37.0x | +55.3% |
| 批量10 共享卷直读 | 10.7s / 0.285 | **9.2s / 0.246** | +14.1% |

## [0.8.1] - 2026-07-17

### Added
- `benchmark_qwen_asr.py` 后端性能基准与吞吐量对比测试脚本
- 宿主机端 `run_bench.bat` 一键基准测试启动脚本

### Fixed
- 修复并解决了 Qwen3-ASR 在集成客户端中失效/假死的问题

### Changed
- 在宿主机 SDK 离线流水线（`OfflinePipeline`）中引入并集成了 VAD 音频自动切割机制，将长音频分割为小段以匹配 ASR 模型最大长度。
- 重构客户端 `QwenEngine.transcribe_batch` 引入宿主机与 Docker 容器间共享临时卷直读（Direct Path）传输机制，免去 Base64 传输与编解码开销，大幅节省大批量切片文件传输耗时。
- 微调容器内 vLLM 核心参数：将 `MAX_MODEL_LEN` 提升至 `4096` 解决长音频 Token 溢出报错；调优 `GPU_MEMORY_UTILIZATION=0.70` 并限制并发 `max_num_seqs=8`，留出 `2.4GB` 物理显存边界，彻底解决跨 PCIe 显存 Swap 导致的假死利用率。
- 收紧 Docker 创建容器参数，限制 `--cpus 8` 与 `--shm-size 2g`，解决 150 线程上下文切换引起的 CPU 锯齿过载。

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
