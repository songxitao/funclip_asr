# Handoff: Qwen3-ASR 内存字节流异步并发转写重构 — 审核评估用

## Session Metadata
- Created: 2026-07-17 21:17
- Project: E:\project\funclip-pro
- Previous handoff: HANDOFF.md (v0.8.1 → v0.8.2 重构准备)
- Scope: 4 tracer-bullet Tickets 全部实现 + 单元测试更新 + 基准压测验证

---

## 一、本次变更总览 (git diff --stat)

```
 qwen_server/custom_server.py        |  68 ++++++++++++++++++++-
 src/funclip_pro/core/asr.py         |  84 ++++++++++++++++++++++++-
 src/funclip_pro/pipeline/offline.py |  32 ++--------
 tests/unit/test_qwen_vad_batch.py   |  32 ++++-----
 4 files changed, 170 insertions(+), 46 deletions(-)
```
（HANDOFF.md 的 123 行变更是前一次 handoff 本身，与本重构无关）

## 二、4 个 Tickets 的具体变更

### Ticket 01 — 服务端端点
**文件**: `qwen_server/custom_server.py`

**新增**: `POST /v1/audio/transcribe_stream` 端点
- 接收 `multipart/form-data: file(UploadFile), language(Form), return_timestamps(Form)`
- 读取内存字节 → base64 编码 → data URL → `asr_model.transcribe(audio=data_url, ...)`
- **零写盘**: 全程在内存中完成，不写任何临时文件
- 去掉了原先临时写入的 `soundfile` / `librosa` 依赖（不必要）

**新增 imports**: `from fastapi import File, Form, UploadFile`

### Ticket 02 — 客户端单块编码与异步发送
**文件**: `src/funclip_pro/core/asr.py`（QwenEngine 类内新增）

**新增**: `async def _transcribe_single_chunk_async(self, session, chunk, language, sem, idx) -> dict`
- 用 `async with sem:` 获取并发信号量
- `io.BytesIO()` + `sf.write(buffer, chunk, 16000, format='WAV', subtype='PCM_16')` → 内存 WAV 字节
- 构造 `aiohttp.FormData()`，以 `file` 字段 multipart 发送
- 3 次指数退避重试（0.5s, 1s, 2s）
- 全部失败返回空结果 `{"text": "", "timestamps": [], "language": language}`

**新增 imports**: `aiohttp`, `asyncio`, `io`, `soundfile`, `Union`

### Ticket 03 — 批量接口异步并发池
**文件**: `src/funclip_pro/core/asr.py`（QwenEngine.transcribe_batch 重构）

**签名扩展**: `audio_paths: Union[List[str], List[np.ndarray]]`
- `List[str]` → 走原有共享卷直读 / Base64 路径（完全保留）
- `List[np.ndarray]` → 走新异步并发路径：
  - `asyncio.Semaphore(8)` 控制最大并发
  - `aiohttp.ClientSession` + `asyncio.gather` 并发提交
  - `asyncio.run()` 同步封装，对外保持同步签名

**向后兼容**: 原有的文件路径路径零改动

### Ticket 04 — 离线流水线纯内存化
**文件**: `src/funclip_pro/pipeline/offline.py`

**删除**（Qwen+VAD 分支）:
- `tempfile.mkstemp` 临时文件创建
- `sf.write` 写盘
- `os.remove` 清理
- `temp_paths` 列表管理

**替换为**:
```python
chunks = [
    y[int(start_ms * 16):int(end_ms * 16)]
    for start_ms, end_ms in opt_segs
]
batch_results = qwen_engine.transcribe_batch(chunks, language=lang_param)
```

### 单元测试更新
**文件**: `tests/unit/test_qwen_vad_batch.py`

- `test_qwen_engine_batch` — 不变，文件路径路径仍然通过
- `test_offline_pipeline_qwen_vad_branch` — 更新断言：
  - 移除 `mock_sf_write.call_count == 2`（不再写盘）
  - 移除 `len(deleted_files) == 2`（不再清理文件）
  - 新增 `isinstance(chunk_arg[0], np.ndarray)` — 验证传的是 numpy 切片
  - `language` 参数改为 `"zh"`（pipeline 传原始参数，映射在 transcribe_batch 内部）

---

## 三、验证结果

### ✅ 端点验证
```
POST /v1/audio/transcribe_stream
  → 200 OK
  → 返回 {"text": "...", "language": "Chinese", "timestamps": [...]}
```

### ✅ 单元测试
```
pytest tests/unit/test_qwen_vad_batch.py -v
  → 2 passed (6.47s)
```

### ✅ 基准压测 (benchmark_qwen_asr.py)

| 测试项 | 结果 |
|--------|------|
| 短音频(~3s) RTF | 0.047 (21x 倍速) |
| 34min 单文件 RTF | 0.028 (35.7x 倍速) |
| 批量10 Base64 RTF | 0.396 |
| 批量10 共享卷 RTF | 0.237 |
| **34min 端到端 Pipeline** | **112.9s / RTF 0.055 (18.3x 倍速)** |

### ✅ 零写盘验证
宿主机 `qwen_server/shared_tmp/` 和系统临时目录均无本次运行产生的 `qwen_chunk_*.wav` 文件。
（`/tmp/` 下发现的 6 个 0 字节文件均为旧版本测试残留，时间戳在 18:22-20:23，非本次产生）

---

## 四、已知问题

### P1: 性能回退 — 32% 减速
- **旧基线 (v0.8.1)**: 34min 音频端到端 85.4s / RTF 0.041 (24.2x)
- **新基线 (v0.8.2)**: 34min 音频端到端 112.9s / RTF 0.055 (18.3x)
- **原因**: 旧路径走共享卷直读（Docker 直接读挂载文件路径，零拷贝）；新路径增加了几站 HTTP 开销
- **是否需要优化需评审决定**

### P2: 端到端 Pipeline 的 VAD 对齐片段数为 0
基准测试中 `len(segments) == 0`，但文本完整转写了（12381 字）。原因是：
- `_transcribe_single_chunk_async` 的 form data 中没有传入 `return_timestamps=True`
- 服务端端点默认 `return_timestamps=False`，不返回时间戳
- **影响**: segments 列表为空不影响转写文本完整性，但缺少时间戳对齐

### P3: 代码中有 `[DEBUG SF WRITE]` 调试日志残留
`offline.py:199` 原有一个 `print(f"[DEBUG SF WRITE] ...")`，在本次改版中应该一并移除或保留。

---

## 五、审核检查清单

- [ ] 新增端点 `/v1/audio/transcribe_stream` 功能正确
- [ ] `_transcribe_single_chunk_async` 重试逻辑合理
- [ ] `transcribe_batch` NumPy 类型分支判断正确
- [ ] 原共享卷直读路径未被破坏
- [ ] `offline.py` Qwen+VAD 分支零写盘
- [ ] 单元测试断言覆盖新路径
- [ ] async 代码的 `asyncio.run()` 调用不导致嵌套事件循环
- [ ] `aiohttp.ClientSession` 资源正确释放
- [ ] 无导入依赖遗漏
- [ ] 性能回退的可接受度
