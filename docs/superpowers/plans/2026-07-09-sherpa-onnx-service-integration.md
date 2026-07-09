# Sherpa-ONNX 接入 ASR 微服务 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把已验证胜出的 Sherpa-ONNX INT8 推理引擎真正接入线上微服务 `asr_onnx_service.py`，替换当前仍在使用、且 CPU 跑评反而更慢的 OpenVINO 版引擎，使 `/transcribe` 接口在保持 ITN+标点输出不变的前提下获得 ~54% 的 CPU 提速。

**Architecture:** 新增一个聚焦的 `sherpa_engine.py` 模块，定义 `SherpaSenseVoice` 类，内部用 `sherpa_onnx.OfflineRecognizer.from_sense_voice` 封装专为其量化的 `model.int8.onnx`，通过 C++ 批量接口 `decode_streams` 实现零 GIL 高并发 CPU 推理。其 `__call__` 接口刻意对齐原 OpenVINO 版 `SenseVoiceSmall.__call__`（接收波形/路径，返回字符串列表），因此只需改动 `asr_onnx_service.py` 的模型加载与标点后处理两处，业务链路（VAD 切分、`<|...|>` 标签清洗、FastAPI 接口）完全不动。

**Tech Stack:** Python 3.x, sherpa-onnx==1.13.4, FastAPI, funasr(AutoModel, 仅用于 VAD/PUNC), pytest。运行环境 `E:\conda\envs\asr_ui_env`。

---

## 背景与约束（接手前必读）

- **已验证结论**（见 `HANDOFF.md` 与 `README_PERF.md`）：在 6 线程 CPU 硬限制下，Sherpa-ONNX INT8 跑 7 分钟音频耗时 **10.52s / RTF 0.0248**，比 PyTorch FP32 CPU 快 **54%**，比 ORT/OpenVINO 快 **~200%**，去标点字符吻合度 **96.92%**。OpenVINO 在 CPU 上反而比 PyTorch 基准还慢（30.96s），定为死路。
- **绝不能踩的坑**：不要把 `model.int8.onnx` 送进 ORT CUDA（32.33s，比 PyTorch FP32 GPU 慢 4.17 倍）。本方案纯 CPU，与此无关，但务必保持 `use_itn=True` 以原生产出 ITN+标点。
- **已存在且必须继续通过的测试**：`tests/test_onnx_service_integration.py` 启动服务后 POST 音频，断言返回 JSON 含 `text` 字段且带中文标点（`，。？！`）。
- **已验证的 Sherpa API 形态**（来自 `tests/test_vad_sherpa_comparison.py`）：
  ```python
  recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
      model=r"...\SenseVoiceSmallOnnx\model.int8.onnx",
      tokens=r"...\SenseVoiceSmallOnnx\tokens.txt",
      num_threads=6,
      use_itn=True,        # 注意：不传 language 参数，沿用基准验证过的默认值
  )
  stream = recognizer.create_stream()
  stream.accept_waveform(16000, chunk)   # chunk 为 np.ndarray，采样率必须 16000
  recognizer.decode_streams([stream])
  text = stream.result.text               # 含标点与 ITN，无 <|...|> 标签（干净文本）
  ```
- **已知 gotcha**：`accept_waveform` 要求切片长度 ≥ 1600 采样点(0.1s)，否则报错。现有 `_run_inference` 中已有 `if len(chunk) < 1600: continue` 过滤，无需改动。
- **已实测验证（tests/verify_punc_sources.py，2026-07-09，VAD 切 `李雪花2.wav` → 85 chunks）**：
  - Sherpa-ONNX `use_itn=True` 原始输出：`含标点=True`，`含 <|...|> 标签=False`（已经是干净带标点文本）。
  - 原生 PyTorch `use_itn=True` 原始输出：`含标点=True`，`含 <|...|> 标签=True`（每段带 `<|zh|><|NEUTRAL|><|Speech|><|withitn|>`）。
  - 结论：Sherpa 路径下 `re.sub` 标签清洗为 no-op（输出无 `<|...|>` 标签），但**`PUNC_MODEL` 不应跳过**——Sherpa 逐段原生标点在 VAD 短片段上不可靠，应在剥掉原生标点后对拼接全文再跑一次 PUNC（见 Step 8）；PyTorch 原始输出带标签、且 ONNX/OpenVINO CTC 路径标点质量差，同样依赖 PUNC 补标点。三条路径统一为「剥原生标点 → 拼全文 → PUNC 一次」。

---

## File Structure

| 文件 | 操作 | 职责 |
|---|---|---|
| `sherpa_engine.py` | **Create** | 新增 `SherpaSenseVoice` 引擎类，封装 sherpa_onnx，接口对齐旧引擎 |
| `asr_onnx_service.py` | **Modify** | 1) 导入并改用 `SherpaSenseVoice`；2) `load_models` 加载 Sherpa INT8 模型；3) 标点后处理：两段（Sherpa / 旧 OpenVINO）都先剥原生标点，再对拼接全文跑一次 PUNC（Sherpa 短片段逐段标点不可靠） |
| `tests/test_sherpa_engine.py` | **Create** | 新引擎的单元测试（TDD 红→绿） |
| `tests/test_onnx_service_integration.py` | **Run (existing)** | 端到端验证接口仍返回带标点的正确 JSON |
| `HANDOFF.md` | **Modify** | 将 Pending Work 的 4 步标记为已完成 |

---

## Task 1: 编写 SherpaSenseVoice 引擎的失败单元测试

**Files:**
- Create: `tests/test_sherpa_engine.py`

- [ ] **Step 1: 写出失败测试**

```python
import os
import librosa
import numpy as np
import pytest

from sherpa_engine import SherpaSenseVoice

SHERPA_MODEL_DIR = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmallOnnx"
AUDIO_PATH = r"E:\下载\下载\李雪花2.wav"

pytestmark = pytest.mark.skipif(
    not os.path.exists(SHERPA_MODEL_DIR) or not os.path.exists(AUDIO_PATH),
    reason="需要 Sherpa INT8 模型与李雪花2.wav 测试音频",
)


@pytest.fixture(scope="module")
def engine():
    return SherpaSenseVoice(SHERPA_MODEL_DIR, num_threads=6, use_itn=True)


def test_engine_instantiates(engine):
    assert engine.recognizer is not None
    assert engine.use_itn is True


def test_call_returns_single_string_for_clip(engine):
    wav, _ = librosa.load(AUDIO_PATH, sr=16000)
    clip = wav[: 16000 * 10]          # 取前 10 秒，避免依赖完整长音频
    result = engine([clip])           # 传入单条波形组成的 list
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], str)
    assert len(result[0].strip()) > 0


def test_call_with_file_path(engine):
    result = engine(AUDIO_PATH)       # 传入文件路径字符串
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(t, str) for t in result)


def test_decode_streams_batch_count(engine):
    wav, _ = librosa.load(AUDIO_PATH, sr=16000)
    chunks = [wav[0:16000*5], wav[16000*5:16000*10], wav[16000*10:16000*15]]
    result = engine(chunks)           # 批量 decode_streams 应返回同数量结果
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(t.strip() for t in result)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd E:\project\funclip-pro && E:\conda\envs\asr_ui_env\python.exe -m pytest tests/test_sherpa_engine.py -v`
Expected: **FAIL**，`ModuleNotFoundError: No module named 'sherpa_engine'`（引擎尚未实现）

---

## Task 2: 实现 SherpaSenseVoice 引擎

**Files:**
- Create: `sherpa_engine.py`

- [ ] **Step 3: 写出最小实现**

```python
"""Sherpa-ONNX 后端的 SenseVoice 推理引擎。

用 sherpa_onnx.OfflineRecognizer 封装专为其量化的 INT8 模型 (model.int8.onnx)，
通过 C++ 批量接口 decode_streams 实现零 GIL 的高并发 CPU 推理。

接口对齐原 OpenVINO 版 SenseVoiceSmall：
- __call__(wav_content, language=[0], textnorm=[15], tokenizer=None) -> List[str]
- wav_content 可为音频文件路径(str) 或多条波形组成的 list[np.ndarray]
"""
from __future__ import annotations

import os

import librosa
import numpy as np
import sherpa_onnx


class SherpaSenseVoice:
    def __init__(
        self,
        model_dir: str,
        model_file: str = "model.int8.onnx",
        tokens_file: str = "tokens.txt",
        num_threads: int = 6,
        use_itn: bool = True,
    ):
        model_path = os.path.join(model_dir, model_file)
        tokens_path = os.path.join(model_dir, tokens_file)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Sherpa ONNX 模型不存在: {model_path}")
        if not os.path.exists(tokens_path):
            raise FileNotFoundError(f"Sherpa 词表不存在: {tokens_path}")

        # 注意：不传 language 参数，沿用基准评测验证过的默认行为
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_path,
            tokens=tokens_path,
            num_threads=num_threads,
            use_itn=use_itn,
        )
        self.sample_rate = 16000
        self.use_itn = use_itn

    def load_data(self, wav_content, fs=None):
        """对齐原接口：支持文件路径(str) 或 波形列表(list[np.ndarray])。"""
        target_sr = fs or self.sample_rate
        if isinstance(wav_content, str):
            wav, _ = librosa.load(wav_content, sr=target_sr)
            return [wav]
        if isinstance(wav_content, (list, tuple)):
            out = []
            for item in wav_content:
                if isinstance(item, np.ndarray):
                    out.append(item)
                else:
                    wav, _ = librosa.load(item, sr=target_sr)
                    out.append(wav)
            return out
        # 单条波形
        return [wav_content]

    def __call__(self, wav_content, language=None, textnorm=None, tokenizer=None, **kwargs):
        waveforms = self.load_data(wav_content, self.sample_rate)
        streams = []
        for wav in waveforms:
            stream = self.recognizer.create_stream()
            stream.accept_waveform(self.sample_rate, wav)
            streams.append(stream)
        self.recognizer.decode_streams(streams)
        # 保留原始富文本（含 <|...|> 标签与标点），由 asr_onnx_service 按原约定清洗标签
        return [s.result.text.strip() for s in streams]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd E:\project\funclip-pro && E:\conda\envs\asr_ui_env\python.exe -m pytest tests/test_sherpa_engine.py -v`
Expected: **PASS**（4 个用例全绿）

- [ ] **Step 5: 提交**

```bash
git add sherpa_engine.py tests/test_sherpa_engine.py
git commit -m "feat: add SherpaSenseVoice engine wrapping sherpa-onnx INT8 model"
```

---

## Task 3: 将引擎接入 asr_onnx_service.py

**Files:**
- Modify: `asr_onnx_service.py`

- [ ] **Step 6: 增加导入（在 `sys.path.append(...SenseVoiceSmall)` 之后）**

替换原 `from utils.model_bin import SenseVoiceSmallONNX` 附近的依赖引入，追加：

```python
import os
# 确保项目根目录在 sys.path，便于导入 sherpa_engine
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sherpa_engine import SherpaSenseVoice
```

> 说明：原 `from utils.model_bin import SenseVoiceSmallONNX` 及 `class SenseVoiceSmall(SenseVoiceSmallONNX)` 保留不删（OpenVINO 版作为参考/回退），但默认不再实例化。

- [ ] **Step 7: 改写 `load_models` 中的 ASR 加载块**

将 `load_models()` 内第 1 步（原 `MODEL = SenseVoiceSmall(...)` 整段）替换为：

```python
    try:
        # 1. 加载 ASR（Sherpa-ONNX INT8 后端，CPU 终极提速方案，已评测验证）
        sherpa_model_dir = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmallOnnx"
        MODEL = SherpaSenseVoice(
            model_dir=sherpa_model_dir,
            num_threads=6,
            use_itn=True,
        )
        logger.info("Sherpa-ONNX ASR 模型加载成功！")
```

其余 VAD_MODEL / PUNC_MODEL 加载保持不变。

- [ ] **Step 8: 标点后处理——剥掉 Sherpa 逐段原生标点，再对拼接全文跑一次 PUNC**

VAD 切分 + 合并 ≤8s 后，Sherpa `decode_streams` 会**逐段**输出自带标点 + ITN 的文本。
但短片段（尤其被 VAD 从句子中间切断的）逐段标点不可靠：标点模型没有整句上下文，
会在片段边界强行断句（见 `tests/verify_punc_sources.py` 讨论）。因此采用与现有
`asr_onnx_service.py` 第 295–304 行一致的做法——**先剥掉每段原生标点（保留 ITN 数字），
拼接成全文，再对全文跑一次 PUNC_MODEL**，让标点模型在整句上下文里决定断句。

新增一个小工具（放在 `sherpa_engine.py` 或 `_run_inference` 顶部）：

```python
import re as _re
_PUNC_RE = _re.compile(r"[，。！？；：、…—·「」『』“”‘’（）《》〈〉【】\[\]\(\)\{\}\"'\.,!?;:\s]")
def strip_punctuation(s: str) -> str:
    # 只移除标点，保留 ITN 已规整的数字与汉字
    return _PUNC_RE.sub("", s).strip()
```

在 `_run_inference` 的第 3 步（`raw_text = "\n".join(clean_texts)`）改为：

```python
        texts = MODEL(chunks)                       # Sherpa 逐段输出（含标点 + ITN）
        clean_texts = []
        for t in texts:
            t = re.sub(r"<\|.*?\|>", "", t).strip()  # 标签清洗（Sherpa 为 no-op）
            t = strip_punctuation(t)                # 剥掉原生标点，保留 ITN 数字
            if t:
                clean_texts.append(t)

        raw_text = "\n".join(clean_texts)            # 拼成全文

        # 4. 标点模型对全文跑一次（整句上下文，断句最准）
        if PUNC_MODEL is not None and raw_text.strip():
            try:
                punc_out = PUNC_MODEL.generate(input=raw_text)
                if punc_out and len(punc_out) > 0:
                    raw_text = punc_out[0].get('text', raw_text)
            except Exception as punc_err:
                logger.error(f"标点符号后处理失败: {punc_err}")
```

（`SherpaSenseVoice` 仍保留 `use_itn=True`；旧 OpenVINO 类路径输出本就弱标点，
剥标点后跑 PUNC 同样适用，两条路径统一为「剥标点 → 拼全文 → PUNC 一次」。）

> 说明：当前 `asr_onnx_service.py` 第 295 行 `raw_text = "\n".join(clean_texts)` 后
> 第 298 行直接 `PUNC_MODEL.generate(input=raw_text)`——它**已经是对拼接全文跑一次 PUNC**，
> 只是没剥 SenseVoice 的原生标点。Sherpa 路径补上「剥标点」这一步即可，结构不变。

- [ ] **Step 9: 运行单元测试确认仍通过**

Run: `cd E:\project\funclip-pro && E:\conda\envs\asr_ui_env\python.exe -m pytest tests/test_sherpa_engine.py -v`
Expected: **PASS**

- [ ] **Step 10: 提交**

```bash
git add asr_onnx_service.py
git commit -m "feat: wire SherpaSenseVoice into ASR microservice, re-punctuate full text via PUNC after stripping chunk punctuation"
```

---

## Task 4: 端到端集成验证

**Files:**
- Run: `tests/test_onnx_service_integration.py` (已存在，禁止修改其断言)

- [ ] **Step 11: 运行集成测试**

Run: `cd E:\project\funclip-pro && E:\conda\envs\asr_ui_env\python.exe -m pytest tests/test_onnx_service_integration.py -v -s`
Expected: **PASS** —— 服务以 Sherpa 引擎启动，`/transcribe` 返回 200，且 `text` 字段非空并含中文标点（`has_punc` 断言通过）。

> 注意：该测试会真实启动服务并加载模型（默认等待 15s），需 `E:\下载\下载\李雪花2.wav` 存在。若环境就绪，运行即验证迁移成功。

- [ ] **Step 12: 提交（如集成测试有配套微调）**

若集成测试通过且无代码改动，跳过提交；若需微调，提交后继续。

---

## Task 5: 收尾文档更新

**Files:**
- Modify: `HANDOFF.md`

- [ ] **Step 13: 更新 HANDOFF.md 的 Pending Work**

将 `## Pending Work → ### Immediate Next Steps` 下的 4 条改为已完成（打勾），并在 `### Tasks Finished` 增加：

```markdown
- [x] 将 `sherpa_onnx.OfflineRecognizer` 接入 `asr_onnx_service.py` 微服务
- [x] 服务模型目录切换为 `model/models/iic/SenseVoiceSmallOnnx`
- [x] 接口 ASR 推理改走 `recognizer.decode_streams` 批量并发
- [x] `tests/test_onnx_service_integration.py` 验证接口仍返回带 ITN+标点的正确 JSON
```

- [ ] **Step 14: 提交并推送（如适用）**

```bash
git add HANDOFF.md
git commit -m "docs: mark Sherpa-ONNX service integration complete in HANDOFF"
```

---

## Self-Review

**1. Spec coverage**
- ✅ 引入 `sherpa_onnx.OfflineRecognizer` → Task 2 / Task 3 Step 7
- ✅ 模型目录切到 `SenseVoiceSmallOnnx` → Task 3 Step 7
- ✅ 推理改走 `decode_streams` → `sherpa_engine.py` `__call__`
- ✅ 集成测试仍返回 ITN+标点 JSON → Task 3 Step 8（剥掉 Sherpa 逐段原生标点，对拼接全文跑一次 PUNC）+ Task 4
- ✅ TDD 红绿 + 频繁提交 → 每 Task 均含测试与 commit

**2. Placeholder scan**
- 无 "TBD / TODO / 待补充 / 类似 Task N"；所有代码步均给出完整代码与确切路径、确切命令与期望输出。

**3. Type / 接口一致性**
- `SherpaSenseVoice.__call__` 返回 `List[str]`（含 `<|...|>` 标签+标点），与原 `SenseVoiceSmall.__call__` 返回形态一致 → `_run_inference` 中 `MODEL(chunks)` / `MODEL(audio_path)` 调用与 `re.sub(r"<\|.*?\|>", "", t)` 清洗逻辑无需改动。
- `use_itn` 属性在两处一致使用（`sherpa_engine.py` 定义、`asr_onnx_service.py` 通过 `getattr(MODEL, "use_itn", False)` 读取）。
- 采样率恒为 16000，`accept_waveform(16000, wav)` 与现有 VAD 切片的 16kHz 波形一致。
