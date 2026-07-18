"""pytest 配置 — 统一重依赖桩 + collect_ignore。

两步走：
1. 在 sys.modules 中注册自动容错桩模块（torch / funasr / librosa ...）
   使 `import torch`、`onnxruntime.SessionOptions()`、scipy 在
   import 时检查 `torch.Tensor` 等操作不会因找不到属性而崩溃。
   桩使用 _StubModule：可调用、auto-creates 子属性。
2. 针对仍会崩溃的文件，用 collect_ignore 跳过。
"""

import sys
import types


class _StubModule(types.ModuleType):
    """自动容错桩：可调用、自动创建子属性、支持 issubclass 检查。

    特点：
    - `stub.SessionOptions()` → 返回 MagicMock（不崩溃）
    - `stub.nn.functional.relu` → 链式自动创建
    - `stub.Tensor` + `issubclass(cls, stub.Tensor)` → 正常工作
    - `_IS_STUB = True` 标记，collect_ignore 可靠检测
    """

    _IS_STUB = True

    def __getattr__(self, name: str):
        if name == "__version__":
            return "0.0.0"
        if name == "__path__":
            return []
        if name.startswith("_"):
            raise AttributeError(name)
        stub = _StubModule(f"{self.__name__}.{name}")
        setattr(self, name, stub)
        return stub

    def __call__(self, *args, **kwargs):
        from unittest.mock import MagicMock
        return MagicMock()


# ── 1. 注册重型依赖的自动容错桩模块 ──────────────────────────────────
_STUB_MODULES: list[str] = [
    "torch",
    "funasr",
    "librosa",
    "fastapi",
    "onnxruntime",
    "onnxruntime_gpu",
    "pyaudio",
    "pyaudiowpatch",
    "gradio",
    "uvicorn",
    "websockets",
    "pyannote",
    "sherpa_onnx",
    "openvino",
]

for mod_name in _STUB_MODULES:
    try:
        __import__(mod_name)
    except ImportError:
        stub = _StubModule(mod_name)
        # 预创建特定 mock 目标
        if mod_name == "funasr":
            stub.AutoModel = type("AutoModel", (), {})
        if mod_name == "torch":
            stub.Tensor = type("Tensor", (), {})  # issubclass 需要真实 type
            stub.cuda.is_available = lambda: False
            stub.set_num_threads = lambda x: None
        sys.modules[mod_name] = stub

# ── 2. collect_ignore（针对桩无法处理的文件） ──────────────────────────

import importlib.util as _util

# ...

collect_ignore: list[str] = []

# pyannote.audio — 桩不支持子包导入，segmentation.py → from pyannote.audio import Model
if _util.find_spec("pyannote.audio") is None:
    collect_ignore.extend([
        "unit/test_offline_pipeline_unit.py",
        "unit/test_qwen_vad_batch.py",
        "integration/test_segmentation_engine.py",
    ])

# torch — 桩有 _IS_STUB 标记
try:
    import torch
    if getattr(torch, '_IS_STUB', False):
        raise ImportError("torch is a stub")
except ImportError:
    collect_ignore.extend([
        "integration/test_onnx_decode_refactor.py",
        "integration/test_onnx_gpu.py",
    ])

# fastapi.testclient — stub not supported
if _util.find_spec("fastapi.testclient") is None:
    collect_ignore.extend([
        "integration/test_asr_api.py",
        "integration/test_pytorch_route.py",
    ])

# asr_onnx_service — 根目录模块，不在 PYTHONPATH=src 中
if _util.find_spec("asr_onnx_service") is None:
    collect_ignore.extend([
        "integration/test_pytorch_inference_refactor.py",
    ])
