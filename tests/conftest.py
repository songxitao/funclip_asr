"""pytest 配置 — 统一重依赖桩 + collect_ignore。

两步走：
1. 在 sys.modules 中注册空桩模块（torch / funasr / librosa ...）
   使 `import torch`、`mock.patch("funasr.AutoModel")` 等操作
   不会因找不到模块而崩溃。桩模块不提供真实功能，仅占位。
2. 针对仍会崩溃的文件（导入后调用了桩不具备的功能），
   用 collect_ignore 跳过。
"""

import sys
import types

# ── 1. 注册重型依赖的空桩模块 ──────────────────────────────────────────
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
        stub = types.ModuleType(mod_name)
        # 预创建常见 mock 目标属性（如 mock.patch("funasr.AutoModel")）
        if mod_name == "funasr":
            stub.AutoModel = type("AutoModel", (), {})
        sys.modules[mod_name] = stub

# torch 的常见子模块属性，mock 可能间接引用
try:
    import torch  # noqa: F401
except ImportError:
    _torch_stub = types.ModuleType("torch")
    _torch_stub.cuda = types.ModuleType("torch.cuda")
    _torch_stub.cuda.is_available = lambda: False
    _torch_stub.set_num_threads = lambda x: None
    sys.modules["torch"] = _torch_stub

# ── 2. collect_ignore（针对桩无法处理的文件） ──────────────────────────

# 注意：不能用 `import torch` / `import fastapi` 做条件判断，
# 因为上面注册的桩让它们总能导入成功。
# 需要检查具体导致崩溃的子模块或真实行为。

collect_ignore: list[str] = []

# pyannote.audio — 我们的 pyannote 桩是空 ModuleType，不支持子包导入
# segmentation.py → from pyannote.audio import Model 会炸
try:
    import pyannote.audio  # noqa: F401
except ImportError:
    collect_ignore.extend([
        "unit/test_offline_pipeline_unit.py",
        "unit/test_qwen_vad_batch.py",
        "integration/test_segmentation_engine.py",
    ])

# funasr + torch — 真实调用 not supported by stubs
try:
    import torch
    import funasr
    _ = torch.tensor([1])
    _ = funasr.AutoModel
except (ImportError, AttributeError):
    collect_ignore.extend([
        "integration/test_onnx_decode_refactor.py",
        "integration/test_onnx_gpu.py",
    ])

# fastapi — stubs 不支持 fastapi.testclient
try:
    import fastapi.testclient  # noqa: F401
except (ImportError, AttributeError):
    collect_ignore.extend([
        "integration/test_asr_api.py",
        "integration/test_pytorch_route.py",
    ])

# asr_onnx_service — 根目录模块，不在 PYTHONPATH=src 中
try:
    import asr_onnx_service  # noqa: F401
except ImportError:
    collect_ignore.extend([
        "integration/test_pytorch_inference_refactor.py",
    ])
