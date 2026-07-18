"""pytest 配置 — 统一重依赖桩 + collect_ignore。

两步走：
1. 在 sys.modules 中注册自动容错的空桩模块（torch / funasr / librosa ...）
   使 `import torch`、`mock.patch("funasr.AutoModel")`、scipy 在
   import 时检查 `torch.Tensor` 等操作不会因找不到属性而崩溃。
   桩使用 LazyStubModule：访问任何未定义的属性自动创建子桩。
2. 针对仍会崩溃的文件，用 collect_ignore 跳过。
"""

import sys
import types


class _LazyStubModule(types.ModuleType):
    """自动容错桩模块：访问未定义属性时自动创建子桩，永不抛 AttributeError。

    例如 torch.Tensor、torch.nn.functional、librosa.effects 等深度
    链式访问均会自动创建中间桩，不会崩溃。
    """

    def __getattr__(self, name: str):
        # 特殊处理公共属性
        if name == "__version__":
            return "0.0.0"
        if name == "__path__":
            return []
        # 私有属性不自动创建（让它们正常抛 AttributeError）
        if name.startswith("_"):
            raise AttributeError(name)
        # 自动创建子桩
        stub = _LazyStubModule(f"{self.__name__}.{name}")
        setattr(self, name, stub)
        return stub


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
        stub = _LazyStubModule(mod_name)
        # 预创建特定 mock 目标
        if mod_name == "funasr":
            stub.AutoModel = type("AutoModel", (), {})
        if mod_name == "torch":
            stub.cuda.is_available = lambda: False
            stub.set_num_threads = lambda x: None
        sys.modules[mod_name] = stub

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
# 注意：桩的 torch.tensor 是 _LazyStubModule（不可调用），
# `torch.tensor([1])` 抛 TypeError，也要捕获
try:
    import torch
    import funasr
    _ = torch.tensor([1])
    _ = funasr.AutoModel
except (ImportError, AttributeError, TypeError):
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
