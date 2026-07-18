"""pytest 配置 — 自动跳过需要重型 ML 依赖的测试文件。

在 CI 环境（无 torch / fastapi / funasr 等重型依赖）中，这些文件会在
导入时崩溃（import-time crash），不是运行时失败。因此不能用 @pytest.mark.slow，
而要用 collect_ignore 在测试发现阶段直接跳过。
"""

collect_ignore: list[str] = []

# ── torch / funasr ──────────────────────────────────────────────────
try:
    import torch  # noqa: F401
except ImportError:
    collect_ignore.extend([
        # 通过 funclip_pro.pipeline.offline → SegmentationEngine → torch 链崩溃
        "unit/test_offline_pipeline_unit.py",
        "unit/test_qwen_vad_batch.py",
        # 直接在模块级 import torch / funasr
        "integration/test_onnx_decode_refactor.py",
        "integration/test_onnx_gpu.py",
        "integration/test_segmentation_engine.py",
    ])

# ── fastapi（含 starlette / pydantic / uvicorn 等） ─────────────────
try:
    import fastapi  # noqa: F401
except ImportError:
    collect_ignore.extend([
        # import asr_onnx_service + fastapi.testclient
        "integration/test_asr_api.py",
        "integration/test_pytorch_route.py",
    ])

# ── asr_onnx_service（根目录模块，不在 PYTHONPATH=src 中） ──────────
# 即使 torch/fastapi 可用，asr_onnx_service.py 在 CI 的 PYTHONPATH=src
# 下也无法导入（它在项目根目录，不在 src/）。
try:
    import asr_onnx_service  # noqa: F401
except ImportError:
    collect_ignore.extend([
        "integration/test_pytorch_inference_refactor.py",
    ])
