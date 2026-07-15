import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient
import asr_onnx_service as asr_service
import inspect
from fastapi import Form
import pytest

pytestmark = pytest.mark.slow

def test_transcribe_route_default_vad():
    sig = inspect.signature(asr_service.transcribe)
    param = sig.parameters.get("vad_split")
    assert param is not None
    assert param.default.default is True
