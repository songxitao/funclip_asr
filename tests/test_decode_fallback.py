"""验证 _decode 的引擎路由与 PyTorch→Sherpa 失败回退逻辑（不加载任何模型）。

通过 monkeypatch 注入假引擎，直接驱动 asr_onnx_service._decode：
- torch 成功路径返回 torch 结果；
- torch 抛错时回退到 MODEL（Sherpa）并仍然返回文本；
- sherpa 直连路径返回 Sherpa 结果。
"""
import numpy as np
import asr_onnx_service as svc


class _FakeEngine:
    def __init__(self, returns):
        self.returns = returns
        self.called = False

    def __call__(self, waveforms):
        self.called = True
        return self.returns


WAVEFORMS = [np.zeros(1600, dtype=np.float32)]


def test_decode_torch_success(monkeypatch):
    torch_fake = _FakeEngine(["torch文本"])
    sherpa_fake = _FakeEngine(["sherpa文本"])
    monkeypatch.setattr(svc, "_get_torch_model", lambda: torch_fake)
    monkeypatch.setattr(svc, "MODEL", sherpa_fake)

    out = svc._decode("torch", WAVEFORMS)

    assert out == ["torch文本"]
    assert torch_fake.called and not sherpa_fake.called


def test_decode_torch_failure_falls_back_to_sherpa(monkeypatch):
    def _boom():
        raise RuntimeError("CUDA OOM simulated")

    sherpa_fake = _FakeEngine(["回退文本"])
    monkeypatch.setattr(svc, "_get_torch_model", _boom)
    monkeypatch.setattr(svc, "MODEL", sherpa_fake)

    out = svc._decode("torch", WAVEFORMS)

    # 关键断言：PyTorch 失败后端点依然返回文本（来自 Sherpa 兜底）
    assert out == ["回退文本"]
    assert sherpa_fake.called


def test_decode_sherpa_direct(monkeypatch):
    sherpa_fake = _FakeEngine(["sherpa文本"])
    monkeypatch.setattr(svc, "MODEL", sherpa_fake)

    out = svc._decode("sherpa", WAVEFORMS)

    assert out == ["sherpa文本"]
    assert sherpa_fake.called
