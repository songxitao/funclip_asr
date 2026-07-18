"""config loader 单测（P0 试点，不依赖真实模型 / GPU）。"""
import pathlib
import sys


# 把项目根的 src 加入路径以导入 funclip_pro
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import funclip_pro.config.loader as L  # noqa: E402


def test_project_root_resolves():
    expected = pathlib.Path(__file__).resolve().parents[2]
    assert L.PROJECT_ROOT.resolve() == expected


def test_resolve_model_path_default():
    p = L.resolve_model_path("models/damo/x")
    assert p == L.PROJECT_ROOT / "model" / "models/damo/x"


def test_resolve_model_path_env_override(monkeypatch):
    monkeypatch.setenv("FUNCLIP_MODEL_ROOT", "/tmp/alt_models")
    p = L.resolve_model_path("models/damo/x")
    assert p == pathlib.Path("/tmp/alt_models") / "models/damo/x"


def test_apply_dll_patch_non_win32(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert L.apply_dll_patch() is None


def test_load_config_fallback_without_yaml(monkeypatch):
    # 模拟 PyYAML 不可用：import yaml 将抛 ImportError
    monkeypatch.setitem(sys.modules, "yaml", None)
    cfg = L.load_config()
    assert cfg["model_base"] == "model"
    assert cfg["device"] == "cuda"
    assert "conda_root" in cfg
