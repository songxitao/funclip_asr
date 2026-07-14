"""TDD 沙箱测试：验证 app_control.py 的 Windows 绝对路径已解耦，且 loader 正确读取配置。

注意：本文件严禁 import app_control（会触发 gradio 与 CPU 亲和性）。
仅依赖 funclip_pro.config.loader + 源码文本读取。
"""
import re
import sys
from pathlib import Path
import tempfile

# 让 src 包可被导入（与 app_control.py 的 sys.path.insert 一致）
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import funclip_pro.config.loader as loader_mod
from funclip_pro.config.loader import load_config


def _restore_project_root():
    # PROJECT_ROOT 由 loader.py 的 __file__ 向上溯源 3 层得到
    loader_mod.PROJECT_ROOT = Path(loader_mod.__file__).resolve().parents[3]


def test_load_config_reads_offline_python_and_conda_root():
    """loader 应在 config.yaml 含 offline_python / conda_root 时返回这些值。

    用临时 yaml + monkeypatch PROJECT_ROOT，不触碰真实项目文件。
    """
    yaml_content = (
        "device: cuda\n"
        "conda_root: C:\\myconda\n"
        "offline_python: C:\\envs\\fp\\python.exe\n"
        "model_base: model\n"
    )
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "config.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        loader_mod.PROJECT_ROOT = Path(d)  # monkeypatch，指向临时 yaml
        try:
            cfg = load_config()
            assert cfg.get("conda_root") == "C:\\myconda"
            assert cfg.get("offline_python") == "C:\\envs\\fp\\python.exe"
        finally:
            _restore_project_root()


def test_app_control_no_hardcoded_windows_paths():
    """源码不得再包含 Windows 盘符 raw 字面量硬编码（正则断言）。"""
    src = Path("app_control.py").read_text(encoding="utf-8")

    # 通用：不得再出现形如 r"X:\ 的 raw 盘符字面量
    raw_drive = re.compile(r'r"[A-Za-z]:\\')
    matches = raw_drive.findall(src)
    assert not matches, f"仍存在 raw 盘符硬编码字面量: {matches}"

    # 具体两项原始字符串（P0 已改为转义字符串向后兼容）
    assert re.search(r'r"D:\\program files\\Miniconda"', src) is None
    assert re.search(r'r"E:\\conda\\envs\\funclip_final', src) is None
