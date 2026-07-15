r"""asr_onnx_service.py 路径解耦校验（P0 收尾，不依赖 torch / funasr / GPU）。

(a) resolve_model_path / PROJECT_ROOT 解析正确，且 FUNCLIP_MODEL_ROOT 整体覆盖生效。
(b) asr_onnx_service.py 源码已无任何 Windows 绝对盘符硬编码字面量（r"E:\... / r"D:\...）。
"""
import pathlib
import re
import sys

import pytest

# 把项目根的 src 加入路径以导入 funclip_pro
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from funclip_pro.config.loader import PROJECT_ROOT, resolve_model_path  # noqa: E402


def test_resolve_segmentation_default():
    """默认相对项目根的模型路径解析正确。"""
    p = resolve_model_path("models/damo/segmentation-3.0")
    assert p == PROJECT_ROOT / "model" / "models" / "damo" / "segmentation-3.0"


def test_resolve_sensevoice_small_default():
    """其余被解耦的相对路径也应指向 model/models 下对应目录。"""
    assert resolve_model_path("models/iic/SenseVoiceSmall-ONNX") == (
        PROJECT_ROOT / "model" / "models" / "iic" / "SenseVoiceSmall-ONNX"
    )
    assert resolve_model_path("models/iic/SenseVoiceSmallOnnx") == (
        PROJECT_ROOT / "model" / "models" / "iic" / "SenseVoiceSmallOnnx"
    )
    assert resolve_model_path("models/damo/speech_campplus_sv_zh-cn_16k-common") == (
        PROJECT_ROOT / "model" / "models" / "damo" / "speech_campplus_sv_zh-cn_16k-common"
    )


def test_resolve_model_root_override(tmpdir, monkeypatch):
    """FUNCLIP_MODEL_ROOT 环境变量应整体覆盖 model_base（Docker / 换机部署）。"""
    monkeypatch_dir = str(tmpdir)
    monkeypatch.setenv("FUNCLIP_MODEL_ROOT", monkeypatch_dir)
    p = resolve_model_path("models/damo/segmentation-3.0")
    assert p == pathlib.Path(monkeypatch_dir) / "models" / "damo" / "segmentation-3.0"
    assert p != PROJECT_ROOT / "model" / "models" / "damo" / "segmentation-3.0"


def test_service_source_has_no_hardcoded_drive_paths():
    """asr_onnx_service.py 不应再含 Windows 绝对盘符硬编码字面量。"""
    src = (ROOT / "asr_onnx_service.py").read_text(encoding="utf-8")
    # 匹配形如 r"E:\... / r"D:\... 的 raw-string 盘符字面量
    hardcoded = re.compile(r'r"[A-Za-z]:\\')
    matches = hardcoded.findall(src)
    assert not matches, f"仍存在硬编码盘符路径字面量: {matches}"
