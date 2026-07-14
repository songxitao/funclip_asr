"""配置加载器（P0 路径解耦试点）。

统一从项目根目录的 config.yaml 读取运行时配置，并提供基于 PROJECT_ROOT 的
动态相对路径解析，消除各文件里硬编码的 Windows 物理盘符绝对路径。

设计约束（来自 P0 spec / AGENTS.md）：
  - PROJECT_ROOT 由本文件 __file__ 向上溯源得到，绝不写死盘符。
  - PyYAML 为可选依赖：沙箱无网络 / 无 yaml 时回退内置 DEFAULTS。
  - apply_dll_patch 仅做降级容错：目录不存在或操作失败只 warning（add_dll_directory + PATH 前置双机制）。
"""
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# loader 位于 src/funclip_pro/config/loader.py
#   parents[0]=config, [1]=funclip_pro, [2]=src, [3]=项目根
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULTS = {
    "device": "cuda",
    "threads": None,
    "conda_root": None,          # 可留空，由 env 或平台默认推断
    "model_base": "model",       # 相对项目根的子目录
}


def load_config() -> dict:
    """读取 PROJECT_ROOT/config.yaml；无 yaml 或文件缺失时回退 DEFAULTS。"""
    cfg = dict(DEFAULTS)
    path = PROJECT_ROOT / "config.yaml"
    if path.exists():
        try:
            import yaml  # PyYAML 可选依赖
        except ImportError:
            logger.warning("PyYAML 未安装，使用内置默认配置（DEFAULTS）。")
            return cfg
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                cfg.update(loaded)
        except Exception as e:  # noqa: BLE001 - 降级容错，绝不中断
            logger.warning("读取 config.yaml 失败，回退默认配置: %s", e)
    return cfg


def resolve_model_path(rel: str) -> Path:
    """返回模型路径：默认 PROJECT_ROOT/<model_base>/<rel>。

    FUNCLIP_MODEL_ROOT 环境变量存在时整体覆盖 model_base
    （用于 Docker / 换机部署，把模型目录整体外挂）。
    """
    override = os.environ.get("FUNCLIP_MODEL_ROOT")
    if override:
        base = Path(override)
    else:
        base = PROJECT_ROOT / load_config()["model_base"]
    return base / rel


def _conda_site_packages_bases() -> list:
    """推断候选 conda 环境的 Lib/site-packages 基目录列表。"""
    bases = []
    cfg = load_config()
    if cfg.get("conda_root"):
        bases.append(cfg["conda_root"])
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        bases.append(prefix)
    root = os.environ.get("CONDA_ROOT")
    if root:
        bases.append(root)
    return bases


def apply_dll_patch() -> None:
    """Windows 平台补丁：把深度学习库的 DLL 目录加入搜索路径。

    等价抽取自 asr_onnx_service.py / tests/test_onnx_gpu.py 顶部的
    DLL 点亮逻辑（os.add_dll_directory + os.environ["PATH"] 前置拼接），
    改为通用、容错实现：目录不存在或操作失败仅 warning，不 raise。
    非 Windows 直接安全返回。

    双机制保留等价点亮：
      - os.add_dll_directory：现代 Windows DLL 搜索路径（主机制）
      - os.environ["PATH"] 前置：兼容旧式 LoadLibrary 解析（冗余保底）
    """
    if sys.platform != "win32":
        return

    candidates = [
        "ctranslate2",
        "nvidia/cudnn/bin",
        "onnxruntime/capi",
        "torch/lib",
    ]
    targets = []
    for base in _conda_site_packages_bases():
        sp = Path(base) / "Lib" / "site-packages"
        for cand in candidates:
            d = sp / cand
            if d.exists():
                targets.append(d)

    for d in targets:
        try:
            os.add_dll_directory(str(d))
        except Exception as e:  # noqa: BLE001 - 降级容错
            logger.warning("add_dll_directory 失败（已跳过）: %s -> %s", d, e)
        # 等价保留原 asr_onnx_service 的 PATH 前置点亮，兼容旧式 DLL 解析
        try:
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
        except Exception:  # noqa: BLE001 - 降级容错
            pass
