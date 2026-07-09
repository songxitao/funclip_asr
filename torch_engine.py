"""PyTorch (funasr AutoModel) 后端的 SenseVoice 推理引擎。

用于 CUDA 可用时的高吞吐 GPU 解码。接口对齐 SherpaSenseVoice:
- __init__(model_dir, device="cpu") 惰性构建 funasr.AutoModel
- 若 device=="cuda" 且 torch.cuda.is_available(): 把模型搬到 cuda
- __call__(waveforms: list[np.ndarray]) -> list[str]（已剥离 <|...|> 标签，过滤空串）

注意：funasr 的导入放在 __init__ 内，避免模块顶层触发重型依赖加载；
模型也仅在实例化时加载，便于单测只验证接口而不强制加载 GPU 权重。
"""
from __future__ import annotations

import os
import re

import numpy as np
import torch


_LABEL_RE = re.compile(r"<\|.*?\|>")


class PyTorchSenseVoice:
    def __init__(self, model_dir: str, device: str = "cpu"):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"PyTorch SenseVoice 模型目录不存在: {model_dir}")

        # funasr 重型依赖延迟导入，避免顶层加载（便于单测轻量导入）
        from funasr import AutoModel

        self.model_dir = model_dir
        self.device = device
        self.model = AutoModel(
            model=model_dir,
            trust_remote_code=True,
            device="cpu",
            disable_update=True,
            disable_pbar=True,
        )
        # CUDA 可用时把权重搬到 GPU（PyTorch-GPU 高吞吐路径）
        if device == "cuda" and torch.cuda.is_available():
            self.model.model.to("cuda")
            self.model.kwargs["device"] = "cuda"

    def __call__(self, waveforms):
        """接收单条波形或多条波形组成的 list[np.ndarray]（16k），返回清洗后的 list[str]。"""
        if not isinstance(waveforms, (list, tuple)):
            waveforms = [waveforms]
        waveforms = [np.asarray(w, dtype=np.float32) for w in waveforms]

        res = self.model.generate(
            input=waveforms,
            batch_size_s=0,
            language="auto",
            use_itn=True,
        )

        out: list[str] = []
        for item in res:
            t = item.get("text", "") if isinstance(item, dict) else str(item)
            # 剥掉 <|...|> 标签（PyTorch 原始输出含语言/情感等标签）
            t = _LABEL_RE.sub("", t).strip()
            if t:
                out.append(t)
        return out
