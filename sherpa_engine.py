"""Sherpa-ONNX 后端的 SenseVoice 推理引擎。

用 sherpa_onnx.OfflineRecognizer 封装专为其量化的 INT8 模型 (model.int8.onnx)，
通过 C++ 批量接口 decode_streams 实现零 GIL 的高并发 CPU 推理。

接口对齐原 OpenVINO 版 SenseVoiceSmall：
- __call__(wav_content) -> List[str]
- wav_content 可为音频文件路径(str) 或多条波形组成的 list[np.ndarray]
"""
from __future__ import annotations

import os

import numpy as np
import sherpa_onnx


class SherpaSenseVoice:
    def __init__(
        self,
        model_dir: str,
        num_threads: int = 6,
        use_itn: bool = True,
    ):
        model_path = os.path.join(model_dir, "model.int8.onnx")
        if not os.path.exists(model_path):
            # 回退到 fp32 模型
            model_path = os.path.join(model_dir, "model.onnx")
        tokens_path = os.path.join(model_dir, "tokens.txt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Sherpa ONNX 模型不存在: {model_path}")
        if not os.path.exists(tokens_path):
            raise FileNotFoundError(f"Sherpa 词表不存在: {tokens_path}")

        # 注意：不传 language 参数，沿用基准评测验证过的默认行为
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_path,
            tokens=tokens_path,
            num_threads=num_threads,
            use_itn=use_itn,
        )
        self.sample_rate = 16000
        self.use_itn = use_itn

    def __call__(self, wav_content):
        """接收单条文件路径(str) 或 多条波形组成的 list[np.ndarray]，返回 List[str]。

        - 对每条波形/path 创建独立 stream 并 accept_waveform；
        - 整个批次一次性 decode_streams（零 GIL 高并发）；
        - 收集每条 stream 的 result.text 原样返回（不剥 <|...|> 标签）。
        """
        import librosa

        if isinstance(wav_content, str):
            # 单条文件路径 -> 包装成单元素波形列表
            waveforms = [librosa.load(wav_content, sr=self.sample_rate)[0]]
        elif isinstance(wav_content, (list, tuple)):
            waveforms = list(wav_content)
        else:
            # 单条 numpy 波形
            waveforms = [wav_content]

        streams = []
        stream_index: list[int] = []  # 记录每个 stream 对应的输入下标
        for idx, wav in enumerate(waveforms):
            # Sherpa 要求切片长度 >= 1600 采样点(0.1s)，否则报错
            if not isinstance(wav, np.ndarray):
                wav = np.asarray(wav, dtype=np.float32)
            if len(wav) < 1600:
                continue  # 过短波形：在结果列表中映射为空串，保持对齐
            stream = self.recognizer.create_stream()
            stream.accept_waveform(self.sample_rate, wav)
            streams.append(stream)
            stream_index.append(idx)

        # 所有波形都过短时直接返回等长空串列表
        if not streams:
            return [""] * len(waveforms)

        self.recognizer.decode_streams(streams)
        out: list[str] = [""] * len(waveforms)
        for stream, idx in zip(streams, stream_index):
            out[idx] = stream.result.text
        return out
