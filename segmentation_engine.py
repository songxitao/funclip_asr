"""pyannote segmentation-3.0 推理引擎。

职责：加载 segmentation-3.0 模型，对音频做帧级说话人活性检测，
切出非重叠的 homogeneous segments（单人纯净段）。

与 Cam++ (speaker_engine.py) 配合使用：
  segmentation 负责"在哪里切"，Cam++ 负责"这段是谁"。
"""
import logging
import os
import numpy as np
import torch
from typing import List, Tuple, Optional
from pyannote.audio import Model

logger = logging.getLogger(__name__)

DEFAULT_SEG_MODEL_DIR = r"E:\project\funclip-pro\model\models\damo\segmentation-3.0"

# segmentation-3.0 的配置参数（来自 config.yaml）
_CHUNK_DURATION_SEC = 10.0
_MAX_SPEAKERS_PER_CHUNK = 3
_MAX_SPEAKERS_PER_FRAME = 2
_SAMPLE_RATE = 16000
_CHUNK_SAMPLES = int(_CHUNK_DURATION_SEC * _SAMPLE_RATE)  # 160000


class SegmentationEngine:
    """pyannote segmentation-3.0 本地推理引擎。"""

    def __init__(self, model_dir: str = DEFAULT_SEG_MODEL_DIR, device: str = "cpu"):
        from pyannote.audio.utils.powerset import Powerset

        # GOTCHA: Model.from_pretrained 如果接收的是目录且该目录下存在 pytorch_model.bin
        # 应自动拼接路径传给 from_pretrained
        load_path = model_dir
        if os.path.isdir(model_dir):
            bin_path = os.path.join(model_dir, "pytorch_model.bin")
            if os.path.exists(bin_path):
                load_path = bin_path

        logger.info(f"[Segmentation] 加载 segmentation-3.0: {load_path} (device={device})")
        self.model = Model.from_pretrained(load_path)
        self.device = device
        self.model.to(torch.device(device))
        self.model.eval()

        self.to_multilabel = Powerset(
            num_classes=_MAX_SPEAKERS_PER_CHUNK,
            max_set_size=_MAX_SPEAKERS_PER_FRAME,
        ).to_multilabel

    def process_chunk(
        self,
        audio_np: np.ndarray,
        sr: int = 16000,
        threshold: float = 0.5,
        min_seg_sec: float = 0.3,
    ) -> List[Tuple[float, float, int]]:
        """对单个 ≤10s chunk 做 segmentation，提取非重叠 homogeneous segments。

        Args:
            audio_np: 单声道 16kHz 音频 numpy 数组
            sr: 采样率
            threshold: 说话人活性二值化阈值
            min_seg_sec: 最短 segment 时长（秒），低于此值丢弃

        Returns:
            List[(start_sec, end_sec, local_speaker_id)]
            local_speaker_id 是 chunk 内部 of 局部 ID (0/1/2)，跨 chunk 不一致。
            只包含非重叠的单人段。
        """
        actual_duration = len(audio_np) / sr

        # 不足 10s 的 chunk 用 zero-padding 补齐
        if len(audio_np) < _CHUNK_SAMPLES:
            padded = np.zeros(_CHUNK_SAMPLES, dtype=np.float32)
            padded[: len(audio_np)] = audio_np
            audio_np = padded

        # 截断超过 10s 的部分
        audio_np = audio_np[:_CHUNK_SAMPLES]

        # 准备输入: (batch=1, channels=1, samples)
        waveform = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            powerset = self.model(waveform.to(self.device))  # (1, F, 7)

        multilabel = self.to_multilabel(powerset)  # (1, F, 3)
        activity = multilabel[0].cpu().numpy()  # (F, 3)

        num_frames = activity.shape[0]
        frame_sec = actual_duration / num_frames  # 每帧对应的秒数

        # 二值化
        binary = (activity > threshold).astype(int)  # (F, 3)

        # 逐帧判定：只取"恰好 1 个说话人活跃"的帧
        # frame_speaker[f] = speaker_id (0/1/2) 或 -1 (静音/重叠)
        frame_speaker = np.full(num_frames, -1, dtype=int)
        for f in range(num_frames):
            active = np.where(binary[f] == 1)[0]
            if len(active) == 1:
                frame_speaker[f] = active[0]
            # len(active) == 0 → 静音，len(active) >= 2 → 重叠，都标 -1

        # 提取连续同人帧段
        segments = []
        seg_start = 0
        seg_spk = frame_speaker[0]
        for f in range(1, num_frames):
            if frame_speaker[f] != seg_spk:
                if seg_spk >= 0:
                    start_sec = seg_start * frame_sec
                    end_sec = f * frame_sec
                    # 不超过实际音频时长
                    end_sec = min(end_sec, actual_duration)
                    if end_sec - start_sec >= min_seg_sec:
                        segments.append((start_sec, end_sec, int(seg_spk)))
                seg_start = f
                seg_spk = frame_speaker[f]
        # 末尾段
        if seg_spk >= 0:
            start_sec = seg_start * frame_sec
            end_sec = min(num_frames * frame_sec, actual_duration)
            if end_sec - start_sec >= min_seg_sec:
                segments.append((start_sec, end_sec, int(seg_spk)))

        return segments

    def process_full_audio(
        self,
        audio_np: np.ndarray,
        sr: int = 16000,
        threshold: float = 0.5,
        min_seg_sec: float = 0.3,
    ) -> List[Tuple[float, float, np.ndarray]]:
        """处理整段音频（任意时长），返回所有非重叠 homogeneous segments + 对应音频。

        将音频按 10s chunk（无重叠）分段，逐 chunk 做 segmentation，
        提取非重叠单人段，拼接全局时间戳。

        Args:
            audio_np: 完整 16kHz 单声道音频
            sr: 采样率
            threshold: 活性阈值
            min_seg_sec: 最短 segment

        Returns:
            List[(global_start_sec, global_end_sec, segment_audio_np)]
            segment_audio_np 是该段对应的原始音频 numpy 切片。
        """
        total_samples = len(audio_np)
        all_segments = []

        pos = 0
        while pos < total_samples:
            end = min(pos + _CHUNK_SAMPLES, total_samples)
            chunk = audio_np[pos:end]
            chunk_offset_sec = pos / sr

            local_segs = self.process_chunk(chunk, sr, threshold, min_seg_sec)

            for s, e, _local_spk in local_segs:
                global_start = chunk_offset_sec + s
                global_end = chunk_offset_sec + e
                start_sample = int(global_start * sr)
                end_sample = min(int(global_end * sr), total_samples)
                if end_sample > start_sample:
                    seg_audio = audio_np[start_sample:end_sample]
                    all_segments.append((global_start, global_end, seg_audio))

            pos += _CHUNK_SAMPLES  # 无重叠步进

        return all_segments
