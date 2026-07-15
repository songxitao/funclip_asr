"""funclip_pro.core.audio — P3.2 音频采集层下沉。

提供 PyAudio 硬件抽象的音频流接口，输出 16kHz 单声道 float32 PCM。
"""

import queue as q_module
import time

import numpy as np

# 条件导入 PyAudio（参考 app_live_local.py 80-85 行）
try:
    import pyaudiowpatch as pyaudio
    HAS_LOOPBACK = True
except ImportError:
    import pyaudio
    HAS_LOOPBACK = False

VOLUME_BOOST = 3.0  # 默认增益


def process_audio_frame(
    data: np.ndarray,
    src_channels: int,
    src_rate: int,
    volume_boost: float = 3.0,
) -> np.ndarray:
    """将原始 int16 PCM 帧转换为 16kHz 单声道 float32 [-1, 1]。

    Args:
        data: int16 ndarray 原始 PCM 数据
        src_channels: 源声道数
        src_rate: 源采样率 (Hz)
        volume_boost: 增益倍数，默认 3.0

    Returns:
        16kHz 单声道 float32 ndarray, 值域 [-1, 1]
    """
    audio = data.astype(np.float32) / 32768.0
    # 确保极值精确映射：32767→1.0, -32768→-1.0
    if data.dtype == np.int16:
        audio[data == 32767] = 1.0
        audio[data == -32768] = -1.0

    # 多声道转单声道
    if src_channels > 1:
        audio = audio.reshape(-1, src_channels).mean(axis=1)

    # 重采样到 16kHz（线性插值简化版）
    if src_rate != 16000:
        num_samples = int(len(audio) * 16000 / src_rate)
        indices = np.linspace(0, len(audio) - 1, num_samples)
        audio = audio[indices.astype(int)]

    # VOLUME_BOOST 增益 + 截断
    audio = audio * volume_boost
    audio = np.clip(audio, -1.0, 1.0)
    return audio.astype(np.float32)


class BaseStream:
    """音频采集基类，封装 PyAudio 回调队列。"""

    def __init__(self):
        self.q = q_module.Queue()
        self.rate = 16000
        self.target_chunk_size = 512
        self.src_channels = 1
        self.src_rate = 16000

    def _get_pyaudio(self):
        """延迟获取 PyAudio 实例（允许无硬件时测试）。"""
        if not hasattr(self, "_pyaudio_instance"):
            self._pyaudio_instance = pyaudio.PyAudio()
        return self._pyaudio_instance

    def callback(self, in_data, frame_count, time_info, status):
        self.q.put(in_data)
        return (None, pyaudio.paContinue)

    def read(self):
        raw = self.q.get()
        return process_audio_frame(
            np.frombuffer(raw, dtype=np.int16),
            self.src_channels,
            self.src_rate,
            volume_boost=VOLUME_BOOST,
        )

    def get_queue_size(self):
        return self.q.qsize()

    def iter_frames(self, max_frames=None):
        count = 0
        while max_frames is None or count < max_frames:
            try:
                yield self.read()
                count += 1
            except Exception:
                break


class LoopbackStream(BaseStream):
    """WASAPI 环回采集（电脑声卡输出）。"""

    def start(self):
        if not HAS_LOOPBACK:
            raise Exception("No pyaudiowpatch library")
        p = self._get_pyaudio()
        target = None
        wasapi_index = next(
            i
            for i in range(p.get_host_api_count())
            if p.get_host_api_info_by_index(i)["type"] == pyaudio.paWASAPI
        )
        wasapi_info = p.get_host_api_info_by_index(wasapi_index)
        default_out = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        for loopback in p.get_loopback_device_info_generator():
            if default_out["name"] in loopback["name"]:
                target = loopback
                break
        if not target:
            loopbacks = list(p.get_loopback_device_info_generator())
            if loopbacks:
                target = loopbacks[0]
            else:
                raise Exception("系统未发现任何 Loopback 设备！")
        self.src_rate = int(target["defaultSampleRate"])
        self.src_channels = target["maxInputChannels"]
        chunk_src = int(self.src_rate * (self.target_chunk_size / 16000))
        self.stream = p.open(
            format=pyaudio.paInt16,
            channels=self.src_channels,
            rate=self.src_rate,
            input=True,
            input_device_index=target["index"],
            frames_per_buffer=chunk_src,
            stream_callback=self.callback,
        )
        return self

    def stop(self):
        try:
            self.stream.stop_stream()
            self.stream.close()
            if hasattr(self, "_pyaudio_instance"):
                self._pyaudio_instance.terminate()
                del self._pyaudio_instance
        except Exception:
            pass


class MicStream(BaseStream):
    """麦克风采集。"""

    def start(self):
        p = self._get_pyaudio()
        target = None
        try:
            target = p.get_default_input_device_info()
        except Exception:
            pass
        if not target:
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info["maxInputChannels"] > 0:
                    target = info
                    break
        if not target:
            raise Exception("未找到任何输入设备")
        self.src_rate = int(target["defaultSampleRate"])
        self.src_channels = target["maxInputChannels"]
        chunk_src = int(self.src_rate * (self.target_chunk_size / 16000))
        self.stream = p.open(
            format=pyaudio.paInt16,
            channels=self.src_channels,
            rate=self.src_rate,
            input=True,
            input_device_index=target["index"],
            frames_per_buffer=chunk_src,
            stream_callback=self.callback,
        )
        return self

    def stop(self):
        try:
            self.stream.stop_stream()
            self.stream.close()
            if hasattr(self, "_pyaudio_instance"):
                self._pyaudio_instance.terminate()
                del self._pyaudio_instance
        except Exception:
            pass


class MixedStream:
    """Mic + Loopback 混音采集。"""

    def __init__(self):
        self.mic = MicStream()
        self.loop = LoopbackStream()

    def start(self):
        loop_ok = False
        try:
            self.loop.start()
            loop_ok = True
        except Exception:
            self.loop = None
        try:
            self.mic.start()
        except Exception:
            if not loop_ok:
                raise Exception("内录和麦克风全都启动失败！")
        return self

    def get_queue_size(self):
        s1 = self.mic.get_queue_size() if hasattr(self.mic, "get_queue_size") else 0
        s2 = self.loop.get_queue_size() if self.loop and hasattr(self.loop, "get_queue_size") else 0
        return max(s1, s2)

    @staticmethod
    def _raw_to_float(raw_bytes: bytes, src_channels: int, src_rate: int) -> np.ndarray:
        """将原始 PCM bytes 转为 float32 ndarray（16kHz 单声道，不做 boost/clip）。"""
        data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if src_channels > 1:
            data = data.reshape(-1, src_channels).mean(axis=1)
        if src_rate != 16000:
            num_samples = int(len(data) * 16000 / src_rate)
            indices = np.linspace(0, len(data) - 1, num_samples)
            data = data[indices.astype(int)]
        return data

    def read(self):
        chunk_mic = None
        try:
            raw_mic = self.mic.q.get(timeout=0.005)
            chunk_mic = self._raw_to_float(raw_mic, self.mic.src_channels, self.mic.src_rate)
        except q_module.Empty:
            pass

        chunk_loop = None
        if self.loop:
            try:
                raw_loop = self.loop.q.get(timeout=0.005)
                chunk_loop = self._raw_to_float(raw_loop, self.loop.src_channels, self.loop.src_rate)
            except q_module.Empty:
                pass

        if chunk_mic is not None and chunk_loop is not None:
            min_len = min(len(chunk_mic), len(chunk_loop))
            mixed = (chunk_mic[:min_len] + chunk_loop[:min_len]) / 2
            return np.clip(mixed * VOLUME_BOOST, -1.0, 1.0)
        elif chunk_mic is not None:
            return np.clip(chunk_mic * VOLUME_BOOST, -1.0, 1.0)
        elif chunk_loop is not None:
            return np.clip(chunk_loop * VOLUME_BOOST, -1.0, 1.0)
        else:
            time.sleep(0.032)
            return np.zeros(512, dtype=np.float32)

    def iter_frames(self, max_frames=None):
        count = 0
        while max_frames is None or count < max_frames:
            try:
                yield self.read()
                count += 1
            except Exception:
                break

    def stop(self):
        if self.loop:
            self.loop.stop()
        self.mic.stop()
