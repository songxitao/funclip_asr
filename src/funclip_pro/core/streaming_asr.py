"""P3.2 流式 ASR 引擎封装。

将 app_live_local.py 中的 SileroVAD / FsmnVadStreaming / run_engine 逻辑
下沉到 core.streaming_asr，提供会话管理、VAD+ASR 流式接口。

设计原则：
  - 所有重型依赖（onnxruntime、funasr）均在方法内部惰性导入，模块顶层仅依赖 numpy。
  - 模型路径由 config.loader.resolve_model_path() 管理，零硬编码。
  - 会话隔离：每个 session 拥有独立的 buffer、VAD 状态机。
"""

from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional

import numpy as np

from funclip_pro.config.loader import resolve_model_path

# ================= 常量 =================
VAD_START_THRESHOLD = 0.5   # 启动门槛
VAD_END_THRESHOLD = 0.3     # 维持门槛
PAUSE_LIMIT_SEC = 0.15      # 句尾停顿
MAX_SENTENCE_SEC = 6.0      # 最长句子
MIN_SENTENCE_SEC = 0.5      # 最短有效长度

CHUNK_DURATION = 0.032      # 32ms / 帧（512 samples @ 16kHz）
PREVIEW_MIN_INTERVAL = 0.3  # 实时预览最小间隔


# ================= Silero VAD =================
class SileroVAD:
    """ONNX 推理的 Silero VAD，支持内部状态（自持）和外部状态（由引擎管理）。

    等价于 app_live_local.py:90-132 的 SileroVAD 类，但模型路径由外部传入。
    """

    def __init__(self, model_path: str):
        import os
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"VAD 模型未找到: {model_path}")

        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.log_severity_level = 3
        self.session = onnxruntime.InferenceSession(
            model_path, providers=['CPUExecutionProvider'], sess_options=opts
        )
        self.input_names = [x.name for x in self.session.get_inputs()]
        self.reset_states()

    def reset_states(self):
        """将内部状态重置为全零。"""
        if 'state' in self.input_names:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def __call__(self, audio_chunk: np.ndarray) -> float:
        """使用内部状态推理，返回说话概率 [0, 1]。"""
        input_tensor = audio_chunk[np.newaxis, :]
        ort_inputs = {
            'input': input_tensor.astype(np.float32),
            'sr': np.array([16000], dtype='int64'),
        }

        if 'state' in self.input_names:
            ort_inputs['state'] = self._state
            out, new_state = self.session.run(None, ort_inputs)
            self._state = new_state
        elif 'h' in self.input_names and 'c' in self.input_names:
            ort_inputs['h'] = self._h
            ort_inputs['c'] = self._c
            out, h, c = self.session.run(None, ort_inputs)
            self._h, self._c = h, c
        else:
            out = self.session.run(None, ort_inputs)[0]

        return float(out[0][0])

    def copy_state(self) -> dict:
        """复制当前内部状态（用于多 session 隔离）。"""
        if 'state' in self.input_names:
            return {'state': self._state.copy()}
        else:
            return {'h': self._h.copy(), 'c': self._c.copy()}

    def restore_state(self, state: dict):
        """恢复内部状态。"""
        if 'state' in self.input_names and 'state' in state:
            self._state = state['state'].copy()
        else:
            if 'h' in state:
                self._h = state['h'].copy()
            if 'c' in state:
                self._c = state['c'].copy()

    def __del__(self):
        """显式释放 ONNX InferenceSession 句柄，减少 CUDA/CPU 句柄泄露风险。"""
        if hasattr(self, "session"):
            try:
                del self.session
            except Exception:
                pass


# ================= FSMN VAD Streaming (能量门控版) =================
class FsmnVadStreaming:
    """FunASR FSMN VAD 流式封装，能量门控版。

    等价于 app_live_local.py:134-230 的 FsmnVadStreaming 类。
    使用能量门控，只在有声音时才调用 FSMN。
    """

    ENERGY_THRESHOLD = 0.001          # 低于此值视为静音
    SILENCE_CHUNKS_TO_FLUSH = 10      # 连续 N 个静音块后刷新 VAD
    MIN_SPEECH_DURATION_MS = 300     # 最短语音段长度

    def __init__(self, chunk_size_ms: int = 200, silence_chunks_to_flush: int = 10):
        from funasr import AutoModel

        self.model = AutoModel(
            model="fsmn-vad",
            model_revision="v2.0.4",
            device="cpu",
            disable_update=True,
            disable_pbar=True,
        )
        self.chunk_size = chunk_size_ms
        self.SILENCE_CHUNKS_TO_FLUSH = silence_chunks_to_flush
        self.cache: dict = {}
        self.sample_rate = 16000
        self.chunk_stride = int(chunk_size_ms * self.sample_rate / 1000)

        # 状态跟踪
        self.accumulated_audio: List[np.ndarray] = []
        self.silence_counter = 0
        self.is_speaking = False
        self.session_start_ms = 0
        self._last_audio: Optional[np.ndarray] = None

    def reset(self):
        """重置所有内部状态。"""
        self.cache = {}
        self.accumulated_audio = []
        self.silence_counter = 0
        self.is_speaking = False
        self.session_start_ms = 0
        self._last_audio = None

    def process_chunk(self, audio_chunk: np.ndarray):
        """处理一块音频。

        返回:
            completed_segments: List[(start_ms, end_ms)] — 完成的语音段落
            accumulated_audio: 如果正在说话，返回累积音频列表；否则 None
        """
        energy = float(np.abs(audio_chunk).mean())
        completed_segments: list = []

        if energy > self.ENERGY_THRESHOLD:
            # ===== 有声音 =====
            self.silence_counter = 0
            self.accumulated_audio.append(audio_chunk)

            if not self.is_speaking:
                self.is_speaking = True
                self.session_start_ms = 0
                self.cache = {}

        else:
            # ===== 静音 =====
            self.silence_counter += 1

            if self.is_speaking and self.silence_counter >= self.SILENCE_CHUNKS_TO_FLUSH:
                if len(self.accumulated_audio) > 0:
                    full_audio = np.concatenate(self.accumulated_audio)
                    self._last_audio = full_audio

                    duration_ms = len(full_audio) / self.sample_rate * 1000
                    if duration_ms >= self.MIN_SPEECH_DURATION_MS:
                        completed_segments.append((0, int(duration_ms)))

                    self.accumulated_audio = []

                self.is_speaking = False
                self.cache = {}

        return completed_segments, self.accumulated_audio if self.is_speaking else None


# ================= FunAsrStreamingEngine =================
class FunAsrStreamingEngine:
    """流式 ASR 引擎管理器。

    提供多会话（session）管理，每个 session 拥有独立的 buffer 和 VAD 状态。
    支持 Silero VAD（默认）和 FSMN VAD 两种模式。
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.vad_mode = self.config.get("vad_mode", "silero").lower()

        # 模型实例（惰性加载）
        self._vad_model: Optional[SileroVAD] = None
        self._asr_model = None

        # 会话状态: session_id -> session_state
        self._sessions: Dict[str, dict] = {}

    def _ensure_vad(self):
        """惰性加载 VAD 模型。"""
        if self._vad_model is not None:
            return

        from funclip_pro.config.loader import apply_dll_patch
        apply_dll_patch()

        if self.vad_mode == "fsmn":
            # FSMN VAD 不需要预加载，FsmnVadStreaming 内部自行惰性加载
            self._vad_model = "fsmn"  # 标记为 FSMN 模式
        else:
            # Silero VAD (默认)
            vad_path = str(resolve_model_path("models/silero_vad.onnx"))
            self._vad_model = SileroVAD(vad_path)

    def _ensure_asr(self):
        """惰性加载 ASR 模型。"""
        if self._asr_model is not None:
            return

        from funclip_pro.config.loader import apply_dll_patch
        apply_dll_patch()

        from funasr import AutoModel

        model_dir = self.config.get("model_dir") or "models/iic/SenseVoiceSmall"
        model_path = str(resolve_model_path(model_dir))
        self._asr_model = AutoModel(
            model=model_path,
            trust_remote_code=True,
            device="cpu",
            disable_update=True,
            disable_pbar=True,
        )

    def create_session(self, language: str = "auto") -> str:
        """创建新的流式识别会话，返回唯一 session_id。"""
        session_id = str(uuid.uuid4())
        session_state = {
            "buffer": [],               # 累积音频帧列表
            "is_speaking": False,       # 是否正在说话
            "silence_cnt": 0,           # 连续静音帧计数
            "vad_cache": {},            # 缓存上一次的 VAD 状态
            "history": [],              # 历史识别文本
            "language": language,       # 识别语言
            "_last_preview_time": 0,    # 上次预览时间戳
        }
        self._sessions[session_id] = session_state
        return session_id

    def destroy_session(self, session_id: str):
        """销毁会话，释放状态。"""
        self._sessions.pop(session_id, None)

    def feed_chunk(self, session_id: str, chunk: np.ndarray) -> List[dict]:
        """输入一块 PCM 音频数据（16kHz float32），返回增量识别结果。

        Args:
            session_id: 会话 ID（由 create_session 返回）
            chunk: 512 点 float32 音频帧 (32ms @ 16kHz)

        Returns:
            List[dict]: 每个元素包含 {"text", "start_ms", "end_ms"}
        """
        if session_id not in self._sessions:
            raise KeyError(f"会话 {session_id} 不存在")

        self._ensure_vad()
        self._ensure_asr()

        session = self._sessions[session_id]
        results: List[dict] = []

        if self.vad_mode == "fsmn":
            results = self._process_fsmn(session, chunk)
        else:
            results = self._process_silero(session, chunk)

        return results

    def _process_silero(self, session: dict, chunk: np.ndarray) -> List[dict]:
        """Silero VAD 模式处理逻辑。

        等价于 app_live_local.py:845-901 的 Silero VAD 分支逻辑。
        """
        results: List[dict] = []

        # 为当前 session 延迟初始化独立的 SileroVAD 实例
        if "_silero_vad" not in session:
            try:
                vad_path = str(resolve_model_path("models/silero_vad.onnx"))
                session["_silero_vad"] = SileroVAD(vad_path)
            except Exception:
                pass

        if "_silero_vad" in session:
            silero_vad = session["_silero_vad"]
            speech_prob = silero_vad(chunk)
        else:
            # 降级到引擎级 VAD（用于测试或向后兼容）
            silero_vad = self._vad_model
            speech_prob = self._vad_model(chunk)

        active_threshold = VAD_END_THRESHOLD if session["is_speaking"] else VAD_START_THRESHOLD
        min_len_points = int(16000 * MIN_SENTENCE_SEC)

        if speech_prob > active_threshold:
            # ===== 检测到语音 =====
            session["is_speaking"] = True
            session["silence_cnt"] = 0
            session["buffer"].append(chunk)

            # █ 实时预览（段内打字效果，不截断 buffer，不追加 history）
            _now = time.time()
            if (_now - session["_last_preview_time"] > PREVIEW_MIN_INTERVAL
                    and len(session["buffer"]) > 5):
                session["_last_preview_time"] = _now
                try:
                    temp_audio = np.concatenate(session["buffer"])
                    preview_text = self._run_asr(temp_audio, session)
                    if preview_text:
                        results.append({
                            "text": preview_text,
                            "start_ms": 0,
                            "end_ms": 0,
                            "is_final": False,
                        })
                except Exception:
                    pass

            # 超时强制截断
            if len(session["buffer"]) * CHUNK_DURATION > MAX_SENTENCE_SEC:
                sentence = np.concatenate(session["buffer"])
                if len(sentence) > min_len_points:
                    text = self._run_asr(sentence, session)
                    if text:
                        duration_ms = int(len(sentence) / 16000 * 1000)
                        results.append({"text": text, "start_ms": 0, "end_ms": duration_ms})
                        session["history"].append(text)
                        if len(session["history"]) > 4:
                            session["history"].pop(0)

                session["buffer"] = []
                session["is_speaking"] = False
                silero_vad.reset_states()

        else:
            # ===== 静音或背景噪声 =====
            if session["is_speaking"]:
                session["silence_cnt"] += 1
                session["buffer"].append(chunk)

                pause_limit_count = int(PAUSE_LIMIT_SEC / CHUNK_DURATION)
                if session["silence_cnt"] > pause_limit_count:
                    sentence = np.concatenate(session["buffer"])
                    if len(sentence) > min_len_points:
                        text = self._run_asr(sentence, session)
                        if text:
                            duration_ms = int(len(sentence) / 16000 * 1000)
                            results.append({"text": text, "start_ms": 0, "end_ms": duration_ms})
                            session["history"].append(text)
                            if len(session["history"]) > 4:
                                session["history"].pop(0)

                    session["buffer"] = []
                    session["is_speaking"] = False
                    session["silence_cnt"] = 0
                    silero_vad.reset_states()

        return results

    def _process_fsmn(self, session: dict, chunk: np.ndarray) -> List[dict]:
        """FSMN VAD 模式处理逻辑。"""
        results: List[dict] = []

        # 每个 session 需要自己的 FsmnVadStreaming 实例
        if "_fsmn_vad" not in session:
            session["_fsmn_vad"] = FsmnVadStreaming(chunk_size_ms=200)

        fsmn_vad = session["_fsmn_vad"]
        completed_segments, _ = fsmn_vad.process_chunk(chunk)

        if completed_segments and hasattr(fsmn_vad, "_last_audio"):
            segment_audio = fsmn_vad._last_audio
            if segment_audio is not None and len(segment_audio) > int(16000 * MIN_SENTENCE_SEC):
                text = self._run_asr(segment_audio, session)
                if text:
                    duration_ms = int(len(segment_audio) / 16000 * 1000)
                    results.append({"text": text, "start_ms": 0, "end_ms": duration_ms})
                    session["history"].append(text)
                    if len(session["history"]) > 4:
                        session["history"].pop(0)

        return results

    def _run_asr(self, audio: np.ndarray, session: dict) -> str:
        """对音频片段执行 ASR 推理，返回文本。"""
        import torch  # noqa: F401

        language = session.get("language", "auto")

        try:
            res = self._asr_model.generate(
                input=[audio],
                language=language,
                use_itn=True,
                batch_size_s=0,
            )
            if res and isinstance(res, list) and len(res) > 0:
                item = res[0]
                if isinstance(item, dict):
                    text = item.get("text", "")
                else:
                    text = str(item)
                # 剥掉 <|...|> 标签
                import re
                text = re.sub(r"<\|.*?\|>", "", text).strip()
                return text
        except Exception:
            pass

        return ""
