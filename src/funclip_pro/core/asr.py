"""ASR 推理引擎封装（core 下沉版）。

将根目录 asr_onnx_service.py / torch_engine.py / sherpa_engine.py 的三套
SenseVoice 推理引擎与解码工具统一下沉到 core 层，供 OfflinePipeline 调度。

包含：
  - SenseVoiceSmall  : OpenVINO/ONNX 推理引擎（原 asr_onnx_service.SenseVoiceSmall）
  - PyTorchSenseVoice: funasr AutoModel PyTorch-GPU 引擎（原 torch_engine）
  - SherpaSenseVoice : sherpa-onnx INT8 CPU 引擎（原 sherpa_engine）
  - 解码/路由工具    : load_models / _decode / _post_punc / _clean / _cheap_trim
                       / _select_engine / _use_vad / _merge_vad_segments

等价优先原则：
  - 推理与解码逻辑与原代码字节级一致，未做 45454 交替 / 字级时间戳对齐优化。
  - 时间戳统一毫秒(ms)。
  - 不实际加载模型权重：重型第三方依赖（torch / sherpa_onnx / openvino /
    funasr / utils.model_bin）均在方法内部惰性导入，模块顶层仅依赖 numpy。
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
from typing import List, Optional, Union

import numpy as np
import io
import soundfile as sf

# 路径解耦：统一由 config.loader 解析模型目录，零硬编码盘符/绝对路径。
from funclip_pro.config.loader import resolve_model_path

from funclip_pro.core.tokenization import CharTokenizer

logger = logging.getLogger("ASRService")

# ===== HANDOFF §4: 三态 VAD 策略 + 引擎自动路由 + 廉价 trim =====
SHORT_AUDIO_MS = 5000           # 短音频阈值(ms)：<= 此值走廉价 trim 直解
SHORT_TRIM_TOP_DB = 40          # librosa.effects.trim 的 top_db（保留轻语音）
TRIM_PAD_MS = 100               # trim 边界缓冲(ms)，防削字

# PyTorch-GPU 引擎模型目录（路径解耦，由 resolve_model_path 解析）
TORCH_MODEL_DIR = str(resolve_model_path("models/iic/SenseVoiceSmall"))

# 标点清洗正则：用于剥掉 ASR 逐段原生标点（保留 ITN 规整后的数字与汉字）
_PUNC_RE = re.compile(
    r"[，。！？；：、…—·「」『』“”‘’（）《》〈〉【】\[\]\(\)\{\}\"'\.,!?;:\s]"
)


def strip_punctuation(s: str) -> str:
    return _PUNC_RE.sub("", s).strip()


# 全局模型句柄（运行时由 load_models / 惰性 getter 填充）
MODEL = None
VAD_MODEL = None
PUNC_MODEL = None

# PyTorch-GPU 引擎惰性加载（锁内构建，保证集成测试启动不超时）
_TORCH_LOCK = threading.Lock()
TORCH_MODEL = None

_LABEL_RE = re.compile(r"<\|.*?\|>")


def _apply_dll_patch_once():
    """等价原 asr_onnx_service 顶部 DLL 点亮：在首次加载重型库前调用一次。"""
    try:
        from funclip_pro.config.loader import apply_dll_patch
        apply_dll_patch()
    except Exception as e:  # noqa: BLE001 - 降级容错
        logger.warning("apply_dll_patch 失败（已跳过）: %s", e)


# ---------------------------------------------------------------------------
# SenseVoiceSmall（OpenVINO / ONNX 推理引擎）
# ---------------------------------------------------------------------------
# base 类来自 SenseVoiceSmall 模型目录下的 utils.model_bin，需要在 sys.path 上。
# 模块导入期惰性解析，解析/导入失败则回退到占位基类，保证 import 永远成功。
def _resolve_sensevoice_onnx_base():
    try:
        _sv_dir = str(resolve_model_path("models/iic/SenseVoiceSmall"))
        if _sv_dir not in sys.path:
            sys.path.insert(0, _sv_dir)
        from utils.model_bin import SenseVoiceSmallONNX  # type: ignore
        return SenseVoiceSmallONNX
    except Exception as e:  # noqa: BLE001 - 模型包缺失时降级
        logger.warning("未能加载 utils.model_bin.SenseVoiceSmallONNX，使用占位基类: %s", e)

        class _SenseVoiceSmallONNXStub:
            def load_data(self, wav_content, fs=None):
                raise RuntimeError("SenseVoiceSmallONNX 基类未加载（模型包不可用）")

        return _SenseVoiceSmallONNXStub


class SenseVoiceSmall(_resolve_sensevoice_onnx_base()):
    """包装类，重写了初始化和调用，适配用户要求的接口（等价原实现）。"""

    def __init__(self, model_dir, batch_size=1, quantize=True, device_id="-1",
                 intra_op_num_threads=4, **kwargs):
        _apply_dll_patch_once()
        from utils.infer_utils import CharTokenizer as _ModelCharTokenizer, read_yaml  # type: ignore
        from utils.frontend import WavFrontend  # type: ignore

        if quantize:
            model_file = os.path.join(model_dir, "model_quant.onnx")
        else:
            model_file = os.path.join(model_dir, "model.onnx")

        config_file = os.path.join(model_dir, "config.yaml")
        cmvn_file = os.path.join(model_dir, "am.mvn")
        config = read_yaml(config_file)

        self.tokenizer = _ModelCharTokenizer()
        config["frontend_conf"]['cmvn_file'] = cmvn_file
        self.frontend = WavFrontend(**config["frontend_conf"])

        from openvino import Core  # type: ignore
        self.core = Core()
        ov_model = self.core.read_model(model_file)
        self.compiled_model = self.core.compile_model(
            ov_model,
            "CPU",
            config={
                "INFERENCE_NUM_THREADS": str(intra_op_num_threads),
                "NUM_STREAMS": "1"
            }
        )

        self.batch_size = batch_size
        self.blank_id = 0

        # 加载 tokens.json 以还原文本
        tokens_path = os.path.join(model_dir, "tokens.json")
        with open(tokens_path, "r", encoding="utf-8") as f:
            self.tokens = json.load(f)

    def infer(self, feats, feats_len, language, textnorm):
        res = self.compiled_model([feats, feats_len, language, textnorm])
        return res[0], res[1]

    def load_data(self, wav_content, fs=None):
        import librosa
        if isinstance(wav_content, list):
            return [item if isinstance(item, np.ndarray) else librosa.load(item, sr=fs)[0]
                    for item in wav_content]
        return super().load_data(wav_content, fs)

    def __call__(self, wav_content, language=[0], textnorm=[15], tokenizer=None, **kwargs):
        if tokenizer is None:
            # 内部默认 Tokenizer（等价原 DefaultTokenizer，统一复用公开 CharTokenizer）
            tokenizer = CharTokenizer(self.tokens)

        waveform_list = self.load_data(wav_content, self.frontend.opts.frame_opts.samp_freq)
        waveform_nums = len(waveform_list)
        asr_res = []

        for beg_idx in range(0, waveform_nums, self.batch_size):
            end_idx = min(waveform_nums, beg_idx + self.batch_size)
            feats, feats_len = self.extract_feat(waveform_list[beg_idx:end_idx])

            cur_batch_size = end_idx - beg_idx
            cur_language = language * cur_batch_size if len(language) == 1 else language
            cur_textnorm = textnorm * cur_batch_size if len(textnorm) == 1 else textnorm

            if len(cur_language) < cur_batch_size:
                cur_language = cur_language + [cur_language[-1]] * (cur_batch_size - len(cur_language))
            else:
                cur_language = cur_language[:cur_batch_size]

            if len(cur_textnorm) < cur_batch_size:
                cur_textnorm = cur_textnorm + [cur_textnorm[-1]] * (cur_batch_size - len(cur_textnorm))
            else:
                cur_textnorm = cur_textnorm[:cur_batch_size]

            ctc_logits, encoder_out_lens = self.infer(
                feats,
                feats_len,
                np.array(cur_language, dtype=np.int32),
                np.array(cur_textnorm, dtype=np.int32)
            )
            # 向量化批量解码，替换逐句循环，消除 CPU-GIL 延迟
            token_ids = np.argmax(ctc_logits, axis=-1)  # [B, T]

            encoder_lens_np = np.array([l.item() if hasattr(l, 'item') else l for l in encoder_out_lens])
            max_time = token_ids.shape[1]
            time_indices = np.arange(max_time)[None, :]
            valid_mask = time_indices < encoder_lens_np[:, None]

            shifted = np.roll(token_ids, 1, axis=-1)
            shifted[:, 0] = -1
            repeat_mask = (token_ids == shifted)

            keep_mask = valid_mask & (~repeat_mask) & (token_ids != self.blank_id)

            for b in range(cur_batch_size):
                token_int = token_ids[b][keep_mask[b]].tolist()
                asr_res.append(tokenizer.tokens2text(token_int))
        return asr_res


# ---------------------------------------------------------------------------
# PyTorchSenseVoice（funasr AutoModel PyTorch-GPU 引擎）
# ---------------------------------------------------------------------------
class PyTorchSenseVoice:
    def __init__(self, model_dir: str, device: str = "cpu"):
        _apply_dll_patch_once()
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"PyTorch SenseVoice 模型目录不存在: {model_dir}")

        # funasr 重型依赖延迟导入，避免顶层加载（便于单测轻量导入）
        import torch  # noqa: F401  # 延迟点亮 GPU 依赖
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
        import torch  # noqa: F401
        import re as _re
        _label_re = _re.compile(r"<\|.*?\|>")

        if not isinstance(waveforms, (list, tuple)):
            waveforms = [waveforms]
        waveforms = [np.asarray(w, dtype=np.float32) for w in waveforms]

        res = self.model.generate(
            input=waveforms,
            batch_size_s=0,
            language="auto",
            use_itn=True,
        )

        out: list = []
        for item in res:
            t = item.get("text", "") if isinstance(item, dict) else str(item)
            # 剥掉 <|...|> 标签（PyTorch 原始输出含语言/情感等标签）
            t = _label_re.sub("", t).strip()
            if t:
                out.append(t)
        return out


# ---------------------------------------------------------------------------
# SherpaSenseVoice（sherpa-onnx INT8 CPU 引擎）
# ---------------------------------------------------------------------------
class SherpaSenseVoice:
    def __init__(
        self,
        model_dir: str,
        num_threads: int = 6,
        use_itn: bool = True,
    ):
        _apply_dll_patch_once()
        model_path = os.path.join(model_dir, "model.int8.onnx")
        if not os.path.exists(model_path):
            # 回退到 fp32 模型
            model_path = os.path.join(model_dir, "model.onnx")
        tokens_path = os.path.join(model_dir, "tokens.txt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Sherpa ONNX 模型不存在: {model_path}")
        if not os.path.exists(tokens_path):
            raise FileNotFoundError(f"Sherpa 词表不存在: {tokens_path}")

        import sherpa_onnx  # type: ignore  # 延迟导入重型依赖

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
        """接收单条文件路径(str) 或 多条波形组成的 list[np.ndarray]，返回 List[str]。"""
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
        stream_index: list = []  # 记录每个 stream 对应的输入下标
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
        out: list = [""] * len(waveforms)
        for stream, idx in zip(streams, stream_index):
            out[idx] = stream.result.text
        return out


# ---------------------------------------------------------------------------
# 解码 / 路由 / 标点 工具
# ---------------------------------------------------------------------------
def load_models():
    """启动时加载 Sherpa-ONNX ASR、VAD(优先 GPU) 与 CPU 标点模型。"""
    global MODEL, VAD_MODEL, PUNC_MODEL
    _apply_dll_patch_once()
    from funasr import AutoModel  # type: ignore

    model_path = str(resolve_model_path("models/iic/SenseVoiceSmall-ONNX"))
    vad_path = str(resolve_model_path("models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"))
    punc_path = str(resolve_model_path("models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"))

    logger.info("正在加载 Sherpa-ONNX ASR 模型、VAD(优先GPU) 和 CPU 标点模型...")
    try:
        # 1. 加载 ASR（Sherpa-ONNX INT8 后端，CPU 终极提速方案，已评测验证）
        SHERPA_MODEL_DIR = str(resolve_model_path("models/iic/SenseVoiceSmallOnnx"))
        MODEL = SherpaSenseVoice(
            model_dir=SHERPA_MODEL_DIR,
            num_threads=6,
            use_itn=True,
        )
        # 2. 加载 VAD（优先 CUDA，失败回退 CPU —— 提速 DER 的 VAD 切分）
        try:
            VAD_MODEL = AutoModel(
                model=vad_path,
                trust_remote_code=True,
                device="cuda",
                disable_update=True,
                disable_pbar=True
            )
            VAD_MODEL.model.to("cuda")
            VAD_MODEL.kwargs["device"] = "cuda"
            logger.info("VAD 模型已加载到 CUDA")
        except Exception as _ve:
            logger.warning(f"VAD CUDA 加载失败，回退 CPU: {_ve}")
            VAD_MODEL = AutoModel(
                model=vad_path,
                trust_remote_code=True,
                device="cpu",
                disable_update=True,
                disable_pbar=True
            )
            VAD_MODEL.model.to("cpu")
            VAD_MODEL.kwargs["device"] = "cpu"

        # 3. 加载 PUNC 标点模型
        PUNC_MODEL = AutoModel(
            model=punc_path,
            trust_remote_code=True,
            device="cpu",
            disable_update=True,
            disable_pbar=True
        )
        PUNC_MODEL.model.to("cpu")
        PUNC_MODEL.kwargs["device"] = "cpu"

        logger.info("所有模型加载成功！")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise e


def _get_torch_model():
    """在锁内惰性构建 PyTorch-GPU 引擎；首次调用才加载模型权重。"""
    global TORCH_MODEL
    with _TORCH_LOCK:
        if TORCH_MODEL is None:
            TORCH_MODEL = PyTorchSenseVoice(model_dir=TORCH_MODEL_DIR, device="cuda")
        return TORCH_MODEL


def _select_engine(engine_override, duration_ms):
    """引擎路由：cpu->sherpa；gpu->torch；qwen/qwen3->qwen；auto->CUDA 可用且长音频走 torch，否则 sherpa。"""
    if engine_override and str(engine_override).lower() in ("qwen", "qwen3", "qwen3 (docker)"):
        return "qwen"
    if engine_override == "cpu":
        return "sherpa"
    if engine_override == "gpu":
        return "torch"
    import torch  # noqa: F401
    if torch.cuda.is_available() and (duration_ms is not None and duration_ms > SHORT_AUDIO_MS):
        return "torch"
    return "sherpa"


def _use_vad(vad_strategy, duration_ms):
    """VAD 三态：always->True；never->False；auto->长音频(>SHORT_AUDIO_MS)才 VAD。"""
    if vad_strategy == "always":
        return True
    if vad_strategy == "never":
        return False
    return duration_ms is not None and duration_ms > SHORT_AUDIO_MS


def _cheap_trim(audio_path, top_db=SHORT_TRIM_TOP_DB, pad_ms=TRIM_PAD_MS):
    """廉价 trim：用 librosa.effects.trim 切首尾静音，不加载任何模型。

    返回 (trimmed_16k_waveform, full_duration_ms)。
    """
    import librosa

    y, sr = librosa.load(audio_path, sr=16000)
    y_trim, (i0, i1) = librosa.effects.trim(y, top_db=top_db)
    pad = int(pad_ms / 1000 * sr)
    y_trim = y[max(0, i0 - pad): min(len(y), i1 + pad)]
    return (y_trim, len(y) / sr * 1000)


def _clean(t):
    """剥 <|...|> 标签 + 剥原生标点（保留 ITN 数字），返回清洗字符串。"""
    t = _LABEL_RE.sub("", t).strip()
    t = strip_punctuation(t)
    return t


def _post_punc(raw_text: str) -> str:
    """对拼接（已剥标点）的全文跑一次 PUNC 标点模型，返回带标点文本。"""
    if PUNC_MODEL is not None and raw_text.strip():
        try:
            punc_out = PUNC_MODEL.generate(input=raw_text)
            if punc_out and len(punc_out) > 0:
                raw_text = punc_out[0].get('text', raw_text)
        except Exception as punc_err:
            logger.error(f"标点符号后处理失败: {punc_err}")
    return raw_text


def _decode(engine_key, waveforms):
    """按引擎解码；torch 分支失败自动回退 Sherpa，保证端点始终返回文本。"""
    if engine_key == "torch":
        try:
            return _get_torch_model()(waveforms)
        except Exception as e:
            logger.warning(f"PyTorch 推理失败，回退 Sherpa-CPU: {e}")
            return MODEL(waveforms)
    return MODEL(waveforms)


def _merge_vad_segments(segments, max_gap_ms=300, max_duration_ms=8000):
    """合并相邻 VAD 段：间隔 < max_gap_ms 且合并后 < max_duration_ms 则合并。"""
    if not segments:
        return []
    merged = []
    curr_start, curr_end = segments[0]
    for next_start, next_end in segments[1:]:
        gap = next_start - curr_end
        duration = (curr_end - curr_start) + (next_end - next_start)
        if gap < max_gap_ms and duration < max_duration_ms:
            curr_end = next_end
        else:
            merged.append([curr_start, curr_end])
            curr_start, curr_end = next_start, next_end
    merged.append([curr_start, curr_end])
    return merged


# ---------------------------------------------------------------------------
# QwenEngine（Qwen2-Audio Docker HTTP API 引擎）
# ---------------------------------------------------------------------------
class QwenEngine:
    """Qwen2-Audio Docker 引擎，通过 HTTP API 调用远程 Docker 服务进行转写。

    接口对齐新版引擎模式，支持 __call__ 快捷转写与 transcribe 完整结果。
    API 端点从 config.json 的 qwen_server.host 读取，默认 http://127.0.0.1:28000。
    """

    def __init__(self, host: str = None):
        # 从 config.json 读取或使用默认值
        if host is None:
            try:
                from funclip_pro.config.loader import load_config
                _cfg = load_config()
                host = _cfg.get("qwen_server", {}).get("host", "http://127.0.0.1:28000")
            except Exception:
                host = "http://127.0.0.1:28000"
        self.host = host.rstrip("/")
        self.model_name = "Qwen2-Audio-7B"

        # Docker 共享存储卷路径
        try:
            import inspect
            _frame = inspect.currentframe()
            # 向上追溯到项目的根目录（funclip_pro 包的上级）
            _this_file = inspect.getfile(type(self))
            self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(_this_file))))
        except Exception:
            self.project_root = None

        self.shared_host_dir = (
            os.path.join(self.project_root, "qwen_server", "shared_tmp")
            if self.project_root else None
        )
        self.shared_docker_dir = "/app/server/shared_tmp"
        self.logger = logger

    def __call__(self, audio_path: str) -> str:
        """快捷调用，返回转写文本字符串。"""
        result = self.transcribe(audio_path)
        return result["text"]

    def transcribe_batch(self, audio_paths: Union[List[str], List[np.ndarray]], language: str = "auto") -> List[dict]:
        """批量转写音频文件。

        Args:
            audio_paths: 音频文件路径列表。
            language: 转写语言。

        Returns:
            List[dict]: 每一个元素的格式为：
                {"text": str, "timestamps": [{"text": str, "start": float, "end": float}, ...]}
        """
        # 语言规范化映射
        _QWEN3_LANG_MAP = {
            "zh": "Chinese",
            "en": "English",
            "ja": "Japanese",
            "ko": "Korean",
            "yue": "Cantonese",
            "ar": "Arabic",
            "de": "German",
            "fr": "French",
            "es": "Spanish",
            "pt": "Portuguese",
            "id": "Indonesian",
            "it": "Italian",
            "ru": "Russian",
            "th": "Thai",
            "vi": "Vietnamese",
            "tr": "Turkish",
            "hi": "Hindi",
            "ms": "Malay",
            "nl": "Dutch",
            "sv": "Swedish",
            "da": "Danish",
            "fi": "Finnish",
            "pl": "Polish",
            "cs": "Czech",
            "fil": "Filipino",
            "fa": "Persian",
            "el": "Greek",
            "hu": "Hungarian",
            "mk": "Macedonian",
            "ro": "Romanian",
            "auto": None,
        }
        lang_key = str(language[0] if isinstance(language, list) else language).lower() if language else "auto"
        api_lang = _QWEN3_LANG_MAP.get(lang_key, language)

        # --- In-memory batch path: numpy 数组输入，单次 Base64 批量 POST ---
        if audio_paths and isinstance(audio_paths[0], np.ndarray):
            import base64
            import io
            import soundfile as sf
            import requests

            b64_list = []
            for chunk in audio_paths:
                buffer = io.BytesIO()
                sf.write(buffer, np.asarray(chunk, dtype=np.float32), 16000,
                         format='WAV', subtype='PCM_16')
                wav_bytes = buffer.getvalue()
                buffer.close()
                b64_list.append(base64.b64encode(wav_bytes).decode("utf-8"))

            payload = {
                "language": api_lang,
                "return_timestamps": True,
                "audio_batch_base64": b64_list,
            }

            try:
                res = requests.post(
                    f"{self.host}/v1/audio/batch_transcriptions",
                    json=payload,
                    timeout=(5.0, 90.0),
                )
                if res.status_code != 200:
                    raise RuntimeError(f"API Error {res.status_code}: {res.text}")
                data = res.json()
                if isinstance(data, dict) and "results" in data:
                    return data["results"]
                elif isinstance(data, list):
                    return data
                return [data]
            except requests.exceptions.ConnectTimeout as e:
                self.logger.error("[QwenEngine] Docker ASR 服务连接超时(5s): %s", e)
                raise RuntimeError("Docker ASR 服务连接超时") from e
            except requests.exceptions.ReadTimeout as e:
                self.logger.error("[QwenEngine] Docker ASR 服务响应超时(90s): %s", e)
                raise RuntimeError("Docker ASR 服务响应超时") from e
            except requests.exceptions.ConnectionError as e:
                self.logger.error("[QwenEngine] Docker ASR 服务无法连接: %s", e)
                raise RuntimeError("Docker ASR 服务无法连接") from e

        import base64
        import requests
        import shutil
        import uuid

        use_shared = False
        docker_paths = []
        copied_host_paths = []

        if self.shared_host_dir:
            try:
                os.makedirs(self.shared_host_dir, exist_ok=True)
                for path in audio_paths:
                    abs_path = os.path.abspath(path)
                    abs_shared = os.path.abspath(self.shared_host_dir)
                    rel = os.path.relpath(abs_path, abs_shared)
                    if not rel.startswith("..") and not os.path.isabs(rel):
                        # 已经在共享目录下
                        d_path = os.path.join(self.shared_docker_dir, rel).replace("\\", "/")
                        docker_paths.append(d_path)
                    else:
                        ext = os.path.splitext(path)[1]
                        filename = f"batch_{uuid.uuid4()}{ext}"
                        dest = os.path.join(self.shared_host_dir, filename)
                        shutil.copy2(path, dest)
                        copied_host_paths.append(dest)
                        d_path = os.path.join(self.shared_docker_dir, filename).replace("\\", "/")
                        docker_paths.append(d_path)
                use_shared = True
                self.logger.info("[QwenEngine] 成功激活 Docker 共享卷直读模式，直读 %d 个切片文件", len(audio_paths))
            except Exception as e:
                self.logger.warning("[QwenEngine] 共享卷模式激活失败，降级为 Base64 传输: %s", e)
                for p in copied_host_paths:
                    try: os.remove(p)
                    except: pass
                copied_host_paths = []
                docker_paths = []
                use_shared = False

        if use_shared:
            payload = {
                "language": api_lang,
                "return_timestamps": True,
                "audio_paths": docker_paths,
            }
        else:
            b64_list = []
            for path in audio_paths:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                b64_list.append(b64)
            payload = {
                "language": api_lang,
                "return_timestamps": True,
                "audio_batch_base64": b64_list,
            }

        try:
            res = requests.post(
                f"{self.host}/v1/audio/batch_transcriptions",
                json=payload,
                timeout=(5.0, 90.0),
            )
            if res.status_code != 200:
                raise RuntimeError(f"API Error {res.status_code}: {res.text}")

            data = res.json()
            if isinstance(data, dict) and "results" in data:
                results = data["results"]
            elif isinstance(data, list):
                results = data
            else:
                results = [data]
            return results
        except requests.exceptions.ConnectTimeout as e:
            self.logger.error("[QwenEngine] Docker ASR 服务连接超时(5s)，请确认服务是否已启动: %s", e)
            raise RuntimeError("Docker ASR 服务连接超时，请检查服务状态") from e
        except requests.exceptions.ReadTimeout as e:
            self.logger.error("[QwenEngine] Docker ASR 服务响应超时(90s)，请确认服务是否正常: %s", e)
            raise RuntimeError("Docker ASR 服务响应超时，请检查服务状态") from e
        except requests.exceptions.ConnectionError as e:
            self.logger.error("[QwenEngine] Docker ASR 服务无法连接，请确认服务是否已启动: %s", e)
            raise RuntimeError("Docker ASR 服务无法连接，请检查服务状态") from e
        except Exception as e:
            self.logger.error("[QwenEngine] 请求失败: %s", e)
            raise RuntimeError(f"QwenEngine ASR 请求异常: {e}") from e
        finally:
            for p in copied_host_paths:
                if os.path.exists(p):
                    try: os.remove(p)
                    except: pass

    def transcribe(self, audio_path: str, language: str = "auto", **kwargs) -> dict:
        """转写音频文件，返回完整结果。

        Returns:
            {"text": str, "srt": str, "raw": dict}
        """
        self.logger.info("[QwenEngine] 开始处理: %s", os.path.basename(audio_path))
        t_start = time.time()

        results = self.transcribe_batch([audio_path], language=language)
        if not results:
            raise RuntimeError("QwenEngine ASR return empty results")

        result_item = results[0]
        text = result_item.get("text", "")
        timestamps = result_item.get("timestamps", [])

        dur = time.time() - t_start
        self.logger.info("[QwenEngine] 处理完成! 耗时: %.2fs", dur)

        # 生成 SRT
        srt_content = self._build_srt(timestamps, text)

        return {
            "text": text,
            "srt": srt_content,
            "raw": result_item,
        }

    def _build_srt(self, timestamps: list, full_text: str) -> str:
        """根据时间戳列表生成 SRT 字幕内容。"""
        from funclip_pro.utils import _ms_to_srt

        if not timestamps:
            # 无时间戳时输出整段
            return f"1\n00:00:00,000 --> 00:00:10,000\n{full_text}\n"

        # 尝试将时间戳与文本分段映射
        lines = []
        idx = 1
        # 时间戳预期格式: [{"start": 0.0, "end": 2.5, "text": "..."}, ...]
        for ts in timestamps:
            start_sec = ts.get("start", 0)
            end_sec = ts.get("end", start_sec + 2)
            seg_text = ts.get("text", "")
            if not seg_text.strip():
                continue
            lines.append(
                f"{idx}\n"
                f"{_ms_to_srt(int(start_sec * 1000))} --> {_ms_to_srt(int(end_sec * 1000))}\n"
                f"{seg_text}\n"
            )
            idx += 1

        if not lines:
            return f"1\n00:00:00,000 --> 00:00:10,000\n{full_text}\n"

        return "\n".join(lines)


def parse_qwen_timestamps(raw: dict) -> list:
    """将 QwenEngine 的 raw 结果解析为 pipeline 段列表。

    Returns:
        List[{"start": ms, "end": ms, "text": str}]
    """
    timestamps = raw.get("timestamps", [])
    if not timestamps:
        return []
    segments = []
    for ts in timestamps:
        start_sec = ts.get("start", 0)
        end_sec = ts.get("end", start_sec + 2)
        seg_text = ts.get("text", "")
        if seg_text.strip():
            segments.append({
                "start": int(start_sec * 1000),
                "end": int(end_sec * 1000),
                "text": seg_text,
            })
    return segments
