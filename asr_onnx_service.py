import os
import psutil

try:
    psutil.Process().cpu_affinity([0, 1, 2, 3, 4, 5])
except Exception as e:
    print(f"警告：设置 CPU 亲和性失败: {e}")

for env_var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[env_var] = "6"

# 2. 动态添加 DLL 搜索目录以点亮 onnxruntime GPU 推理
ctranslate2_dll_path = r"E:\conda\envs\asr_ui_env\Lib\site-packages\ctranslate2"
if os.path.exists(ctranslate2_dll_path):
    try:
        os.add_dll_directory(ctranslate2_dll_path)
    except Exception:
        os.environ["PATH"] += os.pathsep + ctranslate2_dll_path

# 强力点亮 onnxruntime GPU 推理：将 nvidia\cudnn\bin、onnxruntime\capi 和 torch\lib 优先加入 PATH 与 DLL 目录
cudnn_bin = r"E:\conda\envs\asr_ui_env\Lib\site-packages\nvidia\cudnn\bin"
capi_path = r"E:\conda\envs\asr_ui_env\Lib\site-packages\onnxruntime\capi"
torch_lib = r"E:\conda\envs\asr_ui_env\Lib\site-packages\torch\lib"

extra_paths = [cudnn_bin, capi_path, torch_lib]
for path in extra_paths:
    if os.path.exists(path):
        os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
        try:
            os.add_dll_directory(path)
        except Exception:
            pass

import sys
import json
import time
import tempfile
import asyncio
import re
import logging
import threading
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from funasr import AutoModel
import uvicorn
import torch
torch.set_num_threads(6)

from openvino import Core

# 添加项目根目录到 sys.path，便于导入 sherpa_engine
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# 添加 SenseVoiceSmall 目录到 PYTHONPATH
sys.path.append(r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmall")
from utils.model_bin import SenseVoiceSmallONNX

from sherpa_engine import SherpaSenseVoice
from torch_engine import PyTorchSenseVoice
from speaker_engine import CampPlusSpeaker
from segmentation_engine import SegmentationEngine

class SenseVoiceSmall(SenseVoiceSmallONNX):
    """包装类，重写了初始化和调用，适配用户要求的接口"""
    def __init__(self, model_dir, batch_size=1, quantize=True, device_id="-1", intra_op_num_threads=4, **kwargs):
        from utils.infer_utils import CharTokenizer, read_yaml
        from utils.frontend import WavFrontend

        if quantize:
            model_file = os.path.join(model_dir, "model_quant.onnx")
        else:
            model_file = os.path.join(model_dir, "model.onnx")

        config_file = os.path.join(model_dir, "config.yaml")
        cmvn_file = os.path.join(model_dir, "am.mvn")
        config = read_yaml(config_file)

        self.tokenizer = CharTokenizer()
        config["frontend_conf"]['cmvn_file'] = cmvn_file
        self.frontend = WavFrontend(**config["frontend_conf"])
        
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
        import numpy as np
        import librosa
        if isinstance(wav_content, list):
            return [item if isinstance(item, np.ndarray) else librosa.load(item, sr=fs)[0] for item in wav_content]
        return super().load_data(wav_content, fs)

    def __call__(self, wav_content, language=[0], textnorm=[15], tokenizer=None, **kwargs):
        if tokenizer is None:
            # 内部默认 Tokenizer
            class DefaultTokenizer:
                def __init__(self, tokens):
                    self.tokens = tokens
                def tokens2text(self, ids):
                    res = []
                    for i in ids:
                        t = self.tokens[i]
                        if t.startswith("<|") and t.endswith("|>"):
                            continue
                        if t == "<space>":
                            res.append(" ")
                        elif t == "<unk>":
                            continue
                        else:
                            res.append(t)
                    return "".join(res)
            tokenizer = DefaultTokenizer(self.tokens)

        # 核心代码：重写底层以支持 batch_size > 1 时的遍历解码
        import numpy as np
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
            # 1. 向量化批量解码，替换逐句循环，消除 CPU-GIL 延迟
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

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ASRService")

app = FastAPI(title="SenseVoice ASR Service", description="极速语音转写微服务")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
VAD_MODEL = None
PUNC_MODEL = None  # 新增标点模型全局变量
GPU_SEMAPHORE = asyncio.Semaphore(3)  # 并发限制，防范 CUDA OOM
MAX_FILE_SIZE = 50 * 1024 * 1024      # 50MB 内存防线

# 标点清洗正则：用于剥掉 ASR 逐段原生标点（保留 ITN 规整后的数字与汉字）
import re as _re
_PUNC_RE = _re.compile(r"[，。！？；：、…—·「」『』“”‘’（）《》〈〉【】\[\]\(\)\{\}\"'\.,!?;:\s]")
def strip_punctuation(s: str) -> str:
    return _PUNC_RE.sub("", s).strip()

@app.on_event("startup")
def load_models():
    global MODEL, VAD_MODEL, PUNC_MODEL
    model_path = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmall-ONNX"
    vad_path = r"E:\project\funclip-pro\model\models\damo\speech_fsmn_vad_zh-cn-16k-common-pytorch"
    punc_path = r"E:\project\funclip-pro\model\models\damo\punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
    
    logger.info("正在加载 Sherpa-ONNX ASR 模型、VAD(优先GPU) 和 CPU 标点模型...")
    try:
        # 1. 加载 ASR（Sherpa-ONNX INT8 后端，CPU 终极提速方案，已评测验证）
        SHERPA_MODEL_DIR = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmallOnnx"
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


# ===== HANDOFF §4: 三态 VAD 策略 + 引擎自动路由 + 廉价 trim =====
SHORT_AUDIO_MS = 5000           # 短音频阈值(ms)：<= 此值走廉价 trim 直解
SHORT_TRIM_TOP_DB = 40          # librosa.effects.trim 的 top_db（保留轻语音）
TRIM_PAD_MS = 100               # trim 边界缓冲(ms)，防削字
TORCH_MODEL_DIR = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmall"

# PyTorch-GPU 引擎惰性加载（锁内构建，保证集成测试启动不超时）
_TORCH_LOCK = threading.Lock()
TORCH_MODEL = None

_LABEL_RE = re.compile(r"<\|.*?\|>")


def _get_torch_model():
    """在锁内惰性构建 PyTorch-GPU 引擎；首次调用才加载模型权重。"""
    global TORCH_MODEL
    with _TORCH_LOCK:
        if TORCH_MODEL is None:
            TORCH_MODEL = PyTorchSenseVoice(model_dir=TORCH_MODEL_DIR, device="cuda")
        return TORCH_MODEL


# Cam++ 说话人模型惰性加载（仅在 diarize=True 时触发；优先 CUDA 提速，失败回退 CPU）
_SPK_LOCK = threading.Lock()
SPK_MODEL = None
SPK_MODEL_DIR = r"E:\project\funclip-pro\model\models\damo\speech_campplus_sv_zh-cn_16k-common"

def _get_spk_model():
    """在锁内惰性构建 Cam++ 说话人模型；首次 diarize 请求才加载。优先 CUDA。"""
    global SPK_MODEL
    with _SPK_LOCK:
        if SPK_MODEL is None:
            try:
                SPK_MODEL = CampPlusSpeaker(model_dir=SPK_MODEL_DIR, device="cuda")
                logger.info("[Speaker] Cam++ 已加载到 CUDA")
            except Exception as e:
                logger.warning(f"[Speaker] Cam++ CUDA 加载失败，回退 CPU: {e}")
                SPK_MODEL = CampPlusSpeaker(model_dir=SPK_MODEL_DIR, device="cpu")
        return SPK_MODEL


# Segmentation-3.0 说话人分割模型惰性加载
_SEG_LOCK = threading.Lock()
SEG_MODEL = None
SEG_MODEL_DIR = r"E:\project\funclip-pro\model\models\damo\segmentation-3.0"

def _get_seg_model():
    """在锁内惰性构建 Segmentation 模型；首次 seg_clustering 请求才加载。优先 CUDA。"""
    global SEG_MODEL
    with _SEG_LOCK:
        if SEG_MODEL is None:
            try:
                SEG_MODEL = SegmentationEngine(model_dir=SEG_MODEL_DIR, device="cuda")
                logger.info("[Segmentation] pyannote/segmentation-3.0 已加载到 CUDA")
            except Exception as e:
                logger.warning(f"[Segmentation] pyannote/segmentation-3.0 CUDA 加载失败，回退 CPU: {e}")
                SEG_MODEL = SegmentationEngine(model_dir=SEG_MODEL_DIR, device="cpu")
        return SEG_MODEL


def _select_engine(engine_override, duration_ms):
    """引擎路由：cpu->sherpa；gpu->torch；auto->CUDA 可用且长音频走 torch，否则 sherpa。"""
    if engine_override == "cpu":
        return "sherpa"
    if engine_override == "gpu":
        return "torch"
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


def _assign_clauses_to_speakers(asr_start, asr_end, text, refined_segs):
    """将 ASR 段识别出的带标点 text，按照标点分句，将每个子句作为一个整体分配给重叠时间最长的说话人。
    返回列表，每个元素为 {"start": int, "end": int, "speaker": str, "text": str}
    """
    if not text.strip():
        return []

    import re
    # 匹配中英文断句标点，包括逗号
    pattern = r'([^，。？！、；：,.?!;:：\s]+[，。？！、；：,.?!;:：\s]*)'
    clauses = re.findall(pattern, text)
    if not clauses:
        clauses = [text]

    total_len = sum(len(c) for c in clauses)
    if total_len == 0:
        return []

    dur = asr_end - asr_start
    curr_start = asr_start

    assigned_clauses = []
    for clause in clauses:
        c_len = len(clause)
        c_dur = dur * (c_len / total_len)
        c_end = curr_start + c_dur

        best_spk = None
        max_overlap = -1.0
        
        for st_ms, en_ms, spk in refined_segs:
            # 计算重叠时间
            overlap = min(c_end, en_ms) - max(curr_start, st_ms)
            if overlap > max_overlap:
                max_overlap = overlap
                best_spk = spk

        # 兜底：如果重合时间为 0 或没找到
        if max_overlap <= 0 or best_spk is None:
            mid_t = curr_start + c_dur / 2
            min_dist = float('inf')
            for st_ms, en_ms, spk in refined_segs:
                dist = min(abs(mid_t - st_ms), abs(mid_t - en_ms))
                if dist < min_dist:
                    min_dist = dist
                    best_spk = spk

        if best_spk is None:
            best_spk = "1"

        assigned_clauses.append({
            "start": int(curr_start),
            "end": int(c_end),
            "speaker": str(best_spk),
            "text": clause
        })

        curr_start = c_end

    # 合并相邻且相同说话人的子句
    merged_sub = []
    if assigned_clauses:
        curr = assigned_clauses[0]
        for idx in range(1, len(assigned_clauses)):
            nxt = assigned_clauses[idx]
            if nxt["speaker"] == curr["speaker"]:
                curr["text"] += nxt["text"]
                curr["end"] = nxt["end"]
            else:
                merged_sub.append(curr)
                curr = nxt
        merged_sub.append(curr)

    return merged_sub


def _assign_clauses_to_speakers_seamless(asr_start, asr_end, text, seamless_segs):
    """将 ASR 段识别出的带标点 text，按标点分句，分配到无缝说话人时间轴上。
    
    与 _assign_clauses_to_speakers 的区别：
    - seamless_segs 包含确定段(int speaker_id)和未知段(str type:"overlap"/"silence")
    - 优先匹配确定段（取重叠时间最长的说话人）
    - 子句完全落在未知段 → 取同一标点大句内最近确定段的说话人（锚点扩散）
    - 整个大句都没有确定段 → 取时间上最近的确定段说话人（兜底）
    
    Returns:
        list of {"start": int, "end": int, "speaker": str, "text": str}
    """
    if not text.strip():
        return []

    import re
    pattern = r'([^，。？！、；：,.?!;:：\s]+[，。？！、；：,.?!;:：\s]*)'
    clauses = re.findall(pattern, text)
    if not clauses:
        clauses = [text]

    total_len = sum(len(c) for c in clauses)
    if total_len == 0:
        return []

    dur = asr_end - asr_start
    curr_start = asr_start

    # 从 seamless_segs 中提取确定段（用于直接匹配）
    determined_segs = [(st, en, spk) for st, en, spk in seamless_segs if isinstance(spk, int)]

    assigned_clauses = []
    for clause in clauses:
        c_len = len(clause)
        c_dur = dur * (c_len / total_len)
        c_end = curr_start + c_dur

        # 1. 优先匹配确定段
        best_spk = None
        max_overlap = -1.0
        for seg_start_ms, seg_end_ms, spk in determined_segs:
            overlap = min(c_end, seg_end_ms) - max(curr_start, seg_start_ms)
            if overlap > max_overlap:
                max_overlap = overlap
                best_spk = spk

        # 2. 无确定段重叠 → 锚点扩散：取最近确定段
        if max_overlap <= 0 or best_spk is None:
            mid_t = curr_start + c_dur / 2
            min_dist = float('inf')
            for seg_start_ms, seg_end_ms, spk in determined_segs:
                dist = min(abs(mid_t - seg_start_ms), abs(mid_t - seg_end_ms))
                if dist < min_dist:
                    min_dist = dist
                    best_spk = spk

        # 3. 兜底：无任何确定段
        if best_spk is None:
            best_spk = 1

        assigned_clauses.append({
            "start": int(curr_start),
            "end": int(c_end),
            "speaker": str(best_spk),
            "text": clause
        })

        curr_start = c_end

    # 合并相邻相同说话人的子句
    merged_sub = []
    if assigned_clauses:
        curr = assigned_clauses[0]
        for idx in range(1, len(assigned_clauses)):
            nxt = assigned_clauses[idx]
            if nxt["speaker"] == curr["speaker"]:
                curr["text"] += nxt["text"]
                curr["end"] = nxt["end"]
            else:
                merged_sub.append(curr)
                curr = nxt
        merged_sub.append(curr)

    return merged_sub


def _task_diarization(y, num_speakers):
    """在并行管线中运行说话人分离（seg-3.0 + Cam++ + 谱聚类）"""
    t0 = time.time()
    seg_engine = _get_seg_model()
    spk_model = _get_spk_model()
    seamless_segs = spk_model.cluster_with_seamless_segmentation(
        y, segment_engine=seg_engine, sr=16000, n_speakers=num_speakers
    )
    refined_segs = []
    for st_sec, en_sec, val in seamless_segs:
        if isinstance(val, int):
            refined_segs.append((st_sec * 1000, en_sec * 1000, val))
        else:
            refined_segs.append((st_sec * 1000, en_sec * 1000, val))
    refined_segs = sorted(refined_segs, key=lambda x: x[0])
    logger.info(f"[并行] 说话人分离完成，耗时 {time.time()-t0:.1f}s")
    return refined_segs


def _task_asr(audio_path, y, engine_key, duration_ms):
    """在并行管线中运行 VAD + ASR 解码"""
    t0 = time.time()
    vad_out = VAD_MODEL.generate(input=audio_path, batch_size_s=5000, max_single_segment_time=60000)
    raw_segs = vad_out[0]['value'] if vad_out and len(vad_out) > 0 and 'value' in vad_out[0] else [[0, duration_ms]]
    opt_segs = _merge_vad_segments(raw_segs)

    asr_waveforms = []
    final_opt_segs = []
    for start_ms, end_ms in opt_segs:
        s_idx = int(start_ms * 16)
        e_idx = int(end_ms * 16)
        chunk = y[max(0, s_idx - 800): min(len(y), e_idx + 800)]
        if len(chunk) < 1600:
            continue
        asr_waveforms.append(chunk)
        final_opt_segs.append((start_ms, end_ms))

    if not asr_waveforms:
        return None

    texts = _decode(engine_key, asr_waveforms)
    punc_texts = []
    for t in texts:
        cleaned = _clean(t)
        punc_text = _post_punc(cleaned)
        punc_texts.append(punc_text)

    logger.info(f"[并行] ASR 解码完成，耗时 {time.time()-t0:.1f}s")
    return (final_opt_segs, punc_texts)


def _run_inference(audio_path: str, vad_strategy: str = "auto", engine=None, diarize: bool = False,
                   diarize_strategy: str = "two_stage", num_speakers: int = None):
    """同步推理逻辑（在独立线程运行）。

    - 根据 vad_strategy 决定走廉价 trim 直解还是完整 FSMN VAD 切分
    - 根据 engine 覆盖 / 自动路由选择 Sherpa-CPU 或 PyTorch-GPU
    - diarize=True 时先做 VAD 切分，对片段离线聚类，产出段级 [说话人] 标注
    - diarize_strategy: "single" | "two_stage"(默认) | "spectral" — 聚类策略
    返回 (text, engine_key, segments, diarized_text)
    """
    import librosa

    y, sr = librosa.load(audio_path, sr=16000)
    duration_ms = len(y) / sr * 1000

    engine_key = _select_engine(engine, duration_ms)

    if diarize and diarize_strategy == "seg_clustering":
        try:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as pool:
                future_diar = pool.submit(_task_diarization, y, num_speakers)
                future_asr = pool.submit(_task_asr, audio_path, y, engine_key, duration_ms)

                refined_segs = future_diar.result()
                asr_result = future_asr.result()

            if refined_segs is None or asr_result is None:
                raise RuntimeError("并行任务失败")

            final_opt_segs, punc_texts = asr_result

            if not final_opt_segs:
                return ("", engine_key, [], "")

            # 汇合：子句级分配（串行，极快）
            segments = []
            for (asr_start, asr_end), punc_text in zip(final_opt_segs, punc_texts):
                if not punc_text.strip():
                    continue
                sub_segs = _assign_clauses_to_speakers_seamless(asr_start, asr_end, punc_text, refined_segs)
                segments.extend(sub_segs)

            segments = sorted(segments, key=lambda x: x["start"])

            diarized_text = "\n".join(
                f"[说话人{seg['speaker']}] {seg['text']}" for seg in segments if seg["text"].strip()
            )
            raw_text = "\n".join([seg["text"] for seg in segments if seg["text"].strip()])

            return (raw_text, engine_key, segments, diarized_text)
        except Exception as e:
            logger.error(f"并行管线失败: {e}", exc_info=True)
            # 失败则回退原流程

    use_vad = _use_vad(vad_strategy, duration_ms)

    # 说话人分离要求先做语音切分
    if diarize:
        use_vad = True

    if not use_vad:
        # 廉价 trim 直解（防幻觉最低保障，不跑完整 VAD）
        y_trim, _ = _cheap_trim(audio_path)
        waveforms = [y_trim]
    else:
        # 完整 FSMN VAD 切分
        vad_out = VAD_MODEL.generate(input=audio_path, batch_size_s=5000, max_single_segment_time=60000)
        raw_segs = vad_out[0]['value'] if vad_out and len(vad_out) > 0 and 'value' in vad_out[0] else [[0, duration_ms]]

        opt_segs = _merge_vad_segments(raw_segs)

        chunks = []
        seg_meta = []   # 与 chunks 一一对应的 (start_ms, end_ms)，用于段级标注
        for start_ms, end_ms in opt_segs:
            s_idx = int(start_ms * 16)
            e_idx = int(end_ms * 16)
            chunk = y[max(0, s_idx - 800): min(len(y), e_idx + 800)]
            if len(chunk) < 1600:
                continue
            chunks.append(chunk)
            seg_meta.append((start_ms, end_ms))
        waveforms = chunks

    if not waveforms:
        return ("", engine_key, [], "")

    texts = _decode(engine_key, waveforms)
    clean_texts = [_clean(t) for t in texts]   # 与 chunks 一一对齐
    # 全文拼接（向后兼容）：过滤空串后跑一次 PUNC（整句上下文，断句最准）
    joined = "\n".join([t for t in clean_texts if t])
    raw_text = _post_punc(joined)

    # 说话人分离：对切分片段做离线聚类，产出段级 [说话人] 标注
    segments = []
    diarized_text = ""
    if diarize and chunks:
        try:
            if diarize_strategy == "sliding":
                # 滑窗说话人分离：整段音频内部固定窗滑切，逐窗提 Cam++ 向量，
                # 复用 spectral 聚类并合并相邻同人窗。VAD 段仅服务 ASR，不参与分人。
                merged = _get_spk_model().cluster_sliding(
                    y, sr=16000, strategy="spectral",
                    n_speakers=num_speakers, win_sec=1.5, step_sec=0.5,
                )
                for st, en, spk in merged:
                    segments.append({
                        "start": int(st * 1000),
                        "end": int(en * 1000),
                        "speaker": str(spk),
                        "text": "",
                    })
            else:
                spk_cache = _get_spk_model().cluster(
                    chunks, strategy=diarize_strategy, seg_times=seg_meta, n_speakers=num_speakers
                )
                for i, (start_ms, end_ms) in enumerate(seg_meta):
                    spk = str(spk_cache.get(i, "?"))
                    seg_text = clean_texts[i] if i < len(clean_texts) else ""
                    segments.append({
                        "start": start_ms,
                        "end": end_ms,
                        "speaker": spk,
                        "text": seg_text,
                    })
            diarized_text = "\n".join(
                f"[说话人{seg['speaker']}] {seg['text']}" for seg in segments if seg["text"]
            )
        except Exception as spk_err:
            logger.error(f"说话人分离失败，退回无标注: {spk_err}", exc_info=True)

    return (raw_text, engine_key, segments, diarized_text)

@app.post("/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    vad_strategy: str = Form("auto"),
    engine: str = Form(None),
    diarize: bool = Form(False),
    diarize_strategy: str = Form("two_stage"),
    num_speakers: int = Form(None),
    response_format: str = Form("json"),
):
    if MODEL is None or VAD_MODEL is None or PUNC_MODEL is None:
        raise HTTPException(status_code=503, detail="模型未初始化完毕")

    # 1. 安全校验：检查文件大小
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="上传文件过大，限制 50MB 以内")

    start_time = time.time()
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    temp_path = None

    try:
        # 2. 将临时文件生命周期托管于 try 块内
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            content = await file.read()
            await asyncio.to_thread(temp_file.write, content)

        # 3. 校验写入的文件大小
        if os.path.getsize(temp_path) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="文件内容超过 50MB 限制")

        # 4. 并发限制控制与模型推理
        async with GPU_SEMAPHORE:
            text, engine_key, segments, diarized_text = await asyncio.to_thread(
                _run_inference, temp_path, vad_strategy, engine, diarize, diarize_strategy, num_speakers
            )

        latency = (time.time() - start_time) * 1000
        logger.info(f"音频转写完成 (vad_strategy={vad_strategy}, engine={engine_key}, diarize={diarize})，耗时: {latency:.2f} ms")
        
        if response_format == "text":
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(diarized_text if diarize and diarized_text else text)

        resp = {"text": text, "latency_ms": latency, "engine": engine_key, "segments": segments}
        if diarize:
            resp["diarized_text"] = diarized_text
        return resp

        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"语音识别服务内部出错: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="语音识别出错，请联系管理员")
        
    finally:
        # 5. 可靠的垃圾文件清理与异步化
        if temp_path and os.path.exists(temp_path):
            try:
                await asyncio.to_thread(os.remove, temp_path)
            except Exception as e:
                logger.error(f"清理临时文件失败 {temp_path}: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
