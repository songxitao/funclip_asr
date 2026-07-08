import os
import psutil

# 1. Windows CPU 核心硬性亲和性绑定 (只允许在前4个核心上运行，防止全核拉满死机)
try:
    psutil.Process().cpu_affinity([0, 1, 2, 3])
except Exception as e:
    print(f"警告：设置 CPU 亲和性失败: {e}")

# 2. 设置线程数软防线
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

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
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from funasr import AutoModel
import uvicorn
import torch
torch.set_num_threads(4)

# 添加 SenseVoiceSmall 目录到 PYTHONPATH
sys.path.append(r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmall")
from utils.model_bin import SenseVoiceSmallONNX

class SenseVoiceSmall(SenseVoiceSmallONNX):
    """包装类，重写了初始化和调用，适配用户要求的接口"""
    def __init__(self, model_dir, batch_size=1, quantize=True, device_id="-1", intra_op_num_threads=4, **kwargs):
        super().__init__(model_dir, batch_size=batch_size, device_id=device_id, quantize=quantize, intra_op_num_threads=intra_op_num_threads, **kwargs)
        # 加载 tokens.json 以还原文本
        tokens_path = os.path.join(model_dir, "tokens.json")
        with open(tokens_path, "r", encoding="utf-8") as f:
            self.tokens = json.load(f)

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
            # 返回 torch.Tensor 便于按维度解析
            ctc_logits = torch.from_numpy(ctc_logits).float()
            
            # 支持多 batch 的结果解析
            for b in range(end_idx - beg_idx):
                x = ctc_logits[b, : encoder_out_lens[b].item(), :]
                yseq = x.argmax(dim=-1)
                yseq = torch.unique_consecutive(yseq, dim=-1)

                mask = yseq != self.blank_id
                token_int = yseq[mask].tolist()
                
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
GPU_SEMAPHORE = asyncio.Semaphore(3)  # 并发限制，防范 CUDA OOM
MAX_FILE_SIZE = 50 * 1024 * 1024      # 50MB 内存防线

@app.on_event("startup")
def load_models():
    global MODEL, VAD_MODEL
    model_path = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmall-ONNX"
    vad_path = r"E:\project\funclip-pro\model\models\damo\speech_fsmn_vad_zh-cn-16k-common-pytorch"
    
    logger.info("正在加载 ONNX GPU ASR 模型和 CPU VAD 模型...")
    try:
        # 1. 加载 ASR 语音识别模型 (ONNX GPU)
        MODEL = SenseVoiceSmall(
            model_dir=model_path,
            batch_size=1,
            quantize=True,
            device_id="0",
            intra_op_num_threads=4
        )
        # 2. 加载 VAD 语音活动检测模型（在 CPU 上加载）
        VAD_MODEL = AutoModel(
            model=vad_path,
            trust_remote_code=True,
            device="cpu",
            disable_update=True,
            disable_pbar=True
        )
        # 确保 VAD 在 cpu 上运行
        VAD_MODEL.model.to("cpu")
        VAD_MODEL.kwargs["device"] = "cpu"
        
        logger.info("ASR (ONNX GPU) 和 VAD (CPU) 模型全部加载成功！")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise e

def _run_inference(audio_path: str, vad_split: bool = False) -> str:
    """在独立线程中运行的同步推理逻辑，支持双轨模式"""
    if not vad_split:
        # --- 轨道一：极速单句模式 (适合 Dify 语音对话提问) ---
        res = MODEL(audio_path)
        if res and len(res) > 0:
            raw_text = res[0].strip()
            # 过滤情绪/事件富文本标签
            clean_text = re.sub(r"<\|.*?\|>", "", raw_text).strip()
            return clean_text
        return ""
    else:
        # --- 轨道二：长音频 VAD 切句模式 (适合 Dify 知识库索引入库) ---
        import librosa
        
        # 1. 加载音频波形
        audio, _ = librosa.load(audio_path, sr=16000)
        
        # 2. 运行 VAD 切分得到静音区间
        vad_out = VAD_MODEL.generate(input=audio_path, batch_size_s=5000, max_single_segment_time=60000)
        raw_segs = vad_out[0]['value'] if vad_out and len(vad_out) > 0 and 'value' in vad_out[0] else [[0, len(audio)/16*1000]]
        
        # 3. 合并小静音切片，保证每段大约 8 秒以内
        def _merge_vad_segments(segments, max_gap_ms=300, max_duration_ms=8000):
            if not segments: return []
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
            
        opt_segs = _merge_vad_segments(raw_segs)
        
        # 4. 循环切片进行 ASR 识别并过滤标签，用换行拼接提高 RAG 句子分界质量
        texts = []
        for start_ms, end_ms in opt_segs:
            s_idx = int(start_ms * 16)
            e_idx = int(end_ms * 16)
            chunk = audio[max(0, s_idx-800):min(len(audio), e_idx+800)]
            if len(chunk) < 1600: continue
            
            res = MODEL(chunk)
            if res and len(res) > 0:
                raw = res[0].strip()
                clean = re.sub(r"<\|.*?\|>", "", raw).strip()
                if clean:
                    texts.append(clean)
                    
        return "\n".join(texts)

@app.post("/transcribe")
async def transcribe(
    request: Request, 
    file: UploadFile = File(...), 
    vad_split: bool = Form(False)  # 接收 Form 参数决定是否开启 VAD
):
    if MODEL is None or VAD_MODEL is None:
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
            # 异步化同步写入操作
            await asyncio.to_thread(temp_file.write, content)
            
        # 3. 校验写入的文件大小
        if os.path.getsize(temp_path) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="文件内容超过 50MB 限制")
            
        # 4. 并发限制控制与模型推理
        async with GPU_SEMAPHORE:
            text = await asyncio.to_thread(_run_inference, temp_path, vad_split)
            
        latency = (time.time() - start_time) * 1000
        logger.info(f"音频转写完成 (VAD={vad_split})，耗时: {latency:.2f} ms")
        return {"text": text, "latency_ms": latency}
        
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
    uvicorn.run(app, host="127.0.0.1", port=8002)
