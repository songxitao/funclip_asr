import os
import time
import tempfile
import asyncio
import re
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from funasr import AutoModel
import uvicorn

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
    model_path = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmall"
    vad_path = r"E:\project\funclip-pro\model\models\damo\speech_fsmn_vad_zh-cn-16k-common-pytorch"
    
    logger.info("正在 GPU (CUDA) 上加载 ASR 和 VAD 模型...")
    try:
        # 1. 加载 ASR 语音识别模型
        MODEL = AutoModel(
            model=model_path,
            trust_remote_code=True,
            device="cuda",
            disable_update=True
        )
        # 2. 加载 VAD 语音活动检测模型（用于长音频切句）
        VAD_MODEL = AutoModel(
            model=vad_path,
            trust_remote_code=True,
            device="cuda",
            disable_update=True,
            disable_pbar=True
        )
        logger.info("ASR 和 VAD 模型全部加载成功！")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise e

def _run_inference(audio_path: str, vad_split: bool = False) -> str:
    """在独立线程中运行的同步推理逻辑，支持双轨模式"""
    if not vad_split:
        # --- 轨道一：极速单句模式 (适合 Dify 语音对话提问) ---
        res = MODEL.generate(input=audio_path, cache={}, language="auto", use_itn=True)
        if res and len(res) > 0:
            raw_text = res[0].get('text', '').strip()
            # 过滤情绪/事件富文本标签
            clean_text = re.sub(r"<\|.*?\|>", "", raw_text).strip()
            return clean_text
        return ""
    else:
        # --- 轨道二：长音频 VAD 切句模式 (适合 Dify 知识库索引入库) ---
        import librosa
        import torch
        
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
            
            res = MODEL.generate(input=chunk, cache={}, language="auto", use_itn=True)
            if res:
                raw = res[0].get('text', '').strip()
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
    uvicorn.run(app, host="127.0.0.1", port=8001)
