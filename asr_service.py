import os
import time
import tempfile
import asyncio
import re
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from funasr import AutoModel
import uvicorn

app = FastAPI(title="SenseVoice ASR Service", description="极速语音转写微服务")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None

@app.on_event("startup")
def load_model():
    global MODEL
    model_path = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmall"
    print(f"正在 GPU (CUDA) 上加载 SenseVoice 模型: {model_path}...")
    MODEL = AutoModel(
        model=model_path,
        trust_remote_code=True,
        device="cuda",
        disable_update=True
    )
    print("SenseVoice 模型加载成功！")

def _run_inference(audio_path: str) -> str:
    res = MODEL.generate(input=audio_path, cache={}, language="auto", use_itn=True)
    if res and len(res) > 0:
        raw_text = res[0].get('text', '').strip()
        # 过滤 SenseVoice 特有的情感 and 背景事件标签，如 <|happy|>、<|applause|> 等
        clean_text = re.sub(r"<\|.*?\|>", "", raw_text).strip()
        return clean_text
    return ""

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="模型未初始化完毕")
        
    start_time = time.time()
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name
        
    try:
        # 使用 asyncio.to_thread 将同步阻塞的模型计算分发给工作线程运行，防止阻塞主事件循环
        text = await asyncio.to_thread(_run_inference, temp_path)
        latency = (time.time() - start_time) * 1000
        return {"text": text, "latency_ms": latency}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音识别出错: {str(e)}")
        
    finally:
        if os.path.exists(temp_path):
            await asyncio.to_thread(os.remove, temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
