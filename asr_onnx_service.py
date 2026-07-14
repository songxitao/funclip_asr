# asr_onnx_service.py — FastAPI 薄路由层
#
# 扩展-收缩策略的"收缩"步：本文件不再包含任何内联推理 / 对齐 / SRT 逻辑，
# 仅保留 FastAPI app + /transcribe 路由 + 启动保活，推理编排全部委托给
# funclip_pro.pipeline.OfflinePipeline（等价原 _run_inference 的收口层）。
#
# 红线：
#   - 时间戳统一 ms（由 core / pipeline 保证，路由层不重复实现）
#   - 启动期 apply_dll_patch() 保活（在首次加载 torch / onnxruntime 前点亮）
#   - 零硬编码盘符 / 绝对路径（模型目录一律经 resolve_model_path 解析）
#   - 仅依赖 funclip_pro 包 + fastapi，不 import asr_service.py
#   - 模块间绝对导入：from funclip_pro.x import Y

import os
import sys
import psutil

try:
    psutil.Process().cpu_affinity([0, 1, 2, 3, 4, 5])
except Exception as e:
    print(f"警告：设置 CPU 亲和性失败: {e}")

for env_var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[env_var] = "6"

# 1. 将项目 src 目录加入 sys.path（兼容无 PYTHONPATH 启动；等价原顶部逻辑）
_src_root = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(_src_root, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# 2. DLL 补丁：动态点亮 onnxruntime GPU 推理（必须在首次加载 torch / onnxruntime 前）
from funclip_pro.config.loader import apply_dll_patch
apply_dll_patch()

import time
import tempfile
import asyncio
import logging

import torch
torch.set_num_threads(6)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
import uvicorn

# 3. 统一转写流水线（收口层：VAD 三态 + 引擎路由 + 说话人分离 + SRT 组装）
from funclip_pro.pipeline import OfflinePipeline
# SRT 响应组装复用 funclip_pro.utils 工具（等价于原 _segments_to_srt / _merge_same_speaker_segments）
from funclip_pro.utils import _segments_to_srt, _merge_same_speaker_segments

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

# 流水线句柄（启动期实例化并加载模型，等价原 load_models 启动钩子）
PIPELINE = None
GPU_SEMAPHORE = asyncio.Semaphore(3)  # 并发限制，防范 CUDA OOM
MAX_FILE_SIZE = 50 * 1024 * 1024      # 50MB 内存防线


@app.on_event("startup")
def load_pipeline():
    """启动期实例化 OfflinePipeline：加载 Sherpa-ASR / VAD / PUNC 模型权重。

    等价于原 @app.on_event("startup") 的 load_models()，DLL 补丁已在导入期点亮。
    torch / spk / seg 引擎保持惰性，首次请求时才构造（由 OfflinePipeline 内部处理）。
    """
    global PIPELINE
    try:
        logger.info("正在初始化 OfflinePipeline（加载 ASR / VAD / PUNC 模型）...")
        PIPELINE = OfflinePipeline(auto_load=True)
        logger.info("OfflinePipeline 初始化成功！")
    except Exception as e:
        logger.error(f"OfflinePipeline 初始化失败: {e}")
        raise e


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
    if PIPELINE is None:
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

        # 4. 并发限制控制与推理（委托 OfflinePipeline，等价原 _run_inference）
        #    返回四元组 (raw_text, engine_key, segments, diarized_text)
        async with GPU_SEMAPHORE:
            text, engine_key, segments, diarized_text = await asyncio.to_thread(
                PIPELINE.run,
                temp_path,
                vad_strategy=vad_strategy,
                diarize=diarize,
                engine=engine,
                diarize_strategy=diarize_strategy,
                num_speakers=num_speakers,
            )

        latency = (time.time() - start_time) * 1000
        logger.info(f"音频转写完成 (vad_strategy={vad_strategy}, engine={engine_key}, diarize={diarize})，耗时: {latency:.2f} ms")

        if response_format == "text":
            return PlainTextResponse(diarized_text if diarize and diarized_text else text)

        if response_format == "srt":
            if diarize and segments:
                merged = _merge_same_speaker_segments(segments)
                srt_text = _segments_to_srt(merged)
            else:
                # 非说话人分离模式：整段文字作为一条字幕（无时间戳信息）
                srt_text = f"1\n00:00:00,000 --> 00:00:00,000\n{text.strip()}\n" if text.strip() else ""
            return PlainTextResponse(srt_text)

        # 默认 json 响应：字段与原服务完全一致（等价优先）
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
