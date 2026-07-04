import os
import time
import torch
import librosa
from .base import ASREngine
from ..vad.utils import merge_vad_segments, make_srt_block
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading
from funasr import AutoModel # Unified import for ASR
from modelscope.pipelines import pipeline # Back for VAD
from modelscope.utils.constant import Tasks

# Configuration
MODEL_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "model", "models")
# Define model paths map
MODEL_PATHS = {
    "vad": os.path.join(MODEL_ROOT, "iic", "speech_fsmn_vad_zh-cn-16k-common-pytorch"),
    "punc": os.path.join(MODEL_ROOT, "iic", "punc_ct-transformer_zh-cn-common-vocab272727-pytorch"),
    "spk": os.path.join(MODEL_ROOT, "iic", "speech_campplus_sv_zh-cn_16k-common"),
    "sensevoice": os.path.join(MODEL_ROOT, "iic", "SenseVoiceSmall"),
    "nano": os.path.join(MODEL_ROOT, "iic", "FunASR-Nano-Zh"),
    "seaco": os.path.join(MODEL_ROOT, "iic", "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"),
}

class FunASREngine(ASREngine):
    def __init__(self, device="cuda", sub_mode="nano", **kwargs):
        super().__init__(device, **kwargs)
        self.sub_mode = sub_mode # "nano", "sensevoice"
        self.model = None
        self.vad_model = None
        self.spk_engine = None

    def load_model(self):
        self.log(f"🛠️ 初始化 FunASR 引擎 ({self.sub_mode})...")
        
        # 1. Load VAD (Always needed for Manual Pipeline)
        self.log("   [Init] 加载 VAD 模型... (FSMN/Pipeline)")
        self.vad_model = pipeline(
            task=Tasks.voice_activity_detection,
            model=MODEL_PATHS["vad"],
            model_revision=None,
            device=self.device
        )
        
        # 2. Load ASR Model
        asr_path = MODEL_PATHS.get(self.sub_mode, MODEL_PATHS["sensevoice"])
        self.log(f"   [Init] 加载主模型: {self.sub_mode}...")
        
        self.model = AutoModel(
            model=asr_path,
            trust_remote_code=True,
            device=self.device,
            disable_update=True
        )
        self.log("✅ 模型加载完成")

    def transcribe(self, audio_path, language="auto", batch_size=8, hotwords=None, **kwargs):
        self.log(f"🎬 [FunASR] 开始处理: {os.path.basename(audio_path)}")
        t_start = time.time()
        
        # 1. VAD Split
        self.log("⏳ Step 1: VAD 切分...")
        audio, _ = librosa.load(audio_path, sr=16000)
        
        # VAD Parameters (Aggressive)
        vad_kwargs = {
            "max_single_segment_time": 12000,
            "speech_noise_thres": 0.9,
            "max_end_silence_time": 200,
        }
        res = self.vad_model(input=audio_path, batch_size_s=5000, **vad_kwargs)
        
        raw_segs = res[0]['value'] if res and len(res)>0 and 'value' in res[0] else []
        opt_segs = merge_vad_segments(raw_segs, max_duration_ms=12000) # Ensure chunks < 12s
        
        self.log(f"✅ VAD 切分完成: {len(opt_segs)} 段")
        
        # 2. Batch Processing
        self.log(f"🚀 激活 FunASR 并发加速模式 (Batch Size: {batch_size})...")
        
        full_text = ""
        full_srt = ""
        processed_count = 0
        srt_idx = 1
        
        # Prepare Tensors
        def extract_chunk(seg_info):
            s_ms, e_ms, idx = seg_info
            s_idx = int(s_ms * 16)
            e_idx = int(e_ms * 16)
            # Add padding
            chunk = audio[max(0, s_idx-800):min(len(audio), e_idx+800)]
            tensor = torch.from_numpy(chunk).float()
            if self.device == "cuda":
                tensor = tensor.pin_memory()
            return tensor, (s_ms, e_ms)

        # Threaded Preloading
        all_infos = [(s, e, i) for i, (s, e) in enumerate(opt_segs)]
        all_tensors = []
        all_times = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            for tensor, times in executor.map(extract_chunk, all_infos):
                all_tensors.append(tensor)
                all_times.append(times)
                
        # Batch Loop
        batch_size = int(batch_size) if batch_size else 8
        
        for i in range(0, len(all_tensors), batch_size):
            batch_tensors = all_tensors[i:i+batch_size]
            batch_times = all_times[i:i+batch_size]
            
            # Helper to generate results
            try:
                # Move to GPU
                input_tensors = [t.to(self.device, non_blocking=True) for t in batch_tensors]
                
                # Inference
                res_batch = self.model.generate(
                    input=input_tensors,
                    batch_size_s=0, # Disable internal batching
                    disable_pbar=True,
                    language=language,
                    use_itn=True,
                    hotwords=hotwords or ""
                )
                
                # Format Results
                for j, res in enumerate(res_batch):
                    text = res.get('text', '').strip()
                    if not text: continue
                    
                    s_ms, e_ms = batch_times[j]
                    
                    # LOG
                    self.log(f"   Using Batch: {text[:30]}...")
                    
                    full_text += text + "\n"
                    full_srt += make_srt_block(srt_idx, s_ms/1000.0, e_ms/1000.0, text)
                    srt_idx += 1
                
                processed_count += len(batch_tensors)
                pct = processed_count / len(opt_segs) * 100
                self.log(f"   ⚡ [{pct:.1f}%] 已处理 {processed_count}/{len(opt_segs)}")
                
            except Exception as e:
                self.log(f"❌ Batch Error: {e}")
                
        return {
            "text": full_text,
            "srt": full_srt
        }
