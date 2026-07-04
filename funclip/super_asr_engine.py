import os
import sys
import logging
import argparse
import torch
import math
import numpy as np
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [SuperASR] - %(message)s')

# ================= 配置区 =================
MODEL_ROOT = r"E:\FunClip\FunClip\model\models"

MODEL_PATHS = {
    "seaco": os.path.join(MODEL_ROOT, "iic", "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"),
    "vad":   os.path.join(MODEL_ROOT, "damo", "speech_fsmn_vad_zh-cn-16k-common-pytorch"),
    "punc":  os.path.join(MODEL_ROOT, "damo", "punc_ct-transformer_zh-cn-common-vocab272727-pytorch"),
    "spk":   os.path.join(MODEL_ROOT, "damo", "speech_campplus_sv_zh-cn_16k-common"), 
}

SENSE_VOICE_PATH = os.path.join(MODEL_ROOT, "iic", "SenseVoiceSmall")
# =========================================

class SuperASREngine:
    def __init__(self, mode="precision", device="cuda", whisper_size="large-v3"):
        self.mode = mode
        self.device = device
        self.whisper_model = None
        self.vad_model = None
        
        logging.info(f"Initializing SuperASR Engine in [{mode.upper()}] mode on [{device}]...")

        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logging.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {vram:.2f}GB")
        
        try:
            if mode == "precision":
                logging.info("Loading SeACo + CAM++ 200k...")
                self.model = AutoModel(
                    model=MODEL_PATHS["seaco"], vad_model=MODEL_PATHS["vad"], punc_model=MODEL_PATHS["punc"], spk_model=MODEL_PATHS["spk"],
                    device=device, disable_update=True
                )
            elif mode == "emotion":
                logging.info("Loading SenseVoiceSmall (With CTC Timestamp)...")
                self.model = AutoModel(
                    model=SENSE_VOICE_PATH, vad_model=MODEL_PATHS["vad"], spk_model=MODEL_PATHS["spk"], 
                    trust_remote_code=True, device=device, disable_update=True
                )
            elif mode == "whisper":
                logging.info(f"Loading OpenAI Whisper ({whisper_size}) + FSMN VAD...")
                import whisper
                local_whisper_path = fr"E:\FunClip\FunClip\model\models\openai\{whisper_size}.pt"
                if os.path.exists(local_whisper_path):
                     self.whisper_model = whisper.load_model(local_whisper_path, device=device)
                else:
                     self.whisper_model = whisper.load_model(whisper_size, device=device)
                
                self.vad_model = AutoModel(model=MODEL_PATHS["vad"], disable_update=True, device=device)
                
            logging.info("✅ Model loaded successfully!")
            
        except Exception as e:
            logging.error(f"❌ Model loading failed: {e}")
            raise e

    def run(self, input_path, output_dir, hotwords="", language="auto"):
        os.makedirs(output_dir, exist_ok=True)
        file_stem = os.path.splitext(os.path.basename(input_path))[0]
        logging.info(f"Processing: {input_path}")
        
        text_content = ""
        srt_content = ""
        srt_index = 1 

        if self.mode == "whisper":
            # --- Whisper VAD Pipeline ---
            logging.info("Running VAD segmentation...")
            vad_res = self.vad_model.generate(input=input_path)
            raw_segments = vad_res[0]['value'] if vad_res and 'value' in vad_res[0] else []
            
            if not raw_segments:
                logging.warning("VAD found no speech! Fallback to full file.")
                # 构造一个假的全长片段
                import whisper
                audio = whisper.load_audio(input_path)
                merged_segments = [[0, len(audio)/16000*1000]]
            else:
                logging.info(f"Raw VAD segments: {len(raw_segments)}")
                
                # 【新增核心逻辑：智能合并碎片段】
                # 目的：减少 Whisper 调用次数，提高连贯性
                # 规则：如果两个片段间隔小于 500ms，且合并后总长不超过 25秒，就拼起来
                merged_segments = []
                if len(raw_segments) > 0:
                    current_seg = raw_segments[0] # [start, end]
                    
                    for next_seg in raw_segments[1:]:
                        gap = next_seg[0] - current_seg[1] # 间隔时间(ms)
                        duration = next_seg[1] - current_seg[0] # 合并后的总时长(ms)
                        
                        if gap < 800 and duration < 25000: # 容忍800ms静音，最长25秒
                            # 合并：延长当前片段的结束时间
                            current_seg[1] = next_seg[1]
                        else:
                            # 存入当前片段，开始新的片段
                            merged_segments.append(current_seg)
                            current_seg = next_seg
                    
                    merged_segments.append(current_seg) # 存入最后一个
                
                logging.info(f"⚡ Optimized segments: {len(raw_segments)} -> {len(merged_segments)} (Merged short gaps)")

                import whisper
                audio = whisper.load_audio(input_path)

            decode_lang = language if language != "auto" else None
            
            # 【设置阈值】 32000 samples = 2.0 seconds (at 16k sample rate)
            # 只有大于这个长度的片段才会被处理
            MIN_SAMPLES = 32000 

            logging.info(f"Starting Whisper Inference (Filter < {MIN_SAMPLES/16000}s)...")
            
            total_segments = len(merged_segments)
            processed_count = 0

            for i, (beg_ms, end_ms) in enumerate(merged_segments):
                # 切片
                chunk = audio[int(beg_ms*16):int(end_ms*16)]
                
                # 【过滤逻辑】
                if len(chunk) < MIN_SAMPLES:
                    # 太短了，认为是噪音或无意义语气词，跳过
                    continue
                    
                # Whisper 推理
                processed_count += 1
                res = self.whisper_model.transcribe(chunk, language=decode_lang)
                
                if 'segments' in res:
                    for seg in res['segments']:
                        abs_start = (beg_ms / 1000.0) + seg['start']
                        abs_end   = (beg_ms / 1000.0) + seg['end']
                        sentence  = seg['text'].strip()
                        
                        if sentence:
                            text_content += sentence + " "
                            srt_content += self._make_srt_block(srt_index, abs_start, abs_end, sentence)
                            srt_index += 1
                            
                            def sec2str(s):
                                h, r = divmod(s, 3600)
                                m, sec = divmod(r, 60)
                                return f"{int(h):02d}:{int(m):02d}:{int(sec):02d}"
                            
                            progress_tag = f"[{i+1}/{total_segments}]"
                            time_tag = f"[{sec2str(abs_start)} -> {sec2str(abs_end)}]"
                            print(f"{progress_tag} {time_tag} {sentence}")

        else:
            # --- FunASR Logic (保持不变) ---
            generate_kwargs = {
                "input": input_path, "batch_size_s": 300, "return_spk_res": True,
            }
            if self.mode == "precision":
                generate_kwargs["hotword"] = hotwords
                generate_kwargs["sentence_timestamp"] = True
            else:
                generate_kwargs["use_itn"] = True
                generate_kwargs["language"] = "auto"
                generate_kwargs["output_timestamp"] = True

            try:
                res = self.model.generate(**generate_kwargs)
            except Exception as e:
                logging.error(f"Inference error: {e}")
                generate_kwargs["return_spk_res"] = False
                res = self.model.generate(**generate_kwargs)

            result_item = res[0]
            text_content = result_item.get('text', '')
            if self.mode == "emotion":
                try: text_content = rich_transcription_postprocess(text_content)
                except: pass

            if 'sentence_info' in result_item and result_item['sentence_info']:
                for i, sent in enumerate(result_item['sentence_info']):
                    start = sent['timestamp'][0][0] / 1000.0
                    end = sent['timestamp'][-1][1] / 1000.0
                    text = sent['text']
                    spk = sent.get('spk', '')
                    if spk: text = f"[Spk{spk}] {text}"
                    srt_content += self._make_srt_block(srt_index, start, end, text)
                    srt_index += 1
            elif 'timestamp' in result_item and result_item['timestamp']:
                 raw_ts = result_item['timestamp']
                 current_sentence = ""
                 current_start = 0.0
                 sent_idx_local = 0
                 for j, item in enumerate(raw_ts):
                    if len(item)==3: char,s,e = item
                    elif len(item)==2: s,e = item; char=text_content[sent_idx_local] if sent_idx_local<len(text_content) else ""; sent_idx_local+=1
                    else: continue
                    s, e = float(s)/1000.0, float(e)/1000.0
                    if current_sentence=="": current_start=s
                    current_sentence+=str(char)
                    if str(char) in ["。", "？", "！", "，", "\n"] or len(current_sentence)>25:
                        srt_content += self._make_srt_block(srt_index, current_start, e, current_sentence)
                        srt_index += 1
                        current_sentence=""
                 if current_sentence:
                    srt_content += self._make_srt_block(srt_index, current_start, e, current_sentence)
                    srt_index += 1
            elif not srt_content:
                srt_content = self._make_fake_srt(text_content)

        # 写入
        txt_out = os.path.join(output_dir, f"{file_stem}.txt")
        srt_out = os.path.join(output_dir, f"{file_stem}.srt")
        with open(txt_out, "w", encoding="utf-8") as f: f.write(text_content)
        with open(srt_out, "w", encoding="utf-8") as f: f.write(srt_content)
        
        logging.info(f"Saved result to: {output_dir}")
        return text_content

    def _make_srt_block(self, idx, start, end, text):
        def fmt(t):
            h, r = divmod(t, 3600)
            m, s = divmod(r, 60)
            return f"{int(h):02}:{int(m):02}:{int(s):02},{int((t%1)*1000):03}"
        return f"{idx}\n{fmt(start)} --> {fmt(end)}\n{text}\n\n"

    def _make_fake_srt(self, text):
        content = ""
        chars_per_block = 50
        total = math.ceil(len(text) / chars_per_block)
        for i in range(total):
            chunk = text[i*chars_per_block : (i+1)*chars_per_block]
            content += self._make_srt_block(i, i*5, (i+1)*5, chunk)
        return content

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", default="precision")
    parser.add_argument("--hotword", default="")
    parser.add_argument("--whisper_size", default="large-v3")
    parser.add_argument("--language", default="auto")
    args = parser.parse_args()

    files = [args.file]
    if os.path.isdir(args.file):
        files = [os.path.join(args.file, f) for f in os.listdir(args.file) 
                 if f.lower().endswith(('.wav','.mp3','.mp4','.mkv','.mov'))]

    engine = SuperASREngine(mode=args.mode, whisper_size=args.whisper_size)

    for f in files:
        sub_out = os.path.join(args.output_dir, os.path.splitext(os.path.basename(f))[0])
        try:
            engine.run(f, sub_out, args.hotword, language=args.language)
        except Exception as e:
            logging.error(f"Error processing {f}: {e}")

if __name__ == "__main__":
    main()