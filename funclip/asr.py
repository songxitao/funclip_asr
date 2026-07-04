import os
import logging
import argparse
import torch
import librosa
import numpy as np
import re
import sys

# 强制刷新缓冲区，确保 WebUI 能实时拿到 print 内容
sys.stdout.reconfigure(encoding='utf-8')

def log_ui(msg):
    """专门用于向 WebUI 发送状态的打印函数"""
    print(msg, flush=True)

# ================= 🚀 1. 环境配置 =================
# 尝试挂载 Git 源码
GIT_SOURCE_PATH = r"E:\FunClip\FunASR"
if os.path.exists(GIT_SOURCE_PATH):
    sys.path.insert(0, GIT_SOURCE_PATH)
    try:
        from funasr.models.fun_asr_nano.model import FunASRNano
    except ImportError:
        pass

from funasr import AutoModel

try:
    from faster_whisper import WhisperModel
    HAS_FASTER = True
except ImportError:
    HAS_FASTER = False

try:
    import whisper as openai_whisper
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# ================= ⚙️ 2. 路径配置 =================
MODEL_ROOT = r"E:\FunClip\FunClip\model\models"
MODEL_PATHS = {
    "seaco":      os.path.join(MODEL_ROOT, "iic", "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"),
    "sensevoice": os.path.join(MODEL_ROOT, "iic", "SenseVoiceSmall"),
    "nano":       r"E:\FunClip\FunClip\model\models\FunAudioLLM\Fun-ASR-Nano-2512",
    "vad":        os.path.join(MODEL_ROOT, "damo", "speech_fsmn_vad_zh-cn-16k-common-pytorch"),
    "punc":       os.path.join(MODEL_ROOT, "damo", "punc_ct-transformer_zh-cn-common-vocab272727-pytorch"),
    "spk":        os.path.join(MODEL_ROOT, "damo", "speech_campplus_sv_zh-cn_16k-common"), 
}

FASTER_MAP = {"turbo": "deepdml/faster-whisper-large-v3-turbo-ct2", "distil-v3.5": "deepdml/faster-distil-whisper-large-v3.5", "large-v3": "large-v3"}
OPENAI_MAP = {"turbo": "large-v3-turbo", "large-v3": "large-v3"}

# ========================================================

class SpeakerDiarizer:
    def __init__(self, spk_model_path, device="cuda"):
        log_ui("   [Init] 加载说话人模型 (Cam++)...")
        self.model = AutoModel(model=spk_model_path, trust_remote_code=True, device=device, disable_update=True, disable_pbar=True)
        self.profiles = [] 
        self.threshold = 0.35 

    def get_speaker_id(self, audio_chunk):
        if len(audio_chunk) < 1600: return "?"
        try:
            res = self.model.generate(input=[audio_chunk], disable_pbar=True)
            if not res or 'spk_embedding' not in res[0]: return "?"
            emb = torch.tensor(res[0]['spk_embedding']).flatten().cpu()
            
            best_score = -1.0
            best_id = -1
            for profile in self.profiles:
                score = torch.nn.functional.cosine_similarity(emb, profile['emb'], dim=0).item()
                if score > best_score:
                    best_score = score
                    best_id = profile['id']
            
            if best_score > self.threshold:
                return best_id
            else:
                new_id = len(self.profiles) + 1
                self.profiles.append({'id': new_id, 'emb': emb})
                return new_id
        except Exception as e:
            return "?"

class SuperASREngine:
    def __init__(self, backend="faster", model_size="turbo", sub_mode="precision", device="cuda"):
        self.backend = backend
        self.device = device
        self.sub_mode = sub_mode 
        self.model = None
        self.vad_model = None 
        self.spk_engine = None 
        
        log_ui(f"🛠️ 初始化引擎: {backend} ({sub_mode})")

        # 加载 VAD
        if backend != "funasr" or sub_mode != "seaco":
            log_ui("   [Init] 加载 VAD 模型...")
            self.vad_model = AutoModel(model=MODEL_PATHS["vad"], trust_remote_code=True, device=device, disable_update=True, disable_pbar=True)

        # 加载 ASR
        log_ui(f"   [Init] 加载主模型: {backend}...")
        if backend == "faster":
            if not HAS_FASTER: raise ImportError("Missing `faster-whisper`")
            model_id = FASTER_MAP.get(model_size, model_size)
            self.model = WhisperModel(model_id, device=device, compute_type="float16")
        elif backend == "funasr":
            if sub_mode == "emotion":
                self.model = AutoModel(model=MODEL_PATHS["sensevoice"], trust_remote_code=True, device=device, disable_update=True)
            elif sub_mode == "nano":
                # Nano 不需要 vad_model 参数，因为我们在外层手动做 VAD
                self.model = AutoModel(model=MODEL_PATHS["nano"], trust_remote_code=True, device=device, disable_update=True, hub="hf")
            else:
                self.model = AutoModel(model=MODEL_PATHS["seaco"], vad_model=MODEL_PATHS["vad"], punc_model=MODEL_PATHS["punc"], spk_model=MODEL_PATHS["spk"], device=device, disable_update=True)

    def run(self, input_path, output_dir, language="auto", enable_spk=False):
        file_stem = os.path.splitext(os.path.basename(input_path))[0]
        final_out_dir = os.path.join(output_dir, file_stem)
        os.makedirs(final_out_dir, exist_ok=True)
        
        log_ui(f"🎬 开始处理: {os.path.basename(input_path)}")
        
        text_res = ""
        srt_res = ""
        lang_arg = None if language == "auto" else language

        # === 轨道一：手动流水线 (Nano/SenseVoice/Whisper) ===
        if self.backend in ["faster", "openai"] or (self.backend == "funasr" and self.sub_mode in ["emotion", "nano"]):
            
            if enable_spk and self.spk_engine is None:
                self.spk_engine = SpeakerDiarizer(MODEL_PATHS["spk"], device=self.device)

            log_ui("⏳ Step 1: VAD 切分...")
            try:
                audio, _ = librosa.load(input_path, sr=16000)
                vad_out = self.vad_model.generate(input=input_path, batch_size_s=5000, max_single_segment_time=60000)
                raw_segs = vad_out[0]['value'] if vad_out and len(vad_out)>0 and 'value' in vad_out[0] else [[0, len(audio)/16*1000]]
                opt_segs = self._merge_vad_segments(raw_segs, max_gap_ms=300, max_duration_ms=8000)
                log_ui(f"✅ 切分完成: {len(opt_segs)} 段")
            except Exception as e:
                log_ui(f"❌ VAD 失败: {e}")
                return

            srt_idx = 1
            prompt_map = {
                "zh": "以下是普通话的句子，请使用简体中文，并添加标点符号。",
                "en": "The following are sentences in English. Please use English and add punctuation.",
                "ja": "以下は日本語の文章です。日本語を使用し、句読点を付けてください。",
                "auto": "Please add punctuation."
            }
            whisper_prompt = prompt_map.get(lang_arg, prompt_map["auto"])
            buffer_text = ""
            buffer_duration = 0.0
            last_spk_id = "?"

            total_segs = len(opt_segs)
            
            # --- 循环处理 ---
            for i, (start_ms, end_ms) in enumerate(opt_segs):
                if i % 10 == 0:
                    log_ui(f"   👉 进度: {i}/{total_segs}")

                s_idx = int(start_ms * 16); e_idx = int(end_ms * 16)
                chunk = audio[max(0, s_idx-800):min(len(audio), e_idx+800)]
                if len(chunk) < 1600: continue
                
                # A. SPK
                current_spk_label = ""
                if enable_spk and self.spk_engine:
                    spk_id = self.spk_engine.get_speaker_id(chunk)
                    if spk_id != "?":
                        current_spk_label = f"[Spk {spk_id}]"
                        last_spk_id = spk_id
                    else:
                        current_spk_label = f"[Spk {last_spk_id}]"

                # B. ASR
                seg_text = ""
                try:
                    if self.backend == "faster":
                        segs, _ = self.model.transcribe(chunk, beam_size=5, language=lang_arg, condition_on_previous_text=False, without_timestamps=True, initial_prompt=whisper_prompt)
                        seg_text = "".join([s.text for s in segs]).strip()
                    
                    elif self.backend == "funasr" and self.sub_mode == "emotion":
                        res = self.model.generate(input=chunk, language=lang_arg, use_itn=True, disable_pbar=True)
                        if res: 
                            raw = res[0].get('text', '').strip()
                            seg_text = self._clean_sensevoice_tags(raw)
                    
                    elif self.backend == "funasr" and self.sub_mode == "nano":
                        # 🔴 核心修复：传递 language 参数给魔改后的 model.py
                        chunk_tensor = torch.from_numpy(chunk) 
                        res = self.model.generate(
                            input=[chunk_tensor], 
                            batch_size_s=0, 
                            disable_pbar=True,
                            language=lang_arg  # <--- 这里的语言参数现在会被 model.py 识别了！
                        )
                        if res: seg_text = res[0].get('text', '').strip()

                except Exception as e:
                    log_ui(f"⚠️ 识别出错 ({start_ms}ms): {e}")
                    continue

                if not seg_text: continue
                
                # C. 字幕处理 (缓冲拼接)
                current_start_sec = start_ms / 1000.0
                current_end_sec = end_ms / 1000.0
                current_duration = current_end_sec - current_start_sec

                if buffer_text:
                    seg_text = f"{buffer_text} {seg_text}"
                    current_start_sec -= buffer_duration
                    buffer_text = ""
                    buffer_duration = 0.0

                split_lines = self._split_text_smartly(seg_text, max_chars=30)
                
                if split_lines:
                    last_sent = split_lines[-1]
                    is_complete = re.search(r'[\.?!。？！]$', last_sent)
                    if not is_complete and i < len(opt_segs) - 1 and len(last_sent) < 20:
                        buffer_text = last_sent
                        total_len = sum(len(s) for s in split_lines)
                        if total_len > 0:
                            buffer_duration = (len(buffer_text) / total_len) * current_duration
                        split_lines.pop()
                        current_end_sec -= buffer_duration

                valid_chars = sum(len(s) for s in split_lines)
                valid_dur = current_end_sec - current_start_sec
                curr_srt_start = current_start_sec
                
                for line in split_lines:
                    if not line.strip(): continue
                    display_text = f"{current_spk_label} {line}".strip()
                    
                    if valid_chars > 0:
                        line_dur = (len(line) / valid_chars) * valid_dur
                    else:
                        line_dur = valid_dur
                    
                    curr_srt_end = curr_srt_start + line_dur
                    
                    text_res += display_text + "\n"
                    srt_res += self._make_srt_block(srt_idx, curr_srt_start, curr_srt_end, display_text)
                    srt_idx += 1
                    curr_srt_start = curr_srt_end
                
                # 打印预览 (仅前几条)
                if i < 3: log_ui(f"   👀 预览: {display_text}")

            if buffer_text:
                final_txt = f"{'[Spk ?]' if enable_spk else ''} {buffer_text}".strip()
                srt_res += self._make_srt_block(srt_idx, current_end_sec, current_end_sec + buffer_duration, final_txt)

        # === 轨道二：SeACo (Legacy) ===
        else:
            log_ui("🔄 Running SeACo Pipeline...")
            try:
                res = self.model.generate(input=input_path, batch_size_s=300, sentence_timestamp=True, return_spk_res=enable_spk)
                sentence_info = res[0].get('sentence_info', [])
                if sentence_info:
                    srt_idx = 1
                    for sent in sentence_info:
                        text = sent.get('text', '')
                        spk_tag = f"[Spk {sent.get('spk')}] " if enable_spk and 'spk' in sent else ""
                        full_line = f"{spk_tag}{text}"
                        start = sent['timestamp'][0][0] / 1000.0
                        end = sent['timestamp'][-1][1] / 1000.0
                        text_res += full_line + "\n"
                        srt_res += self._make_srt_block(srt_idx, start, end, full_line)
                        srt_idx += 1
            except Exception as e:
                log_ui(f"❌ SeACo Error: {e}")

        # === 强制保存 ===
        txt_path = os.path.join(final_out_dir, f"{file_stem}.txt")
        srt_path = os.path.join(final_out_dir, f"{file_stem}.srt")
        
        log_ui("💾 正在写入文件...")
        with open(txt_path, "w", encoding="utf-8") as f: 
            f.write(text_res if text_res else "(无识别内容)")
        with open(srt_path, "w", encoding="utf-8") as f: 
            f.write(srt_res if srt_res else "")
            
        log_ui(f"🎉 全部完成！\nTXT: {txt_path}\nSRT: {srt_path}")

    # --- 辅助函数 ---
    def _split_text_smartly(self, text, max_chars=25):
        if not text: return []
        strong_pattern = r'([\.?!。？！][”"’\']?)'
        chunks = re.split(strong_pattern, text)
        strong_sentences = []
        curr = ""
        for chunk in chunks:
            curr += chunk
            if re.search(strong_pattern, chunk):
                strong_sentences.append(curr.strip())
                curr = ""
        if curr.strip(): strong_sentences.append(curr.strip())
        
        final_sentences = []
        for sent in strong_sentences:
            if len(sent) > max_chars:
                weak_pattern = r'([,，、])'
                subs = re.split(weak_pattern, sent)
                sub_curr = ""
                for sub in subs:
                    sub_curr += sub
                    if re.search(weak_pattern, sub) and len(sub_curr) > 5:
                        final_sentences.append(sub_curr.strip())
                        sub_curr = ""
                if sub_curr.strip(): final_sentences.append(sub_curr.strip())
            else:
                final_sentences.append(sent)
        return [s for s in final_sentences if s]

    def _clean_sensevoice_tags(self, text):
        return re.sub(r"<\|.*?\|>", "", text).strip()

    def _merge_vad_segments(self, segments, max_gap_ms=300, max_duration_ms=15000):
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

    def _make_srt_block(self, idx, start, end, text):
        if not text.strip(): return ""
        if end <= start: end = start + 0.01
        def fmt(t):
            h, r = divmod(t, 3600)
            m, s = divmod(r, 60)
            return f"{int(h):02}:{int(m):02}:{int(s):02},{int((t%1)*1000):03}"
        return f"{idx}\n{fmt(start)} --> {fmt(end)}\n{text}\n\n"

    def _make_fake_srt(self, text): 
        return f"1\n00:00:00,000 --> 00:00:10,000\n{text}\n\n"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--backend", default="faster") 
    parser.add_argument("--model_size", default="turbo")
    parser.add_argument("--sub_mode", default="precision")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--enable_spk", action="store_true")
    
    args = parser.parse_args()

    engine = SuperASREngine(
        backend=args.backend,
        model_size=args.model_size,
        sub_mode=args.sub_mode
    )
    
    engine.run(
        args.file, 
        args.output_dir, 
        language=args.language,
        enable_spk=args.enable_spk
    )