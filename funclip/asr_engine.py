import os
import logging
import argparse
import torch
import librosa
import numpy as np
import re
from funasr import AutoModel

# 尝试导入两种 Whisper 库
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [SuperASR] - %(message)s')

# ================= 用户配置区 =================
MODEL_ROOT = r"E:\FunClip\FunClip\model\models"

MODEL_PATHS = {
    "seaco": os.path.join(MODEL_ROOT, "iic", "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"),
    "vad":   os.path.join(MODEL_ROOT, "damo", "speech_fsmn_vad_zh-cn-16k-common-pytorch"),
    "punc":  os.path.join(MODEL_ROOT, "damo", "punc_ct-transformer_zh-cn-common-vocab272727-pytorch"),
    "spk":   os.path.join(MODEL_ROOT, "damo", "speech_campplus_sv_zh-cn_16k-common"), 
}
SENSE_VOICE_PATH = os.path.join(MODEL_ROOT, "iic", "SenseVoiceSmall")

FASTER_MAP = {
    "turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "distil-v3.5": "deepdml/faster-distil-whisper-large-v3.5",
    "large-v3": "large-v3"
}

OPENAI_MAP = {
    "turbo": "large-v3-turbo", 
    "large-v3": "large-v3"
}
# ============================================

class SuperASREngine:
    def __init__(self, backend="faster", model_size="turbo", sub_mode="precision", device="cuda"):
        self.backend = backend
        self.device = device
        self.sub_mode = sub_mode 
        self.model = None
        self.vad_model = None 
        
        logging.info(f"Initializing Engine: Backend=[{backend}], SubMode=[{sub_mode}]...")

        # === 1. 加载独立的 VAD 模型 (手动切片神器) ===
        # 只要不是 SeACo，全部走手动切片，这是最稳的
        if backend != "funasr" or sub_mode == "emotion":
            logging.info("Loading Standalone VAD Model...")
            self.vad_model = AutoModel(
                model=MODEL_PATHS["vad"],
                trust_remote_code=True,
                device=device,
                disable_update=True,
                disable_pbar=True
            )

        # === 2. 加载主模型 ===
        if backend == "faster":
            if not HAS_FASTER: raise ImportError("Missing `faster-whisper` package.")
            model_id = FASTER_MAP.get(model_size, model_size)
            self.model = WhisperModel(model_id, device=device, compute_type="float16")

        elif backend == "openai":
            if not HAS_OPENAI: raise ImportError("Missing `openai-whisper` package.")
            model_name = OPENAI_MAP.get(model_size, model_size)
            self.model = openai_whisper.load_model(model_name, device=device)

        elif backend == "funasr":
            if sub_mode == "emotion":
                logging.info("Loading FunASR (SenseVoice) - Clean Mode...")
                # 放弃加载本地补丁，使用原版加载方式，避免 Windows 路径报错
                self.model = AutoModel(
                    model=SENSE_VOICE_PATH,
                    trust_remote_code=True,
                    device=device,
                    disable_update=True
                )
            else:
                logging.info("Loading FunASR (SeACo)...")
                self.model = AutoModel(
                    model=MODEL_PATHS["seaco"],
                    vad_model=MODEL_PATHS["vad"],
                    punc_model=MODEL_PATHS["punc"],
                    spk_model=MODEL_PATHS["spk"], 
                    device=device, disable_update=True
                )

    # def run(self, input_path, output_dir, language="auto", enable_spk=False):
    #     file_stem = os.path.splitext(os.path.basename(input_path))[0]
    #     final_out_dir = os.path.join(output_dir, file_stem)
    #     os.makedirs(final_out_dir, exist_ok=True)
        
    #     text_res = ""
    #     srt_res = ""
    #     lang_arg = None if language == "auto" else language

    #     # =========================================================================
    #     # 策略 A: 手动 VAD 切片模式 (Whisper & SenseVoice)
    #     # =========================================================================
    #     if self.backend in ["faster", "openai"] or (self.backend == "funasr" and self.sub_mode == "emotion"):
            
    #         logging.info(f"Running Manual VAD Pipeline (Slicing)...")
            
    #         # 1. 读取音频
    #         audio, _ = librosa.load(input_path, sr=16000)
            
    #         # 2. VAD 扫描
    #         logging.info("Running VAD Scan...")
    #         vad_out = self.vad_model.generate(input=input_path, batch_size_s=5000, max_single_segment_time=60000)
    #         raw_segs = vad_out[0]['value'] if vad_out and len(vad_out)>0 and 'value' in vad_out[0] else [[0, len(audio)/16*1000]]
            
    #         # 3. 优化切分 (控制在 15秒 以内)
    #         opt_segs = self._merge_vad_segments(raw_segs, max_gap_ms=500, max_duration_ms=15000)
    #         logging.info(f"Processing {len(opt_segs)} segments...")

    #         srt_idx = 1
    #         whisper_prompt = "以下是普通话的句子，请使用简体中文，并添加标点符号。"

    #         for i, (start_ms, end_ms) in enumerate(opt_segs):
    #             s_idx = int(start_ms * 16); e_idx = int(end_ms * 16)
    #             chunk = audio[max(0, s_idx-800):min(len(audio), e_idx+800)]
    #             if len(chunk) < 1600: continue
                
    #             seg_text = ""
                
    #             # --- Whisper 推理 ---
    #             if self.backend == "faster":
    #                 segs, _ = self.model.transcribe(chunk, beam_size=5, language=lang_arg, condition_on_previous_text=False, without_timestamps=True, initial_prompt=whisper_prompt)
    #                 seg_text = "".join([s.text for s in segs]).strip()
    #             elif self.backend == "openai":
    #                 result = self.model.transcribe(chunk, beam_size=5, language=lang_arg, condition_on_previous_text=False, no_speech_threshold=0.6, initial_prompt=whisper_prompt)
    #                 seg_text = result['text'].strip()
                
    #             # --- SenseVoice 推理 (Emotion) ---
    #             elif self.backend == "funasr" and self.sub_mode == "emotion":
    #                     # 使用 generate 接口
    #                 res = self.model.generate(
    #                         input=chunk, 
    #                         language=lang_arg,  # <--- 核心：这里必须是变量，不能写死 "auto"
    #                         use_itn=True
    #                     )
    #                 if res and len(res) > 0:
    #                     raw_text = res[0].get('text', '').strip()
    #                     # 智能清洗标签
    #                     #seg_text = self._clean_sensevoice_tags(raw_text)

    #             # --- 结果写入 (关键优化：智能拆分) ---
    #             if seg_text:
    #                 # 1. 清理
    #                 seg_text = seg_text.replace("\n", " ")
    #                 # 2. 写入纯文本 (保持原样)
    #                 text_res += seg_text + "\n"
                    
    #                 # 3. 【核心】智能拆分字幕
    #                 # 将 15秒 的长文本，按标点拆分成多行短字幕
    #                 split_lines = self._split_text_smartly(seg_text, max_chars=25)
                    
    #                 total_duration = (end_ms - start_ms) / 1000.0
    #                 total_chars = sum(len(x) for x in split_lines)
    #                 current_start = start_ms / 1000.0
                    
    #                 for line in split_lines:
    #                     # 根据字数比例分配时间
    #                     if total_chars > 0:
    #                         line_dur = (len(line) / total_chars) * total_duration
    #                     else:
    #                         line_dur = total_duration
                        
    #                     current_end = current_start + line_dur
                        
    #                     srt_res += self._make_srt_block(srt_idx, current_start, current_end, line)
    #                     srt_idx += 1
    #                     current_start = current_end
                    
    #                 if i % 5 == 0:
    #                     print(f"[{i+1}/{len(opt_segs)}] {seg_text[:30]}...")
    def run(self, input_path, output_dir, language="auto", enable_spk=False):
        file_stem = os.path.splitext(os.path.basename(input_path))[0]
        final_out_dir = os.path.join(output_dir, file_stem)
        os.makedirs(final_out_dir, exist_ok=True)
        
        text_res = ""
        srt_res = ""
        lang_arg = None if language == "auto" else language

        # =========================================================================
        # 策略 A: 手动 VAD 切片模式 (Whisper & SenseVoice)
        # =========================================================================
        if self.backend in ["faster", "openai"] or (self.backend == "funasr" and self.sub_mode == "emotion"):
            
            logging.info(f"Running Manual VAD Pipeline (Slicing)...")
            audio, _ = librosa.load(input_path, sr=16000)
            vad_out = self.vad_model.generate(input=input_path, batch_size_s=5000, max_single_segment_time=60000)
            raw_segs = vad_out[0]['value'] if vad_out and len(vad_out)>0 and 'value' in vad_out[0] else [[0, len(audio)/16*1000]]
            
            # 使用更激进的切分策略 (8s)
            opt_segs = self._merge_vad_segments(raw_segs, max_gap_ms=500, max_duration_ms=8000)
            logging.info(f"Processing {len(opt_segs)} segments...")

            srt_idx = 1
            # whisper_prompt = "以下是普通话的句子，请使用简体中文，并添加标点符号。"
            prompt_map = {
                "zh": "以下是普通话的句子，请使用简体中文，并添加标点符号。",
                "en": "Hello, welcome. Please use English, add punctuation, and split sentences naturally.",
                "ja": "こんにちは。日本語で書き起こし、句読点を追加してください。",
                "auto": "Please add punctuation to the output."  # 默认兜底
            }

            # 1. 获取动态 Prompt
            # self.run 函数的入参里有 language="auto" 
            # 我们优先使用传入的 language，如果没有在表里，就用 auto
            target_lang = lang_arg if lang_arg in prompt_map else "auto"
            whisper_prompt = prompt_map[target_lang]
            # --- 🌟 核心升级：跨段拼接缓冲区 ---
            # buffer_text: 存储上一段剩下的“尾巴” (如 "and bisexual")
            # buffer_duration: 存储这个尾巴占用的时间 (秒)
            buffer_text = ""
            buffer_duration = 0.0

            for i, (start_ms, end_ms) in enumerate(opt_segs):
                s_idx = int(start_ms * 16); e_idx = int(end_ms * 16)
                chunk = audio[max(0, s_idx-800):min(len(audio), e_idx+800)]
                if len(chunk) < 1600: continue
                
                # 1. 执行推理
                seg_text = ""
                if self.backend == "faster":
                    segs, _ = self.model.transcribe(chunk, beam_size=5, language=lang_arg, condition_on_previous_text=False, without_timestamps=True, initial_prompt=whisper_prompt)
                    seg_text = "".join([s.text for s in segs]).strip()
                elif self.backend == "funasr" and self.sub_mode == "emotion":
                    res = self.model.generate(input=chunk, language=lang_arg, use_itn=True)
                    # if res: seg_text = res[0].get('text', '').strip()
                    # seg_text = self._clean_sensevoice_tags(raw_text)
                    if res and len(res) > 0:
                        raw_text = res[0].get('text', '').strip()
                        # 智能清洗标签
                        seg_text = self._clean_sensevoice_tags(raw_text)

                if not seg_text: continue
                clean_check = re.sub(r'[。？！，、\.\?\!,\s\n\r"\'”’]', '', seg_text)
                if len(clean_check) == 0:
                    logging.info(f"Dropped Garbage Segment: [{seg_text}]")
                    continue

                # 2. 【关键步骤】处理缓冲区 (接力棒逻辑)
                current_start_sec = start_ms / 1000.0
                current_end_sec = end_ms / 1000.0
                current_duration = current_end_sec - current_start_sec

                # 如果缓冲区有东西，说明上一句有尾巴遗留
                if buffer_text:
                    # 把尾巴拼接到当前句子的开头
                    seg_text = f"{buffer_text} {seg_text}"
                    # 【时光倒流】当前句子的开始时间，要向前延伸，去覆盖那个尾巴的时间
                    current_start_sec -= buffer_duration
                    # 清空缓冲
                    buffer_text = ""
                    buffer_duration = 0.0

                # 3. 语义分析与尾巴截断
                # 我们检查这句话的结尾是否有标点
                # 如果没有标点，且不是最后一段，我们认为它是“悬垂尾巴”，需要留给下一段
                # split_pattern = r'([\.?!。？！][”"’\']?)'

                # 3. 语义分析与尾巴截断
                # split_pattern = r'([\.?!。？！][”"’\']?)'
                split_pattern = r'([。？！，?!,]|(?<!\d)\.(?!\d))[”"’\']?'
                # 先按标点粗切
                sentences = re.split(split_pattern, seg_text)
                # 重组句子 (把标点拼回去)
                real_sentences = []
                curr = ""
                for s in sentences:
                    curr += s
                    if re.search(split_pattern, s):
                        real_sentences.append(curr.strip())
                        curr = ""
                if curr.strip(): real_sentences.append(curr.strip())

                # 检查最后一句是否完整
                if real_sentences:
                    last_sent = real_sentences[-1]
                    is_complete = re.search(r'[\.?!。？！]$', last_sent)
                    
                    # 如果最后一句没有标点，且不是全篇结束，且它不太长 (避免把超长句吞了)
                    if not is_complete and i < len(opt_segs) - 1 and len(last_sent) < 20:
                        # ✂️ 剪切动作：把它存入缓冲
                        buffer_text = last_sent
                        
                        # 计算这个尾巴的时间占比 (线性插值)
                        total_chars = len(seg_text)
                        tail_chars = len(buffer_text)
                        if total_chars > 0:
                            buffer_duration = (tail_chars / total_chars) * current_duration
                        
                        # 从当前列表移除最后一句
                        real_sentences.pop()
                        
                        # 修正当前段的结束时间 (把时间留给下一段)
                        current_end_sec -= buffer_duration

                # 4. 生成字幕块
                # 重新计算剩余句子的时间分配
                valid_text_len = sum(len(s) for s in real_sentences)
                valid_duration = current_end_sec - current_start_sec
                
                curr_srt_start = current_start_sec
                
                for line in real_sentences:
                    if not line.strip(): continue
                    
                    # 分配时间
                    if valid_text_len > 0:
                        line_dur = (len(line) / valid_text_len) * valid_duration
                    else:
                        line_dur = valid_duration
                    
                    curr_srt_end = curr_srt_start + line_dur
                    
                    # 写入
                    text_res += line + "\n"
                    srt_res += self._make_srt_block(srt_idx, curr_srt_start, curr_srt_end, line)
                    srt_idx += 1
                    curr_srt_start = curr_srt_end

            # 循环结束后，如果缓冲里还有东西 (全篇最后一句没标点)，把它吐出来
            if buffer_text:
                srt_res += self._make_srt_block(srt_idx, current_end_sec, current_end_sec + buffer_duration, buffer_text)

        # ... (Else FunASR SeACo 逻辑保持不变) ...


        # =========================================================================
        # 策略 B: FunASR SeACo (保持原样)
        # =========================================================================
        else:
            logging.info("Running FunASR Native Pipeline (SeACo)...")
            actual_enable_spk = enable_spk
            try:
                res = self.model.generate(
                    input=input_path, batch_size_s=300, sentence_timestamp=True, return_spk_res=actual_enable_spk
                )
            except Exception as e:
                logging.error(f"SeACo Error: {e}")
                if actual_enable_spk:
                    actual_enable_spk = False
                    res = self.model.generate(input=input_path, batch_size_s=300, sentence_timestamp=True, return_spk_res=False)
                else: raise e
            
            sentence_info = res[0].get('sentence_info', [])
            if sentence_info:
                srt_idx = 1
                for sent in sentence_info:
                    text = sent.get('text', '')
                    prefix = f"[Spk{sent['spk']}] " if (actual_enable_spk and 'spk' in sent) else ""
                    formatted_text = f"{prefix}{text}"
                    
                    start_sec = sent['timestamp'][0][0] / 1000.0 if 'timestamp' in sent else 0.0
                    end_sec = sent['timestamp'][-1][1] / 1000.0 if 'timestamp' in sent else 0.0
                    
                    text_res += formatted_text + "\n"
                    srt_res += self._make_srt_block(srt_idx, start_sec, end_sec, formatted_text)
                    srt_idx += 1
            else:
                text_res = str(res[0].get('text', ''))
                srt_res = self._make_fake_srt(text_res)

        # 保存
        with open(os.path.join(final_out_dir, f"{file_stem}.txt"), "w", encoding="utf-8") as f: f.write(text_res)
        with open(os.path.join(final_out_dir, f"{file_stem}.srt"), "w", encoding="utf-8") as f: f.write(srt_res)
        logging.info(f"Done. Saved to: {final_out_dir}")

    # ==============================================================
    # 【核心功能】 智能文本拆分
    # ==============================================================
    # def _split_text_smartly(self, text, max_chars=25):
    #     """
    #     按标点符号拆分长句子，避免字幕墙
    #     """
    #     if not text: return []
    #     # 按大标点切分
    #     chunks = re.split(r'([。？！\?\.!])', text)
    #     sentences = []
    #     current = ""
    #     for chunk in chunks:
    #         current += chunk
    #         if re.match(r'[。？！\?\.!]', chunk):
    #             sentences.append(current)
    #             current = ""
    #     if current: sentences.append(current)
        
    #     # 如果单句太长，按逗号再切
    #     final = []
    #     for sent in sentences:
    #         if len(sent) > max_chars:
    #             subs = re.split(r'([，,])', sent)
    #             curr_sub = ""
    #             for s in subs:
    #                 curr_sub += s
    #                 if re.match(r'[，,]', s):
    #                     final.append(curr_sub)
    #                     curr_sub = ""
    #             if curr_sub: final.append(curr_sub)
    #         else:
    #             final.append(sent)
    #     return [s.strip() for s in final if s.strip()]
    def _split_text_smartly(self, text, max_chars=25):
        """
        【层级切分算法 v2.0】
        Level 1: 强标点 (。？！) -> 保证句子独立性
        Level 2: 弱标点 (，、)   -> 解决单句过长 (> max_chars)
        """
        if not text: return []
        
        # --- Level 1: 强标点切分 ---
        # 匹配 . ? ! 。 ？！ 以及它们后面可能跟随的引号
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
        
        # --- Level 2: 弱标点切分 (按需激活) ---
        final_sentences = []
        
        for sent in strong_sentences:
            # 只有当句子长度超过阈值时，才动用逗号切分
            if len(sent) > max_chars:
                # 按逗号切分
                weak_pattern = r'([,，、])'
                subs = re.split(weak_pattern, sent)
                
                sub_curr = ""
                for sub in subs:
                    sub_curr += sub
                    # 如果遇到了逗号，且当前累积的长度已经像个短句了(比如>5个字)，就切一刀
                    # 避免把 "1, 2, 3" 这种切得太碎
                    if re.search(weak_pattern, sub) and len(sub_curr) > 5:
                        final_sentences.append(sub_curr.strip())
                        sub_curr = ""
                if sub_curr.strip(): final_sentences.append(sub_curr.strip())
            else:
                # 如果句子本来就不长，保留逗号，保持语气的连贯性
                final_sentences.append(sent)
                
        return [s for s in final_sentences if s]

    def _clean_sensevoice_tags(self, text):
        if not text: return ""
        emo_map = {
            "<|HAPPY|>": "(开心) ", "<|SAD|>": "(悲伤) ", "<|ANGRY|>": "(愤怒) ",
            "<|LAUGH|>": "(大笑) ", "<|NEUTRAL|>": ""
        }
        
        # for tag, label in emo_map.items():
        #     text = text.replace(tag, label)
        cleaned = re.sub(r"<\|.*?\|>", "", text)
        return cleaned.strip()

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
        # 🔴 修复：如果是空文本，直接返回空字符串（不生成块）
        if not text.strip():
            return ""
            
        # 🔴 修复：强制结束时间必须大于开始时间，至少给 10ms
        if end <= start:
            end = start + 0.01
            
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