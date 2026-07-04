import os
# 🔇 必须在导入 transformers 之前设置环境变量
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
# 提前设置 transformers 日志级别
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.generation").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

import argparse
import torch
import librosa
import numpy as np
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import warnings

# 强制刷新缓冲区，确保 WebUI 能实时拿到 print 内容
sys.stdout.reconfigure(encoding='utf-8')

# 🔇 抑制 HuggingFace transformers 的 attention_mask 警告
warnings.filterwarnings("ignore", message=".*attention_mask.*")
warnings.filterwarnings("ignore", message=".*pad_token_id.*")
warnings.filterwarnings("ignore", message=".*Setting.*eos_token_id.*")
warnings.filterwarnings("ignore", category=UserWarning)

# ================= 🚀 多进程支持 =================
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass
from queue import Empty
# ===============================================

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

# ================= 🔥 强制导入 FunASR 源码版本的 Nano 模型 =================
# 必须在 from funasr import AutoModel 之前导入，才能正确注册模型类
import sys
FUNASR_NANO_MODEL_DIR = r"E:\FunClip\FunASR\funasr\models\fun_asr_nano"
if FUNASR_NANO_MODEL_DIR not in sys.path:
    sys.path.insert(0, FUNASR_NANO_MODEL_DIR)
try:
    import importlib
    import model as nano_model_module
    importlib.reload(nano_model_module)  # 强制重新加载
    print("✅ 成功加载 FunASR 源码版本的 Nano model.py")
except Exception as e:
    print(f"⚠️ 加载 FunASR Nano model.py 失败: {e}")

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


# -------------------------------------------------------------------------
# 🔥 新增：并行 Worker 进程函数 (必须在顶级作用域)
# -------------------------------------------------------------------------
def worker_process_fn(worker_id, task_queue, result_queue, model_path, device, sub_mode, language):
    """
    Worker 进程：独立加载模型，循环处理任务
    """
    try:
        # 重定向输出，防止乱码
        sys.stdout.reconfigure(encoding='utf-8')
        
        # 1. 限制显存 (可选，防止 OOM)
        if torch.cuda.is_available():
            # 这里的 0.45 是经验值，两个模型共占 90%，留 10% 给系统
            # 如果显存非常大 (24G)，可以去掉这个限制
            total_mem = torch.cuda.get_device_properties(0).total_memory
            if total_mem < 16 * 1024**3: # < 16GB 才限制
                 torch.cuda.set_per_process_memory_fraction(0.45, 0)
        
        print(f"   [Worker {worker_id}] ⏳ 正在加载模型...", flush=True)
        
        # 2. 加载模型 (每个进程一份)
        # 注意：这里我们只支持 Nano 和 SenseVoice
        if sub_mode == "emotion":
             model = AutoModel(model=MODEL_PATHS["sensevoice"], trust_remote_code=True, device=device, disable_update=True, disable_pbar=True)
        else:
             # Nano
             model = AutoModel(model=MODEL_PATHS["nano"], trust_remote_code=True, device=device, disable_update=True, disable_pbar=True)
             
        print(f"   [Worker {worker_id}] ✅ 模型加载完毕！等待任务...", flush=True)
        
        # 3. 循环处理
        while True:
            try:
                # timeout 避免死锁
                task = task_queue.get(timeout=2) 
            except Empty:
                continue
                
            if task is None: # 退出信号
                break
            
            task_idx, chunk_tensor, time_range = task
            
            try:
                # 推理
                # print(f"   [Worker {worker_id}] 处理片段 {task_idx}...", flush=True)
                
                t0 = time.time()
                # Nano/SenseVoice 通用推理接口
                res = model.generate(
                    input=[chunk_tensor], 
                    batch_size_s=0, 
                    disable_pbar=True,
                    language=language, # 传递语言参数
                    use_itn=True
                )
                t1 = time.time()
                infer_dur = t1 - t0
                
                # 提取文本
                raw_text = ""
                if res and len(res) > 0:
                    raw_text = res[0].get('text', '').strip()
                    
                result_queue.put((task_idx, raw_text, time_range, infer_dur))
                
            except Exception as e:
                print(f"   [Worker {worker_id}] ❌ 推理出错: {e}", flush=True)
                result_queue.put((task_idx, "", time_range, 0.0))
                
    except Exception as e:
        print(f"   [Worker {worker_id}] ❌ 进程崩溃: {e}", flush=True)
        import traceback
        traceback.print_exc()

# -------------------------------------------------------------------------
# 🔥 新增：并行编排引擎
# -------------------------------------------------------------------------
class ParallelASREngine:
    def __init__(self, backend="funasr", model_size="turbo", sub_mode="nano", device="cuda", folder_mode=False, num_workers=2):
        self.backend = backend
        self.sub_mode = sub_mode
        self.device = device
        self.folder_mode = folder_mode
        self.num_workers = num_workers
        
        log_ui(f"🚀 初始化并行引擎 (Worker数: {num_workers})")
        log_ui(f"   后端: {backend} | 模式: {sub_mode}")
        
        # 1. 加载 VAD (主进程只负责切分)
        log_ui("   [Init] 加载 VAD 模型 (主进程)...")
        self.vad_model = AutoModel(model=MODEL_PATHS["vad"], trust_remote_code=True, device=device, disable_update=True, disable_pbar=True)
        
    def run(self, input_path, output_dir, language="auto", enable_spk=False, batch_size=8):
        # 1. 准备输出路径
        raw_stem = os.path.splitext(os.path.basename(input_path))[0]
        file_stem = raw_stem[:50] + "_cut" if len(raw_stem) > 50 else raw_stem
        final_out_dir = output_dir if self.folder_mode else os.path.join(output_dir, file_stem)
        os.makedirs(final_out_dir, exist_ok=True)
        
        log_ui(f"🎬 [并行] 开始处理: {os.path.basename(input_path)}")
        
        # 2. VAD 切分
        log_ui("⏳ Step 1: VAD 切分...")
        try:
            audio, _ = librosa.load(input_path, sr=16000)
             # 针对有声书/BGM优化的激进参数
            vad_kwargs = {
                "max_single_segment_time": 12000, 
                "speech_noise_thres": 0.9,       
                "max_end_silence_time": 200,      
                "speech_to_sil_time_thres": 200,
            }
            vad_out = self.vad_model.generate(input=input_path, batch_size_s=5000, **vad_kwargs)
            raw_segs = vad_out[0]['value'] if vad_out and len(vad_out)>0 and 'value' in vad_out[0] else [[0, len(audio)/16*1000]]
            opt_segs = self._merge_vad_segments(raw_segs, max_gap_ms=300, max_duration_ms=12000)
            log_ui(f"✅ VAD 完成: {len(opt_segs)} 段")
        except Exception as e:
            log_ui(f"❌ VAD 失败: {e}")
            return

        # 3. 准备多进程环境
        ctx = mp.get_context('spawn')
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()
        workers = []
        
        # 4. 启动 Workers
        log_ui(f"🔥 启动 {self.num_workers} 个推理进程...")
        for i in range(self.num_workers):
            p = ctx.Process(
                target=worker_process_fn,
                args=(i, task_queue, result_queue, MODEL_PATHS["nano"], self.device, self.sub_mode, language)
            )
            p.start()
            workers.append(p)
            
        # 5. 分发任务
        valid_tasks = 0
        log_ui("📤 分发任务中...")
        for i, (start_ms, end_ms) in enumerate(opt_segs):
             if (end_ms - start_ms) < 400: continue
             
             s_idx = int(start_ms * 16)
             e_idx = int(end_ms * 16)
             chunk = audio[max(0, s_idx-800):min(len(audio), e_idx+800)]
             
             # 转 Tensor 并 share_memory (对于 spawn 模式其实是 pickling)
             chunk_tensor = torch.from_numpy(chunk).share_memory_()
             
             task_queue.put((valid_tasks, chunk_tensor, (start_ms, end_ms)))
             valid_tasks += 1
             
        # 6. 收集结果
        log_ui(f"📥 等待结果 (共 {valid_tasks} 个片段)...")
        results_map = {}
        total_audio_dur = 0.0
        total_infer_dur = 0.0
        
        t_start_process = time.time()
        
        for _ in range(valid_tasks):
            idx, text, time_range, i_dur = result_queue.get()
            results_map[idx] = (text, time_range)
            
            # 统计
            seg_dur = (time_range[1] - time_range[0]) / 1000.0
            total_audio_dur += seg_dur
            total_infer_dur += i_dur
            rtf = seg_dur / i_dur if i_dur > 0.001 else 0.0
            
            if _ % 5 == 0: 
                 # 实时显示每段速度（这就很爽了）
                 print(f"\r   [进度 {_}/{valid_tasks}] 片段#{idx} ({seg_dur:.1f}s) -> 耗时 {i_dur:.2f}s (Speed: {rtf:.1f}x)", end="", flush=True)
                 
        print("", flush=True)
        t_total_process = time.time() - t_start_process
        
        # 打印详细性能报告
        print("\n" + "="*40, flush=True)
        print(f"📊 性能统计报告", flush=True)
        print(f"----------------------------------------", flush=True)
        print(f"音频总时长: {total_audio_dur/60:.1f} 分钟", flush=True)
        print(f"处理总耗时: {t_total_process:.1f} 秒", flush=True)
        print(f"累积推理值: {total_infer_dur:.1f} 秒 (单卡等效)", flush=True)
        
        # 真正的并行效率 = 累积推理耗时 / 实际流逝时间
        parallel_eff = total_infer_dur / t_total_process if t_total_process > 0 else 0
        overall_speed = total_audio_dur / t_total_process if t_total_process > 0 else 0
        
        print(f"并行效率值: {parallel_eff:.2f}x (理论上限 {self.num_workers}.0x)", flush=True)
        print(f"整体加速比: {overall_speed:.1f}x (RTF)", flush=True)
        print(f"========================================", flush=True)
        
        # 7. 清理进程
        log_ui("🧹 清理进程...")
        for _ in range(self.num_workers):
            task_queue.put(None)
        for w in workers:
            w.join()
            
        # 8. 组装结果
        log_ui("📝 组装 SRT...")
        text_res = ""
        srt_res = ""
        srt_idx = 1
        
        # 按顺序重组
        for i in range(valid_tasks):
            if i not in results_map: continue
            raw_text, (s_ms, e_ms) = results_map[i]
            
            # 这里的后处理逻辑复用之前的 _split_text_smartly
            # 为简单起见，这里再复制一遍核心逻辑，或者从 Self 中调用静态方法
            # 由于 ParallelASREngine 和 SuperASREngine 独立，我们把工具函数抽离出来最好
            # 这里先简单实现
            if not raw_text: continue
            
            # 感知标签清理
            if self.sub_mode == "emotion":
                raw_text = re.sub(r"<\|.*?\|>", "", raw_text).strip()
            
            # 智能切分
            split_lines = self._split_text_smartly(raw_text)
            
            valid_chars = sum(len(s) for s in split_lines)
            total_dur_sec = (e_ms - s_ms) / 1000.0
            curr_srt_start = s_ms / 1000.0
            
            for line in split_lines:
                if not line.strip(): continue
                if valid_chars > 0:
                    line_dur = (len(line) / valid_chars) * total_dur_sec
                else:
                    line_dur = total_dur_sec
                    
                curr_srt_end = curr_srt_start + line_dur
                text_res += line + "\n"
                srt_res += self._make_srt_block(srt_idx, curr_srt_start, curr_srt_end, line)
                srt_idx += 1
                curr_srt_start = curr_srt_end
                
        # 9. 保存
        txt_path = os.path.join(final_out_dir, f"{file_stem}.txt")
        srt_path = os.path.join(final_out_dir, f"{file_stem}.srt")
        with open(txt_path, "w", encoding="utf-8") as f: f.write(text_res)
        with open(srt_path, "w", encoding="utf-8") as f: f.write(srt_res)
        log_ui(f"🎉 并行处理完成！\nSRT: {srt_path}")

    # 复用辅助函数 (从 SuperASREngine 复制，或者独立出来)
    def _split_text_smartly(self, text, max_chars=25):
        # ... (完全复用原逻辑) ...
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

    def _merge_vad_segments(self, segments, max_gap_ms=300, max_duration_ms=12000):
        # 复用 VAD 逻辑
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


class SuperASREngine:
    # def __init__(self, backend="faster", model_size="turbo", sub_mode="precision", device="cuda"):
    #     self.backend = backend
    #     self.device = device
    #     self.sub_mode = sub_mode 
    #     self.model = None
    #     self.vad_model = None 
    #     self.spk_engine = None 
    def __init__(self, backend="faster", model_size="turbo", sub_mode="precision", device="cuda", folder_mode=False):
        self.backend = backend
        self.device = device
        self.sub_mode = sub_mode
        self.folder_mode = folder_mode
        self.model = None
        self.vad_model = None   # 🔥 修复：必须初始化
        self.spk_engine = None  # 🔥 修复：必须初始化
        
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
                self.model = AutoModel(model=MODEL_PATHS["nano"], trust_remote_code=True, device=device, disable_update=True)
            else:
                self.model = AutoModel(model=MODEL_PATHS["seaco"], vad_model=MODEL_PATHS["vad"], punc_model=MODEL_PATHS["punc"], spk_model=MODEL_PATHS["spk"], device=device, disable_update=True)
    # def run(self, input_path, output_dir, language="auto", enable_spk=False, batch_size=8):
    #         file_stem = os.path.splitext(os.path.basename(input_path))[0]
    #         final_out_dir = os.path.join(output_dir, file_stem)
    #         os.makedirs(final_out_dir, exist_ok=True)
    def run(self, input_path, output_dir, language="auto", enable_spk=False, batch_size=8):
        # 1. 获取基础文件名 (增加一个截断逻辑防止文件名过长报错)
        raw_stem = os.path.splitext(os.path.basename(input_path))[0]
        if len(raw_stem) > 50:
            file_stem = raw_stem[:50] + "_cut"
        else:
            file_stem = raw_stem
            
        # 2. 核心逻辑：判断是否使用扁平化文件夹模式
        if self.folder_mode:
            # ✅ 文件夹模式：直接存入 output_dir，不创建子文件夹
            final_out_dir = output_dir 
        else:
            # ⏹️ 默认模式：为每个视频创建一个同名子文件夹
            final_out_dir = os.path.join(output_dir, file_stem)
            
        os.makedirs(final_out_dir, exist_ok=True)
        
        log_ui(f"🎬 开始处理: {os.path.basename(input_path)}")
        
        text_res = ""
        srt_res = ""
        lang_arg = None if language == "auto" else language

        # =========================================================
        # 🛤️ 轨道一：手动流水线 (Nano / SenseVoice / Whisper)
        # =========================================================
        if self.backend in ["faster", "openai"] or (self.backend == "funasr" and self.sub_mode in ["emotion", "nano"]):
            
            # 🔥 说话人识别：懒加载 SPK 模型
            if enable_spk and self.spk_engine is None:
                # ⚠️ Nano + SPK 性能警告
                if self.sub_mode == "nano":
                    log_ui("⚠️ 提示: Nano + SPK 组合较慢 (Nano 基于 LLM)，建议使用 SenseVoice + SPK 获得更快速度")
                
                log_ui("   [Init] 加载说话人模型 (Cam++)...")
                try:
                    self.spk_engine = SpeakerDiarizer(MODEL_PATHS["spk"], device=self.device)
                    log_ui("   ✅ SPK 模型加载成功")
                except Exception as e:
                    log_ui(f"⚠️ SPK 模型加载失败: {e}，已跳过说话人识别")
                    enable_spk = False

            # --- 1. VAD 切分 (应用抗 BGM 激进参数) ---
            log_ui("⏳ Step 1: VAD 切分 (抗 BGM 激进模式)...")
            try:
                audio, _ = librosa.load(input_path, sr=16000)
                
                # 针对有声书/BGM优化的激进参数
                vad_kwargs = {
                    "max_single_segment_time": 12000, # [强制] 12秒切一刀 (缩短以减少语言漂移)
                    "speech_noise_thres": 0.9,        # [抗噪] 过滤BGM
                    "max_end_silence_time": 200,      # [切断] 快速切断
                    "speech_to_sil_time_thres": 200,
                }

                vad_out = self.vad_model.generate(
                    input=input_path, 
                    batch_size_s=5000, 
                    **vad_kwargs
                )
                
                raw_segs = vad_out[0]['value'] if vad_out and len(vad_out)>0 and 'value' in vad_out[0] else [[0, len(audio)/16*1000]]
                opt_segs = self._merge_vad_segments(raw_segs, max_gap_ms=300, max_duration_ms=12000)
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
            whisper_prompt = prompt_map.get(language if language != "auto" else "auto", prompt_map["auto"])

            # --- 2. 并发推理 (FunASR) vs 串行推理 (Whisper) ---
            is_funasr_batch = (self.backend == "funasr" and self.sub_mode in ["nano", "emotion"])
            
            # [分支 A] FunASR 并发模式
            if is_funasr_batch:
                log_ui(f"🚀 激活 FunASR 并发加速模式 (Batch Size: {batch_size})...")
                total_segments = len(opt_segs)
                processed_count = 0
                global_start_time = time.time()
                
                # 统计变量
                total_infer_dur_s = 0.0
                total_audio_dur_s = 0.0
                
                # 内部函数：处理 Batch
                def process_batch(chunk_tensors, time_ranges):
                    nonlocal srt_idx, text_res, srt_res, processed_count, total_infer_dur_s, total_audio_dur_s
                    if not chunk_tensors: return

                    t_start = time.time()
                    try:
                        # 开启 ITN 找回标点
                        res_batch = self.model.generate(
                            input=chunk_tensors,
                            batch_size_s=0,
                            disable_pbar=True,
                            language=language,
                            use_itn=True 
                        )
                        
                        processed_count += len(chunk_tensors)
                        batch_cost = time.time() - t_start
                        
                        # 统计
                        batch_audio_dur = sum((e-s) for s,e in time_ranges) / 1000.0
                        total_infer_dur_s += batch_cost
                        total_audio_dur_s += batch_audio_dur
                        rtf = batch_audio_dur / batch_cost if batch_cost > 0.001 else 0.0
                        
                        progress_pct = (processed_count / total_segments) * 100
                        log_ui(f"   ⚡ [{progress_pct:.1f}%] 已处理 {processed_count}/{total_segments} | 耗时: {batch_cost:.2f}s | Speed: {rtf:.1f}x")

                        # 🔍 调试：打印返回结果类型
                        if res_batch and len(res_batch) > 0:
                            log_ui(f"   📝 返回 {len(res_batch)} 个结果")
                        else:
                            log_ui(f"   ⚠️ 模型返回空结果！")
                            return

                        for idx, res in enumerate(res_batch):
                            s_ms, e_ms = time_ranges[idx]
                            raw_text = res.get('text', '').strip()
                            
                            # 🔍 调试：打印每个结果
                            if idx < 3:  # 只打印前3个
                                log_ui(f"      片段{idx}: '{raw_text[:50]}...' " if len(raw_text) > 50 else f"      片段{idx}: '{raw_text}'")
                            
                            if self.sub_mode == "emotion":
                                raw_text = self._clean_sensevoice_tags(raw_text)
                            if not raw_text: continue
                            
                            # 🔥 说话人识别：在片段级别调用（而不是每行）
                            spk_prefix = ""
                            if enable_spk and self.spk_engine is not None:
                                try:
                                    chunk_np = chunk_tensors[idx].numpy() if hasattr(chunk_tensors[idx], 'numpy') else chunk_tensors[idx]
                                    spk_id = self.spk_engine.get_speaker_id(chunk_np)
                                    spk_prefix = f"[SPK:{spk_id}] "
                                except Exception as spk_e:
                                    log_ui(f"   ⚠️ SPK 获取失败: {spk_e}")
                            
                            # 暴力切分长难句
                            split_lines = self._split_text_smartly(raw_text, max_chars=30)
                            valid_chars = sum(len(s) for s in split_lines)
                            total_dur_sec = (e_ms - s_ms) / 1000.0
                            curr_srt_start = s_ms / 1000.0

                            for line in split_lines:
                                if not line.strip(): continue
                                if valid_chars > 0:
                                    line_dur = (len(line) / valid_chars) * total_dur_sec
                                else:
                                    line_dur = total_dur_sec
                                
                                curr_srt_end = curr_srt_start + line_dur
                                full_line = spk_prefix + line 
                                text_res += full_line + "\n"
                                srt_res += self._make_srt_block(srt_idx, curr_srt_start, curr_srt_end, full_line)
                                srt_idx += 1
                                curr_srt_start = curr_srt_end

                    except Exception as e:
                        log_ui(f"❌ Batch Error: {e}")
                        import traceback
                        traceback.print_exc()

                # =========================================================
                # 🚀 优化 A+B: 多线程预加载 + CUDA 双缓冲流水线
                # =========================================================
                
                # 内部函数：单个音频片段提取 (用于线程池)
                def extract_chunk_worker(seg_info):
                    start_ms, end_ms, seg_idx = seg_info
                    if (end_ms - start_ms) < 400:  # 过滤更短的噪音片段
                        return None, None, seg_idx
                    s_idx = int(start_ms * 16)
                    e_idx = int(end_ms * 16)
                    chunk = audio[max(0, s_idx-800):min(len(audio), e_idx+800)]
                    # 使用 pinned memory 加速 CPU->GPU 传输
                    tensor = torch.from_numpy(chunk.copy())
                    if torch.cuda.is_available():
                        tensor = tensor.pin_memory()
                    return tensor, (start_ms, end_ms), seg_idx
                
                # 准备分批次的 segment 列表
                all_seg_infos = [(s, e, i) for i, (s, e) in enumerate(opt_segs)]
                
                log_ui(f"   ⏳ 启动 4 线程并行预加载...")
                
                # 使用线程池并行提取所有音频片段
                all_tensors = []
                all_times = []
                with ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(executor.map(extract_chunk_worker, all_seg_infos))
                
                # 过滤有效结果并保持顺序
                for tensor, time_range, _ in sorted(results, key=lambda x: x[2]):
                    if tensor is not None:
                        all_tensors.append(tensor)
                        all_times.append(time_range)
                
                log_ui(f"   ✅ 预加载完成: {len(all_tensors)} 个有效片段")
                
                # =========================================================
                # 🔥 SPK 先行策略：先跑完所有说话人识别，再跑 ASR
                # =========================================================
                spk_cache = {}  # key: segment_index, value: speaker_id
                
                if enable_spk and self.spk_engine is not None:
                    log_ui(f"   🎭 Step 2: 并行预处理说话人识别 ({len(all_tensors)} 个片段)...")
                    spk_start_time = time.time()
                    
                    def get_spk_for_segment(args):
                        """对单个片段进行说话人识别"""
                        seg_idx, tensor = args
                        try:
                            chunk_np = tensor.numpy() if hasattr(tensor, 'numpy') else tensor
                            spk_id = self.spk_engine.get_speaker_id(chunk_np)
                            return seg_idx, spk_id
                        except:
                            return seg_idx, "?"
                    
                    # 使用线程池并行处理 SPK（SPK 模型较快，可以并行）
                    spk_tasks = [(i, t) for i, t in enumerate(all_tensors)]
                    with ThreadPoolExecutor(max_workers=2) as spk_executor:
                        spk_results = list(spk_executor.map(get_spk_for_segment, spk_tasks))
                    
                    # 缓存结果
                    for seg_idx, spk_id in spk_results:
                        spk_cache[seg_idx] = spk_id
                    
                    log_ui(f"   ✅ SPK 预处理完成！耗时: {time.time() - spk_start_time:.2f}s")
                
                # 现在 process_batch 可以直接查询 spk_cache
                # 修改 process_batch 内部的 SPK 获取逻辑
                def process_batch_with_spk_cache(chunk_tensors, time_ranges, batch_offset):
                    """带 SPK 缓存的批量处理"""
                    nonlocal srt_idx, text_res, srt_res, processed_count, total_infer_dur_s, total_audio_dur_s
                    if not chunk_tensors: return

                    t_start = time.time()
                    try:
                        res_batch = self.model.generate(
                            input=chunk_tensors,
                            batch_size_s=0,
                            disable_pbar=True,
                            language=language,
                            use_itn=True 
                        )
                        
                        processed_count += len(chunk_tensors)
                        batch_cost = time.time() - t_start
                        
                        batch_audio_dur = sum((e-s) for s,e in time_ranges) / 1000.0
                        total_infer_dur_s += batch_cost
                        total_audio_dur_s += batch_audio_dur
                        rtf = batch_audio_dur / batch_cost if batch_cost > 0.001 else 0.0
                        
                        progress_pct = (processed_count / total_segments) * 100
                        log_ui(f"   ⚡ [{progress_pct:.1f}%] 已处理 {processed_count}/{total_segments} | 耗时: {batch_cost:.2f}s | Speed: {rtf:.1f}x")

                        if res_batch and len(res_batch) > 0:
                            log_ui(f"   📝 返回 {len(res_batch)} 个结果")
                        else:
                            log_ui(f"   ⚠️ 模型返回空结果！")
                            return

                        for idx, res in enumerate(res_batch):
                            global_seg_idx = batch_offset + idx  # 全局片段索引
                            s_ms, e_ms = time_ranges[idx]
                            raw_text = res.get('text', '').strip()
                            
                            if idx < 3:
                                log_ui(f"      片段{idx}: '{raw_text[:50]}...' " if len(raw_text) > 50 else f"      片段{idx}: '{raw_text}'")
                            
                            if self.sub_mode == "emotion":
                                raw_text = self._clean_sensevoice_tags(raw_text)
                            if not raw_text: continue
                            
                            # 🔥 从缓存获取 SPK（不再调用模型！）
                            spk_prefix = ""
                            if enable_spk and global_seg_idx in spk_cache:
                                spk_prefix = f"[SPK:{spk_cache[global_seg_idx]}] "
                            
                            split_lines = self._split_text_smartly(raw_text, max_chars=30)
                            valid_chars = sum(len(s) for s in split_lines)
                            total_dur_sec = (e_ms - s_ms) / 1000.0
                            curr_srt_start = s_ms / 1000.0

                            for line in split_lines:
                                if not line.strip(): continue
                                if valid_chars > 0:
                                    line_dur = (len(line) / valid_chars) * total_dur_sec
                                else:
                                    line_dur = total_dur_sec
                                
                                curr_srt_end = curr_srt_start + line_dur
                                full_line = spk_prefix + line 
                                text_res += full_line + "\n"
                                srt_res += self._make_srt_block(srt_idx, curr_srt_start, curr_srt_end, full_line)
                                srt_idx += 1
                                curr_srt_start = curr_srt_end

                    except Exception as e:
                        log_ui(f"❌ Batch Error: {e}")
                        import traceback
                        traceback.print_exc()
                
                # 分批处理 (双缓冲: 当前批推理时，下一批已在 GPU)
                cuda_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
                
                log_ui(f"   🚀 Step 3: 开始 Nano ASR 批量识别...")
                for batch_start in range(0, len(all_tensors), batch_size):
                    batch_end = min(batch_start + batch_size, len(all_tensors))
                    current_tensors = all_tensors[batch_start:batch_end]
                    current_times = all_times[batch_start:batch_end]
                    
                    # 异步预加载下一批到 GPU (如果有)
                    if cuda_stream and batch_end < len(all_tensors):
                        next_batch_end = min(batch_end + batch_size, len(all_tensors))
                        with torch.cuda.stream(cuda_stream):
                            for t in all_tensors[batch_end:next_batch_end]:
                                if t.is_pinned():
                                    t.cuda(non_blocking=True)
                    
                    # 处理当前批（传入 batch_start 作为偏移量）
                    process_batch_with_spk_cache(current_tensors, current_times, batch_start)
                
                log_ui(f"🏁 FunASR 处理完毕！总耗时: {time.time() - global_start_time:.2f}s")

            # [分支 B] Whisper 串行模式
            else:
                log_ui("🐢 运行在串行模式 (Whisper)...")
                buffer_text = ""
                buffer_duration = 0.0
                
                for i, (start_ms, end_ms) in enumerate(opt_segs):
                    if i % 10 == 0: log_ui(f"   👉 进度: {i}/{len(opt_segs)}")
                    s_idx = int(start_ms * 16); e_idx = int(end_ms * 16)
                    chunk = audio[max(0, s_idx-800):min(len(audio), e_idx+800)]
                    if len(chunk) < 1600: continue
                    
                    seg_text = ""
                    try:
                        if self.backend == "faster":
                            segs, _ = self.model.transcribe(chunk, beam_size=5, language=lang_arg, initial_prompt=whisper_prompt)
                            seg_text = "".join([s.text for s in segs]).strip()
                    except Exception as e:
                        log_ui(f"⚠️ 识别出错: {e}")
                        continue

                    if not seg_text: continue
                    
                    # 简单的字幕拼接逻辑
                    current_start = start_ms / 1000.0
                    current_end = end_ms / 1000.0
                    text_res += seg_text + "\n"
                    srt_res += self._make_srt_block(srt_idx, current_start, current_end, seg_text)
                    srt_idx += 1

        # =========================================================
        # 🛤️ 轨道二：SeACo (Legacy)
        # =========================================================
        else:
            log_ui("🔄 Running SeACo Pipeline (Old Model)...")
            try:
                # SeACo 自带 VAD 和推理，不支持我们的 VAD 优化
                res = self.model.generate(
                    input=input_path, 
                    batch_size_s=300, 
                    sentence_timestamp=True, 
                    return_spk_res=enable_spk
                )
                sentence_info = res[0].get('sentence_info', [])
                if sentence_info:
                    srt_idx = 1
                    for sent in sentence_info:
                        text = sent.get('text', '')
                        start = sent['timestamp'][0][0] / 1000.0
                        end = sent['timestamp'][-1][1] / 1000.0
                        text_res += text + "\n"
                        srt_res += self._make_srt_block(srt_idx, start, end, text)
                        srt_idx += 1
            except Exception as e:
                log_ui(f"❌ SeACo Error: {e}")

        # === 强制保存 ===
        txt_path = os.path.join(final_out_dir, f"{file_stem}.txt")
        srt_path = os.path.join(final_out_dir, f"{file_stem}.srt")
        
        log_ui("💾 正在写入文件...")
        with open(txt_path, "w", encoding="utf-8") as f: f.write(text_res)
        with open(srt_path, "w", encoding="utf-8") as f: f.write(srt_res)
        log_ui(f"🎉 全部完成！\nSRT: {srt_path}")
#     def run(self, input_path, output_dir, language="auto", enable_spk=False, batch_size=8):
#         file_stem = os.path.splitext(os.path.basename(input_path))[0]
#         final_out_dir = os.path.join(output_dir, file_stem)
#         os.makedirs(final_out_dir, exist_ok=True)
        
#         log_ui(f"🎬 开始处理: {os.path.basename(input_path)}")
        
#         text_res = ""
#         srt_res = ""
#         lang_arg = None if language == "auto" else language

#         # === 轨道一：手动流水线 (Nano/SenseVoice/Whisper) ===
#         if self.backend in ["faster", "openai"] or (self.backend == "funasr" and self.sub_mode in ["emotion", "nano"]):
            
#             if enable_spk and self.spk_engine is None:
#                 self.spk_engine = SpeakerDiarizer(MODEL_PATHS["spk"], device=self.device)

#             log_ui("⏳ Step 1: VAD 切分...")
#             try:
#                 audio, _ = librosa.load(input_path, sr=16000)
#                 vad_out = self.vad_model.generate(input=input_path, batch_size_s=5000, max_single_segment_time=60000)
#                 raw_segs = vad_out[0]['value'] if vad_out and len(vad_out)>0 and 'value' in vad_out[0] else [[0, len(audio)/16*1000]]
#                 opt_segs = self._merge_vad_segments(raw_segs, max_gap_ms=300, max_duration_ms=8000)
                
#                 log_ui(f"✅ 切分完成: {len(opt_segs)} 段")
#             except Exception as e:
#                 log_ui(f"❌ VAD 失败: {e}")
#                 return
#             # ========================================================

#             srt_idx = 1
#             prompt_map = {
#                 "zh": "以下是普通话的句子，请使用简体中文，并添加标点符号。",
#                 "en": "The following are sentences in English. Please use English and add punctuation.",
#                 "ja": "以下は日本語の文章です。日本語を使用し、句読点を付けてください。",
#                 "auto": "Please add punctuation."
#             }
#         whisper_prompt = prompt_map.get(language if language != "auto" else "auto", prompt_map["auto"])

#         # =========================================================
#         # 🔥🔥🔥 核心分支：区分 FunASR (Batch) 和 Whisper (Serial)
#         # =========================================================
        
#         is_funasr_batch = (self.backend == "funasr" and self.sub_mode in ["nano", "emotion"])
        
#         if is_funasr_batch:
#             log_ui(f"🚀 激活 FunASR 并发加速模式 (Batch Size: {batch_size})...")
            
#             # --- 统计总工作量 ---
#             total_segments = len(opt_segs)
#             processed_count = 0
#             global_start_time = time.time() # 开始计时
            
#             # --- 内部函数：处理一个 Batch ---
#             def process_batch(chunk_tensors, time_ranges):
#                 nonlocal srt_idx, text_res, srt_res, processed_count
#                 if not chunk_tensors: return

#                 batch_len = len(chunk_tensors)
#                 t_start = time.time() # 批次开始时间

#                 try:
#                     # 🔥 批量推理核心
#                     res_batch = self.model.generate(
#                         input=chunk_tensors,
#                         batch_size_s=0,
#                         disable_pbar=True,
#                         language=language 
#                     )
                    
#                     t_end = time.time() # 批次结束时间
#                     batch_cost = t_end - t_start
#                     processed_count += batch_len
                    
#                     # 📊【新增】计算并打印速度信息
#                     # RTF (Real Time Factor) = 处理耗时 / 音频时长 (越小越快)
#                     # 这里粗略计算：假设每个片段平均 5 秒 (仅作参考)
#                     avg_audio_dur = batch_len * 5.0 
#                     rtf = batch_cost / avg_audio_dur if avg_audio_dur > 0 else 0
                    
#                     progress_pct = (processed_count / total_segments) * 100
#                     log_ui(f"   ⚡ [{progress_pct:.1f}%] 已处理 {processed_count}/{total_segments} | 本批耗时: {batch_cost:.2f}s | 速度: {batch_len/batch_cost:.1f}段/秒")

#                     # 遍历结果 (保持原有的字幕切分逻辑)
#                     for idx, res in enumerate(res_batch):
#                         s_ms, e_ms = time_ranges[idx]
#                         raw_text = res.get('text', '').strip()
                        
#                         if self.sub_mode == "emotion":
#                             raw_text = self._clean_sensevoice_tags(raw_text)
                            
#                         if not raw_text: continue
                        
#                         # ✂️ 智能切分逻辑
#                         split_lines = self._split_text_smartly(raw_text, max_chars=30)
#                         valid_chars = sum(len(s) for s in split_lines)
#                         total_dur_sec = (e_ms - s_ms) / 1000.0
#                         curr_srt_start = s_ms / 1000.0

#                         for line in split_lines:
#                             if not line.strip(): continue
#                             if valid_chars > 0:
#                                 line_dur = (len(line) / valid_chars) * total_dur_sec
#                             else:
#                                 line_dur = total_dur_sec
                            
#                             curr_srt_end = curr_srt_start + line_dur
#                             full_line = line 
#                             text_res += full_line + "\n"
#                             srt_res += self._make_srt_block(srt_idx, curr_srt_start, curr_srt_end, full_line)
#                             srt_idx += 1
#                             curr_srt_start = curr_srt_end

#                 except Exception as e:
#                     log_ui(f"❌ Batch Error: {e}")

#             # --- 循环填充 Buffer ---
#             tensor_buf = []
#             time_buf = []
            
#             log_ui(f"   ⏳ 正在装填数据并预热 GPU...")
            
#             for i, (start_ms, end_ms) in enumerate(opt_segs):
#                 # 过滤极短片段
#                 if (end_ms - start_ms) < 500: continue
                
#                 s_idx = int(start_ms * 16)
#                 e_idx = int(end_ms * 16)
#                 chunk = audio[max(0, s_idx-800):min(len(audio), e_idx+800)]
                
#                 chunk_tensor = torch.from_numpy(chunk)
#                 tensor_buf.append(chunk_tensor)
#                 time_buf.append((start_ms, end_ms))
                
#                 # 攒够了就发车
#                 if len(tensor_buf) >= batch_size:
#                     process_batch(tensor_buf, time_buf)
#                     tensor_buf = []
#                     time_buf = []
            
#             # 处理剩下的尾巴
#             if tensor_buf:
#                 process_batch(tensor_buf, time_buf)
                
#             # 📊【新增】打印最终总耗时
#             global_end_time = time.time()
#             total_cost = global_end_time - global_start_time
#             log_ui(f"🏁 FunASR 并发处理完毕！总耗时: {total_cost:.2f}秒 (平均 {(total_segments/total_cost):.1f} 段/秒)")
#             # --- 内部函数：处理一个 Batch ---
#             # def process_batch(chunk_tensors, time_ranges):
#             #     nonlocal srt_idx, text_res, srt_res
#             #     if not chunk_tensors: return

#             #     try:
#             #         # 🔥 批量推理核心：一次喂给模型多个 Tensor
#             #         res_batch = self.model.generate(
#             #             input=chunk_tensors,
#             #             batch_size_s=0,
#             #             disable_pbar=True,
#             #             language=language # 确保传入语言
#             #         )
                    
#             #         # 遍历结果
#             #         for idx, res in enumerate(res_batch):
#             #             s_ms, e_ms = time_ranges[idx]
#             #             raw_text = res.get('text', '').strip()
                        
#             #             # 简单的清理
#             #             if self.sub_mode == "emotion":
#             #                 raw_text = self._clean_sensevoice_tags(raw_text)
                            
#             #             if not raw_text: continue
                        
#             #             # 格式化
#             #             full_line = raw_text
#             #             s_sec = s_ms / 1000.0
#             #             e_sec = e_ms / 1000.0
                        
#             #             # SPK 逻辑 (Batch 模式下为了速度暂略，或需单独做 Batch)
#             #             # 如果需要 SPK，建议此处单独调用 self.spk_engine.get_speaker_id(原始numpy数据)
                        
#             #             text_res += full_line + "\n"
#             #             srt_res += self._make_srt_block(srt_idx, s_sec, e_sec, full_line)
#             #             srt_idx += 1
#             #     except Exception as e:
#             #         log_ui(f"❌ Batch Error: {e}")



# # =======================================================================================================            
#             # def process_batch(chunk_tensors, time_ranges):
#             #     nonlocal srt_idx, text_res, srt_res
#             #     if not chunk_tensors: return

#             #     try:
#             #         # 🔥 批量推理
#             #         res_batch = self.model.generate(
#             #             input=chunk_tensors,
#             #             batch_size_s=0,
#             #             disable_pbar=True,
#             #             language=language 
#             #         )
                    
#             #         # 遍历结果
#             #         for idx, res in enumerate(res_batch):
#             #             s_ms, e_ms = time_ranges[idx]
#             #             raw_text = res.get('text', '').strip()
                        
#             #             if self.sub_mode == "emotion":
#             #                 raw_text = self._clean_sensevoice_tags(raw_text)
                            
#             #             if not raw_text: continue
                        
#             #             # ===========================================
#             #             # ✂️✂️✂️ 新增：智能切分逻辑 (修复长难句) ✂️✂️✂️
#             #             # ===========================================
#             #             # 1. 先把一大段话切成符合阅读习惯的短句
#             #             split_lines = self._split_text_smartly(raw_text, max_chars=30)
                        
#             #             # 2. 计算总字符数和总时长，用于分配时间轴
#             #             valid_chars = sum(len(s) for s in split_lines)
#             #             total_dur_sec = (e_ms - s_ms) / 1000.0
#             #             curr_srt_start = s_ms / 1000.0

#             #             # 3. 循环写入每一小句
#             #             for line in split_lines:
#             #                 if not line.strip(): continue
                            
#             #                 # 根据字数比例分配时间
#             #                 if valid_chars > 0:
#             #                     line_dur = (len(line) / valid_chars) * total_dur_sec
#             #                 else:
#             #                     line_dur = total_dur_sec
                            
#             #                 curr_srt_end = curr_srt_start + line_dur
                            
#             #                 # 写入结果
#             #                 full_line = line # 这里可以加 SPK 标签
#             #                 text_res += full_line + "\n"
#             #                 srt_res += self._make_srt_block(srt_idx, curr_srt_start, curr_srt_end, full_line)
                            
#             #                 # 更新下一句的开始时间
#             #                 srt_idx += 1
#             #                 curr_srt_start = curr_srt_end

#             #     except Exception as e:
#             #         log_ui(f"❌ Batch Error: {e}")

#             # # --- 循环填充 Buffer ---
#             # tensor_buf = []
#             # time_buf = []
            
#             # for i, (start_ms, end_ms) in enumerate(opt_segs):
#             #     # 过滤极短片段 (解决幻觉问题的关键！)
#             #     if (end_ms - start_ms) < 500: continue
                
#             #     # 切分音频
#             #     s_idx = int(start_ms * 16)
#             #     e_idx = int(end_ms * 16)
#             #     chunk = audio[max(0, s_idx-800):min(len(audio), e_idx+800)]
                
#             #     # FunASR 需要 Tensor
#             #     chunk_tensor = torch.from_numpy(chunk)
                
#             #     tensor_buf.append(chunk_tensor)
#             #     time_buf.append((start_ms, end_ms))
                
#             #     # 攒够了就发车
#             #     if len(tensor_buf) >= batch_size:
#             #         if i % (batch_size * 2) == 0: log_ui(f"   ⚡ 进度: {i}/{len(opt_segs)}")
#             #         process_batch(tensor_buf, time_buf)
#             #         tensor_buf = []
#             #         time_buf = []
            
#             # # 处理剩下的
#             # if tensor_buf:
#             #     process_batch(tensor_buf, time_buf)


#         # =========================================================
#         # 🐢 传统分支：Whisper 或其他模式 (保持原有串行逻辑)
#         # =========================================================
#         else:
#             log_ui("🐢 运行在串行模式 (Whisper 或 SeACo)...")
            
#             # --- 循环处理 ---
#             for i, (start_ms, end_ms) in enumerate(opt_segs):
#                 if i % 10 == 0:
#                     log_ui(f"   👉 进度: {i}/{total_segs}")

#                 s_idx = int(start_ms * 16); e_idx = int(end_ms * 16)
#                 chunk = audio[max(0, s_idx-800):min(len(audio), e_idx+800)]
#                 if len(chunk) < 1600: continue
                
#                 # A. SPK
#                 current_spk_label = ""
#                 if enable_spk and self.spk_engine:
#                     spk_id = self.spk_engine.get_speaker_id(chunk)
#                     if spk_id != "?":
#                         current_spk_label = f"[Spk {spk_id}]"
#                         last_spk_id = spk_id
#                     else:
#                         current_spk_label = f"[Spk {last_spk_id}]"

#                 # B. ASR
#                 seg_text = ""
#                 try:
#                     if self.backend == "faster":
#                         segs, _ = self.model.transcribe(chunk, beam_size=5, language=lang_arg, condition_on_previous_text=False, without_timestamps=True, initial_prompt=whisper_prompt)
#                         seg_text = "".join([s.text for s in segs]).strip()
                    
#                     elif self.backend == "funasr" and self.sub_mode == "emotion":
#                         res = self.model.generate(input=chunk, language=lang_arg, use_itn=True, disable_pbar=True)
#                         if res: 
#                             raw = res[0].get('text', '').strip()
#                             seg_text = self._clean_sensevoice_tags(raw)
                    
#                     elif self.backend == "funasr" and self.sub_mode == "nano":
#                         # 🔴 核心修复：传递 language 参数给魔改后的 model.py
#                         chunk_tensor = torch.from_numpy(chunk) 
#                         res = self.model.generate(
#                             input=[chunk_tensor], 
#                             batch_size_s=0, 
#                             disable_pbar=True,
#                             language=lang_arg  # <--- 这里的语言参数现在会被 model.py 识别了！
#                         )
#                         if res: seg_text = res[0].get('text', '').strip()

#                 except Exception as e:
#                     log_ui(f"⚠️ 识别出错 ({start_ms}ms): {e}")
#                     continue

#                 if not seg_text: continue
                
#                 # C. 字幕处理 (缓冲拼接)
#                 current_start_sec = start_ms / 1000.0
#                 current_end_sec = end_ms / 1000.0
#                 current_duration = current_end_sec - current_start_sec

#                 if buffer_text:
#                     seg_text = f"{buffer_text} {seg_text}"
#                     current_start_sec -= buffer_duration
#                     buffer_text = ""
#                     buffer_duration = 0.0

#                 split_lines = self._split_text_smartly(seg_text, max_chars=30)
                
#                 if split_lines:
#                     last_sent = split_lines[-1]
#                     is_complete = re.search(r'[\.?!。？！]$', last_sent)
#                     if not is_complete and i < len(opt_segs) - 1 and len(last_sent) < 20:
#                         buffer_text = last_sent
#                         total_len = sum(len(s) for s in split_lines)
#                         if total_len > 0:
#                             buffer_duration = (len(buffer_text) / total_len) * current_duration
#                         split_lines.pop()
#                         current_end_sec -= buffer_duration

#                 valid_chars = sum(len(s) for s in split_lines)
#                 valid_dur = current_end_sec - current_start_sec
#                 curr_srt_start = current_start_sec
                
#                 for line in split_lines:
#                     if not line.strip(): continue
#                     display_text = f"{current_spk_label} {line}".strip()
                    
#                     if valid_chars > 0:
#                         line_dur = (len(line) / valid_chars) * valid_dur
#                     else:
#                         line_dur = valid_dur
                    
#                     curr_srt_end = curr_srt_start + line_dur
                    
#                     text_res += display_text + "\n"
#                     srt_res += self._make_srt_block(srt_idx, curr_srt_start, curr_srt_end, display_text)
#                     srt_idx += 1
#                     curr_srt_start = curr_srt_end
                
#                 # 打印预览 (仅前几条)
#                 if i < 3: log_ui(f"   👀 预览: {display_text}")
#             # ... (前文 VAD 切分代码保持不变) ...
#     # opt_segs 是 VAD 切分好的时间戳列表 [[start, end], [start, end]...]


#             # ================= 🚀 核心修改结束 =================
#             if buffer_text:
#                 final_txt = f"{'[Spk ?]' if enable_spk else ''} {buffer_text}".strip()
#                 srt_res += self._make_srt_block(srt_idx, current_end_sec, current_end_sec + buffer_duration, final_txt)

#         # === 轨道二：SeACo (Legacy) ===
#             else:
#                 log_ui("🔄 Running SeACo Pipeline...")
#                 try:
#                     res = self.model.generate(input=input_path, batch_size_s=300, sentence_timestamp=True, return_spk_res=enable_spk)
#                     sentence_info = res[0].get('sentence_info', [])
#                     if sentence_info:
#                         srt_idx = 1
#                         for sent in sentence_info:
#                             text = sent.get('text', '')
#                             spk_tag = f"[Spk {sent.get('spk')}] " if enable_spk and 'spk' in sent else ""
#                             full_line = f"{spk_tag}{text}"
#                             start = sent['timestamp'][0][0] / 1000.0
#                             end = sent['timestamp'][-1][1] / 1000.0
#                             text_res += full_line + "\n"
#                             srt_res += self._make_srt_block(srt_idx, start, end, full_line)
#                             srt_idx += 1
#                 except Exception as e:
#                     log_ui(f"❌ SeACo Error: {e}")

#         # === 强制保存 ===
#         txt_path = os.path.join(final_out_dir, f"{file_stem}.txt")
#         srt_path = os.path.join(final_out_dir, f"{file_stem}.srt")
        
#         log_ui("💾 正在写入文件...")
#         with open(txt_path, "w", encoding="utf-8") as f: 
#             f.write(text_res if text_res else "(无识别内容)")
#         with open(srt_path, "w", encoding="utf-8") as f: 
#             f.write(srt_res if srt_res else "")
            
#         log_ui(f"🎉 全部完成！\nTXT: {txt_path}\nSRT: {srt_path}")

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

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--file", required=True)
#     parser.add_argument("--output_dir", required=True)
#     parser.add_argument("--backend", default="faster") 
#     parser.add_argument("--model_size", default="turbo")
#     parser.add_argument("--sub_mode", default="precision")
#     parser.add_argument("--language", default="auto")
#     parser.add_argument("--enable_spk", action="store_true")
#     parser.add_argument("--batch_size", type=int, default=8)

#     args = parser.parse_args()

#     # 🔥🔥🔥 漏掉的就是这一段！必须先创建引擎，才能跑！🔥🔥🔥
#     engine = SuperASREngine(
#         backend=args.backend,
#         model_size=args.model_size,
#         sub_mode=args.sub_mode
#     )
    
#     # 然后才是运行
#     engine.run(
#         args.file, 
#         args.output_dir, 
#         language=args.language,
#         enable_spk=args.enable_spk,
#         batch_size=args.batch_size
#     )
# ... (上面的 import 和 class SuperASREngine 代码保持不变，不要动) ...

# ================= 🚀 程序的入口 (修改了这里) =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 允许传入单个文件，或者一个包含文件列表的txt
    parser.add_argument("--file", default=None)
    parser.add_argument("--file_list", default=None) # 新增参数
    
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--backend", default="faster") 
    parser.add_argument("--model_size", default="turbo")
    parser.add_argument("--sub_mode", default="precision")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--enable_spk", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--folder_mode", action="store_true", help="开启扁平化文件夹归档模式")
    parser.add_argument("--parallel_mode", action="store_true", help="开启双模型并行识别")
    parser.add_argument("--parallel_workers", type=int, default=2)

    args = parser.parse_args()

    # 1. 初始化引擎 
    # ---------------------------------------------------------
    print(f"🔥 正在初始化... (并行模式: {args.parallel_mode})", flush=True)
    start_init = time.time()
    
    try:
        # 🔥 根据参数选择引擎
        if args.parallel_mode and args.backend == "funasr" and args.sub_mode in ["nano", "emotion"]:
            engine = ParallelASREngine(
                backend=args.backend, # funasr
                sub_mode=args.sub_mode, # nano/emotion
                folder_mode=args.folder_mode,
                num_workers=args.parallel_workers
            )
        else:
            # 传统单模型引擎
            if args.parallel_mode:
                print("⚠️ 警告: 并行模式仅支持 FunASR Nano/SenseVoice，已自动降级为单模型模式。", flush=True)
            engine = SuperASREngine(
                backend=args.backend,
                model_size=args.model_size,
                sub_mode=args.sub_mode,
                folder_mode=args.folder_mode
            )
            
        print(f"✅ 引擎加载完成，耗时: {time.time() - start_init:.2f}s", flush=True)
    except Exception as e:
        print(f"❌ 引擎初始化失败: {e}", flush=True)
        import traceback
        traceback.print_exc() # 打印详细堆栈
        sys.exit(1)

    # 2. 整理任务列表
    # ---------------------------------------------------------
    tasks = []
    if args.file_list and os.path.exists(args.file_list):
        # 如果传入的是列表文件，读取每一行
        with open(args.file_list, "r", encoding="utf-8") as f:
            for line in f:
                path = line.strip()
                if path and os.path.exists(path):
                    tasks.append(path)
    elif args.file:
        # 如果是单文件模式
        tasks.append(args.file)

    if not tasks:
        print("❌ 没有找到有效的待处理文件。", flush=True)
        sys.exit(0)

    print(f"📋 任务队列: 共 {len(tasks)} 个文件待处理", flush=True)

    # 3. 内部循环处理 (模型一直驻留在显存中)
    # ---------------------------------------------------------
    for i, file_path in enumerate(tasks):
        print(f"\n=======================================================", flush=True)
        print(f"▶️ [{i+1}/{len(tasks)}] 正在处理: {os.path.basename(file_path)}", flush=True)
        print(f"=======================================================", flush=True)
        
        try:
            engine.run(
                file_path, 
                args.output_dir, 
                language=args.language,
                enable_spk=args.enable_spk,
                batch_size=args.batch_size
            )
        except Exception as e:
            print(f"❌ 处理文件出错 {file_path}: {e}", flush=True)
            # 出错不退出，继续处理下一个
            continue
            
    print("\n🏁 所有任务执行完毕！", flush=True)    
    # engine.run(
    #     args.file, 
    #     args.output_dir, 
    #     language=args.language,
    #     enable_spk=args.enable_spk
    # )