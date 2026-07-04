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

try:
    from . import qwen_client
except ImportError:
    import qwen_client # Fallback for direct execution


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
from queue import Empty, Queue
# ===============================================

def log_ui(msg):
    """专门用于向 WebUI 发送状态的打印函数"""
    print(msg, flush=True)

# ================= 🔧 加载魔改版 Nano 模型 =================
# 尝试加载提取出来的本地修改版 model.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 如果是在 funclip 目录下，由于 asr1.py 在 funclip/ 里，我们需要往上一级找或者直接用绝对路径
# 这里为了稳妥，直接使用 E:\FunClip\FunClip\custom_nano_model.py
manual_model_path = r"E:\FunClip\FunClip\custom_nano_model.py"
if os.path.exists(manual_model_path):
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("funasr.models.fun_asr_nano.model", manual_model_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["funasr.models.fun_asr_nano.model"] = module
        spec.loader.exec_module(module)
        print(f"✅ 成功加载本地提取版 Nano 模型: {manual_model_path}")
    except Exception as e:
        print(f"⚠️ 加载本地 Nano 模型失败: {e}")

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
    """
    说话人识别器 - 离线聚类版本
    优点：全局分析所有片段后再分组，比在线聚类更准确
    """
    def __init__(self, spk_model_path, device="cuda"):
        log_ui("   [Init] 加载说话人模型 (Cam++)...")
        self.model = AutoModel(model=spk_model_path, trust_remote_code=True, device=device, disable_update=True, disable_pbar=True)
        self.distance_threshold = 0.6  # 🔥 聚类距离阈值 (调大: 更倾向于合并 -> 减少说话人数量)

    def extract_embedding(self, audio_chunk):
        """提取单个片段的声纹向量"""
        if len(audio_chunk) < 1600: 
            return None
        try:
            res = self.model.generate(input=[audio_chunk], disable_pbar=True)
            if not res or 'spk_embedding' not in res[0]: 
                return None
            emb = torch.tensor(res[0]['spk_embedding']).flatten().cpu().numpy()
            return emb
        except Exception as e:
            return None

    def cluster_speakers(self, audio_chunks):
        """
        离线聚类：先提取所有声纹，再全局聚类
        返回: dict[seg_idx -> speaker_id]
        """
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import cosine_distances
        import numpy as np
        
        # 1. 批量提取声纹
        log_ui(f"      📊 提取 {len(audio_chunks)} 个片段的声纹...")
        embeddings = []
        valid_indices = []
        
        for i, chunk in enumerate(audio_chunks):
            chunk_np = chunk.numpy() if hasattr(chunk, 'numpy') else chunk
            emb = self.extract_embedding(chunk_np)
            if emb is not None:
                embeddings.append(emb)
                valid_indices.append(i)
        
        if len(embeddings) < 2:
            # 片段太少，全部归为 spk1
            return {i: 1 for i in range(len(audio_chunks))}
        
        # 2. 计算余弦距离矩阵
        emb_matrix = np.vstack(embeddings)
        distance_matrix = cosine_distances(emb_matrix)
        
        # 3. 层次聚类 (不需要预设 K)
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            metric='precomputed',
            linkage='average'
        )
        labels = clustering.fit_predict(distance_matrix)
        
        # 4. 构建结果映射
        result = {}
        for idx, seg_idx in enumerate(valid_indices):
            result[seg_idx] = int(labels[idx]) + 1  # spk1, spk2, ...
        
        # 对于无效片段，标记为 "?"
        for i in range(len(audio_chunks)):
            if i not in result:
                result[i] = "?"
        
        n_speakers = len(set(labels))
        log_ui(f"      ✅ 聚类完成：检测到 {n_speakers} 个说话人")
        
        return result


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
    def run(self, input_path, output_dir, language="auto", enable_spk=False, batch_size=8, hotwords=None):
        # 🔥 热词列表 (用于提升专业术语识别准确率)
        hotwords = hotwords or []
        if hotwords:
            log_ui(f"🔥 启用热词: {', '.join(hotwords[:5])}{'...' if len(hotwords) > 5 else ''}")
        
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
        # 🛤️ 轨道X：Qwen3 (Docker) - 极速模式
        # =========================================================
        # =========================================================
        # 🛤️ 轨道X：Qwen3 (Docker) - 极速分片模式
        # =========================================================
        if self.backend == "qwen_vllm":
            log_ui("🚀 调用 Qwen3-vLLM 服务端 (Docker)...")
            try:
                from .qwen_client import QwenASRClient
            except ImportError:
                import qwen_client # Fallback
                from qwen_client import QwenASRClient

            client = QwenASRClient()
            
            # 🆕 Qwen3 语言参数映射 (ISO代码 -> 完整名称)
            QWEN3_LANG_MAP = {
                "auto": None,  # 自动检测
                "zh": "Chinese",
                "en": "English",
                "ja": "Japanese",
                "ko": "Korean",
                "yue": "Cantonese",
                "ar": "Arabic",
                "de": "German",
                "fr": "French",
                "es": "Spanish",
                "pt": "Portuguese",
                "id": "Indonesian",
                "it": "Italian",
                "ru": "Russian",
                "th": "Thai",
                "vi": "Vietnamese",
                "tr": "Turkish",
                "hi": "Hindi",
                "ms": "Malay",
                "nl": "Dutch",
                "sv": "Swedish",
                "da": "Danish",
                "fi": "Finnish",
                "pl": "Polish",
                "cs": "Czech",
                "fil": "Filipino",
                "fa": "Persian",
                "el": "Greek",
                "hu": "Hungarian",
                "mk": "Macedonian",
                "ro": "Romanian",
            }
            # 转换语言参数
            qwen_language = QWEN3_LANG_MAP.get(language, language if language in QWEN3_LANG_MAP.values() else None)
            log_ui(f"🔍 [Confidence Check] User Input: '{language}' -> Mapped to Qwen3: '{qwen_language}'")
            
            # --- 1. FFmpeg 预处理 (转为 WAV 以便 VAD 切分) ---
            log_ui("🔄 正在预处理视频/音频 (FFmpeg -> 16k WAV)...")
            import uuid
            import tempfile
            import subprocess
            
            temp_wav = os.path.join(tempfile.gettempdir(), f"asr_process_{uuid.uuid4()}.wav")
            try:
                # 强制转为 16k 单声道 WAV
                cmd = [
                    "ffmpeg", "-y", "-i", input_path,
                    "-vn", "-ac", "1", "-ar", "16000", 
                    "-f", "wav", temp_wav
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            except Exception as e:
                log_ui(f"❌ FFmpeg 转换失败: {e}")
                return

            try:
                # 🔥 每次处理文件前，先清理一下共享目录 (防止垃圾堆积)
                SHARED_DIR = r"E:\FunClip\qwen_server\shared_tmp"
                if os.path.exists(SHARED_DIR):
                    log_ui("🧹 正在清理共享临时目录...")
                    for f in os.listdir(SHARED_DIR):
                        if f.endswith(".wav"): # 只删 wav 防止误删
                            try: os.remove(os.path.join(SHARED_DIR, f))
                            except: pass

                # --- 2. VAD 切分 (解决长音频 Context Limit 问题) ---
                log_ui("✂️ 正在进行 VAD 切分 (长音频优化)...")
                audio, _ = librosa.load(temp_wav, sr=16000)
                
                vad_kwargs = {
                    "max_single_segment_time": 30000, # 30秒一刀 (Qwen context 安全区)
                    "speech_noise_thres": 0.8,
                    "max_end_silence_time": 300
                }
                vad_out = self.vad_model.generate(
                    input=temp_wav, 
                    batch_size_s=5000, 
                    **vad_kwargs
                )
                
                raw_segs = vad_out[0]['value'] if vad_out and len(vad_out)>0 and 'value' in vad_out[0] else [[0, len(audio)/16*1000]]
                # 合并得不需要太碎，30s左右挺好
                opt_segs = self._merge_vad_segments(raw_segs, max_gap_ms=500, max_duration_ms=25000)
                log_ui(f"✅ 切分完成: 共 {len(opt_segs)} 个片段")

                # =========================================================
                # 🔊 SPK 预处理：先对 VAD 片段做说话人聚类
                # =========================================================
                spk_cache = {}
                chunk_audios = []
                for seg in opt_segs:
                    start_ms, end_ms = seg
                    start_sample = int(start_ms * 16)
                    end_sample = int(end_ms * 16)
                    chunk_audios.append(audio[start_sample:end_sample])

                if enable_spk and self.spk_engine is not None:
                    try:
                        log_ui("🔊 [SPK] 开始对 VAD 片段进行说话人聚类...")
                        spk_start_time = time.time()
                        spk_cache = self.spk_engine.cluster_speakers(chunk_audios)
                        log_ui(f"   ✅ SPK 聚类完成！耗时: {time.time() - spk_start_time:.2f}s")
                    except Exception as e:
                        log_ui(f"⚠️ [SPK] 聚类失败，已跳过说话人识别: {e}")
                        spk_cache = {}

                # --- 3. 并发发送片段 (ThreadPool) ---
                full_text_list = []
                full_timestamps = []
                # 收集临时文件以便清理
                segment_files = []

                import soundfile as sf
                from tqdm import tqdm
                from concurrent.futures import ThreadPoolExecutor, as_completed
                
                t_start_all = time.time()
                
                # 生成所有切片文件的路径列表
                batch_paths = []
                # Keep track of segment info for results processing
                # tasks structure: [(index, (start_ms, end_ms), path), ...]
                tasks = []  # 🔥 修复：必须初始化局部变量，否则会污染全局 tasks！
                
                # 🛠️ SPK 0: 初始化模型 (懒加载)
                if enable_spk and self.spk_engine is None:
                    log_ui("🔊 [SPK] 初始化说话人识别模型 (Cam++)...")
                    try:
                        self.spk_engine = SpeakerDiarizer(MODEL_PATHS["spk"], device=self.device)
                    except Exception as e:
                        log_ui(f"❌ [SPK] 模型加载失败: {e}")
                
                log_ui(f"📦 准备批处理: 共 {len(opt_segs)} 个片段...")
                
                for i, seg in enumerate(opt_segs):
                    start_ms, end_ms = seg
                    chunk_audio = chunk_audios[i] if i < len(chunk_audios) else audio[int(start_ms * 16):int(end_ms * 16)]
                    
                    # 🔥 性能优化：直接写到共享目录，避免 C 盘 IO 和后续复制
                    SHARED_DIR = r"E:\FunClip\qwen_server\shared_tmp"
                    if os.path.exists(SHARED_DIR):
                        base_dir = SHARED_DIR
                    else:
                        base_dir = tempfile.gettempdir()
                        
                    seg_tmp_path = os.path.join(base_dir, f"seg_{uuid.uuid4()}.wav")
                    sf.write(seg_tmp_path, chunk_audio, 16000)
                    segment_files.append(seg_tmp_path)
                    batch_paths.append(seg_tmp_path)
                    
                    tasks.append((i, seg, seg_tmp_path))

                # 获取 batch_size (默认24)
                # 直接使用方法参数传入的 batch_size
                bs = batch_size
                if bs is None or int(bs) <= 0: bs = 24
                else: bs = int(bs)
                
                log_ui(f"🚀 发送批处理请求 (总片段={len(batch_paths)} | 批大小={bs})...")
                
                # 结果容器 (按 index 排序)
                results_map = {}
                
                # 分块处理
                total_chunks = (len(batch_paths) + bs - 1) // bs
                
                for chunk_idx in range(total_chunks):
                    start_i = chunk_idx * bs
                    end_i = min((chunk_idx + 1) * bs, len(batch_paths))
                    
                    chunk_paths = batch_paths[start_i:end_i]
                    log_ui(f"   🌊 处理批次 [{chunk_idx+1}/{total_chunks}] ({len(chunk_paths)} 段)...")
                    
                    t_batch_start = time.time() # ⏱️ 批次开始计时
                    try:
                        # 批量调用
                        batch_res = client.transcribe_batch(
                            chunk_paths, 
                            language=qwen_language,  # 🆕 使用 Qwen3 格式的语言名称 
                            return_timestamps=True, 
                            preprocess=False, 
                            verbose=True # 开启日志以排查问题
                        )
                        
                        t_batch_end = time.time() # ⏱️ 批次结束计时
                        batch_dur = t_batch_end - t_batch_start
                        log_ui(f"      ⏱️ 批次 [{chunk_idx+1}] 耗时: {batch_dur:.2f}s (Avg: {batch_dur/len(chunk_paths):.2f}s/seg)")

                        if batch_res and len(batch_res) == len(chunk_paths):
                            # 将结果映射回全局 index
                            for j, res in enumerate(batch_res):
                                global_idx = start_i + j
                                results_map[global_idx] = (res, None)
                        else:
                            log_ui(f"⚠️ 批次 [{chunk_idx+1}] 结果数量不匹配或为空")
                            # 填补 None 防止后续崩溃
                            for j in range(len(chunk_paths)):
                                global_idx = start_i + j
                                results_map[global_idx] = (None, "Batch Error")
                                
                    except Exception as e:
                        t_batch_end = time.time()
                        log_ui(f"❌ 批次 [{chunk_idx+1}] 失败 (耗时 {t_batch_end - t_batch_start:.2f}s): {e}")
                        for j in range(len(chunk_paths)):
                             global_idx = start_i + j
                             results_map[global_idx] = (None, str(e))

                log_ui(f"✅ 所有批处理完成！")
                
                # =========================================================
                # 🔥 逻辑重构: 5s 智能凑句 (Smart Merge with Hierarchical Punc)
                # =========================================================
                # 用户要求: "句号切，其他标点不切，准备凑...满5s结算"
                
                final_subtitles = []
                full_text_list = []
                
                # 标点定义
                HARD_PUNC = set(['。', '？', '！', '?', '!', '；', ';', '.'])
                
                log_ui("🔄 正在执行: 5s 智能凑句逻辑 (Hierarchical Merge v3.1)...")
                
                import re
                
                for i in range(len(tasks)):
                    if i not in results_map: continue
                    res, err = results_map[i]
                    if not res or "text" not in res: continue
                    
                    text_seg = res["text"].strip()
                    ts_list = res.get("timestamps", [])
                    if not text_seg: continue
                    
                    # 基础信息
                    seg_start_ms, seg_end_ms = tasks[i][1]
                    seg_start_sec = seg_start_ms / 1000.0
                    
                    # --- SPK 逻辑 (VAD 片段预聚类结果) ---
                    seg_spk_id = None
                    if enable_spk and spk_cache:
                        seg_spk_id = spk_cache.get(i)
                    
                    # Fallback (无时间戳)
                    if not ts_list:
                        display_text = text_seg
                        if enable_spk and seg_spk_id not in (None, "?"):
                            display_text = f"[spk{seg_spk_id}] {display_text}"
                        full_text_list.append(display_text)
                        final_subtitles.append({
                            "text": display_text,
                            "start": seg_start_ms,
                            "end": seg_end_ms,
                            "char_timestamps": [],
                            "spk": seg_spk_id
                        })
                        continue

                    # --- Step 1: 提取原子短语 (Atomic Phrases) ---
                    # 规则: 遇到 [任何标点] 或 [停顿 > 0.8s] 就划为一个原子单元
                    atomic_phrases = []
                    curr_phrase_tokens = []
                    search_pos = 0
                    
                    for t_idx, token in enumerate(ts_list):
                        token_text = token.get("text", "").strip()
                        if not token_text: continue
                        
                        # 定位开始位置 (跳过可能的空格)
                        token_start_pos = text_seg.find(token_text, search_pos)
                        if token_start_pos < 0:
                            # 容错处理：如果没找到，就用 search_pos
                            token_start_pos = search_pos
                        
                        search_pos = token_start_pos + len(token_text)
                        
                        # 向后探测标点和空格 (贪婪匹配)
                        # 修改正则，涵盖英文句号 .
                        match = re.match(r'^(\s*[。？！?!；;，,、…—.\s]*)', text_seg[search_pos:])
                        trailing_stuff = ""
                        is_hard_punc = False
                        has_any_punc = False
                        
                        if match:
                            trailing_stuff = match.group(1)
                            # 记录查找起始位置用于精准探测
                            punc_search_base = search_pos 
                            search_pos += len(trailing_stuff)
                            
                            # 检测是否包含硬标点
                            for rel_idx, char in enumerate(trailing_stuff):
                                if char == '.':
                                    # 🔥 核心防误切逻辑: 如果句号后紧跟数字 (如 0.1), 则判定为小数点
                                    abs_char_idx = punc_search_base + rel_idx
                                    if abs_char_idx + 1 < len(text_seg):
                                        if text_seg[abs_char_idx + 1].isdigit():
                                            continue # 跳过小数点，不作为断句标点
                                    is_hard_punc = True
                                    has_any_punc = True
                                elif char in HARD_PUNC:
                                    is_hard_punc = True
                                    has_any_punc = True
                                elif char in ['，', ',', '、', '…', '—']:
                                    has_any_punc = True
                                
                        token['slice_start'] = token_start_pos
                        token['slice_end'] = search_pos
                        token['abs_start'] = (token['start'] + seg_start_sec) * 1000
                        token['abs_end'] = (token['end'] + seg_start_sec) * 1000
                        
                        curr_phrase_tokens.append(token)
                        
                        # 判断原子断点
                        is_atomic_break = False
                        if has_any_punc: 
                            is_atomic_break = True
                        elif t_idx < len(ts_list) - 1:
                            gap = ts_list[t_idx+1]['start'] - token['end']
                            if gap > 0.8: # 用户要求，宽松一点
                                is_atomic_break = True
                        else:
                            is_atomic_break = True
                            
                        if is_atomic_break:
                            # 直接从原句切片，完美保留空格标点
                            s_pos = curr_phrase_tokens[0]['slice_start']
                            e_pos = curr_phrase_tokens[-1]['slice_end']
                            phrase_text = text_seg[s_pos:e_pos]
                            
                            atomic_phrases.append({
                                "text": phrase_text,
                                "start": curr_phrase_tokens[0]['abs_start'],
                                "end": curr_phrase_tokens[-1]['abs_end'],
                                "tokens": curr_phrase_tokens,
                                "has_hard_punc": is_hard_punc,
                                "slice_range": (s_pos, e_pos)
                            })
                            curr_phrase_tokens = []

                    # --- Step 2: 智能凑句 (Smart Merge) ---
                    # 规则: 只要未满 5s 且没遇到硬标点，就一直凑下去
                    if not atomic_phrases: continue
                    
                    buffer = []
                    for idx, p in enumerate(atomic_phrases):
                        buffer.append(p)
                        
                        # 计算当前 buffer 跨度
                        curr_dur = buffer[-1]['end'] - buffer[0]['start']
                        
                        should_flush = False
                        
                        # 1. 遇到硬标点 (句号等) -> 强制结算
                        if p['has_hard_punc']:
                            should_flush = True
                        
                        # 2. 时长预判: 如果加上下一句就爆 5s 了 -> 结算当前
                        elif idx < len(atomic_phrases) - 1:
                            next_phrase = atomic_phrases[idx+1]
                            estimated_dur = next_phrase['end'] - buffer[0]['start']
                            if estimated_dur > 5000: # 超过 5秒
                                should_flush = True
                        
                        # 3. 最后一个原子短语 -> 结算
                        else:
                            should_flush = True
                            
                        if should_flush:
                            # 再次执行大范围原句切片，确保拼接处也完美
                            s_off = buffer[0]['slice_range'][0]
                            e_off = buffer[-1]['slice_range'][1]
                            merged_text = text_seg[s_off:e_off].strip()
                            display_text = merged_text
                            if enable_spk and seg_spk_id not in (None, "?"):
                                display_text = f"[spk{seg_spk_id}] {display_text}"
                            
                            merged_ts = []
                            for ph in buffer:
                                merged_ts.extend(ph['tokens'])
                                
                            full_text_list.append(display_text)
                            final_subtitles.append({
                                "text": display_text,
                                "start": buffer[0]['start'],
                                "end": buffer[-1]['end'],
                                "char_timestamps": merged_ts, # ✅ 完美保留字级别信息
                                "spk": seg_spk_id
                            })
                            buffer = []

                total_process_time = time.time() - t_start_all
                log_ui(f"🏁 流程总耗时: {total_process_time:.2f}s")

                full_text = " ".join(full_text_list)
                log_ui(f"✅ 全文转写完成，耗时: {total_process_time:.2f}s")
                
                # SRT 时间格式转换 Helper
                def format_srt_time(ms):
                    seconds, milliseconds = divmod(ms, 1000)
                    minutes, seconds = divmod(seconds, 60)
                    hours, minutes = divmod(minutes, 60)
                    return int(hours), int(minutes), int(seconds), int(milliseconds)

                srt_lines = []
                for idx, t in enumerate(final_subtitles):
                    s_h, s_m, s_s, s_ms = format_srt_time(t['start'])
                    e_h, e_m, e_s, e_ms = format_srt_time(t['end'])
                    srt_lines.append(f"{idx+1}\n{s_h:02}:{s_m:02}:{s_s:02},{s_ms:03} --> {e_h:02}:{e_m:02}:{e_s:02},{e_ms:03}\n{t['text']}\n")
                
                srt_res = "\n".join(srt_lines)
                
                # 保存
                text_res = full_text
                log_ui(f"💾 准备保存: 文本长度={len(text_res)}, 输出目录={final_out_dir}")
                if text_res:
                    # 🆕 确保输出目录存在
                    os.makedirs(final_out_dir, exist_ok=True)
                    
                    txt_path = os.path.join(final_out_dir, f"{file_stem}.txt")
                    srt_path = os.path.join(final_out_dir, f"{file_stem}.srt")
                    with open(txt_path, "w", encoding="utf-8") as f: f.write(text_res)
                    with open(srt_path, "w", encoding="utf-8") as f: f.write(srt_res)
                    
                    # =========================================================
                    # 🎤 ASS 卡拉 OK 字幕生成 (逐字高亮)
                    # =========================================================
                    ass_path = os.path.join(final_out_dir, f"{file_stem}.ass")
                    
                    # ASS 文件头
                    ass_header = """[Script Info]
; Generated by FunClip Qwen3-ASR
Title: Karaoke Subtitles
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,微软雅黑,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,10,10,30,1
Style: Karaoke,微软雅黑,56,&H00FFFFFF,&H0000D4FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,2,2,10,10,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
                    
                    def format_ass_time(ms):
                        total_cs = int(ms / 10)  # 转为厘秒
                        cs = total_cs % 100
                        total_s = total_cs // 100
                        s = total_s % 60
                        total_m = total_s // 60
                        m = total_m % 60
                        h = total_m // 60
                        return f"{h}:{m:02}:{s:02}.{cs:02}"
                    
                    # 🆕 时间戳平滑算法
                    def smooth_timestamps(tokens):
                        if not tokens: return []
                        smoothed = [t.copy() for t in tokens]
                        
                        for i in range(len(smoothed)):
                            curr = smoothed[i]
                            curr_dur = curr["end"] - curr["start"]
                            
                            # 如果当前词时长为0（或极短），且不是最后一个词
                            if curr_dur < 0.05 and i < len(smoothed) - 1:
                                next_token = smoothed[i+1]
                                gap = next_token["start"] - curr["end"]
                                
                                # 如果与下一个词之间有空隙，则借用一部分空隙
                                if gap > 0.05:
                                    # 最多借用空隙的 80%，或者补足到 0.2秒
                                    borrow = min(gap * 0.8, 0.2)
                                    curr["end"] += borrow
                                    # 注意：不改变 next_token 的 start，只填充空隙
                                    
                        return smoothed

                    ass_events = []
                    
                    # 🆕 遍历 final_subtitles (与 SRT 行划分一致)
                    puncts = set(['。', '？', '！', '?', '!', '，', ',', '、', '：', ':', '"', '"', '"', "'", ''', ''', '；', ';', '（', '）', '(', ')', '…', '—', '·'])
                    
                    for sub in final_subtitles:
                        # 使用原始文本作为参考（虽然 ASS 主要靠 tokens 生成）
                        text = sub.get("text", "")
                        start_ms = sub.get("start", 0)
                        end_ms = sub.get("end", 0)
                        
                        # 1. 获取并平滑时间戳
                        raw_ts = sub.get("char_timestamps", [])
                        char_ts = smooth_timestamps(raw_ts)
                        
                        if not char_ts:
                            # 🔥 修复：如果缺少时间戳，降级为普通字幕 (Style: Default)，而不是跳过
                            start_str = format_ass_time(start_ms)
                            end_str = format_ass_time(end_ms)
                            ass_events.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text}")
                            continue
                        
                        # 2. 智能对齐：将时间戳 tokens 映射回带标点的 text
                        # 逻辑：遍历 text（含标点），尝试匹配当前词 token。如果匹配成功，插入 \kf；否则直接附加（标点/空格）。
                        karaoke_text = ""
                        prev_end_sec = start_ms / 1000.0
                        
                        ts_idx = 0
                        text_idx = 0
                        
                        # --- 🆕 预处理：跳过开头的 [spkX] 标签以防干扰对齐 ---
                        if text.startswith("[spk"):
                            end_bracket = text.find("]", 0)
                            if end_bracket != -1:
                                # 把标签直接加到输出，不参与对齐
                                karaoke_text += text[:end_bracket + 1]
                                text_idx = end_bracket + 1
                        
                        while text_idx < len(text):
                            matched = False
                            
                            # 尝试匹配当前时间戳 token
                            if ts_idx < len(char_ts):
                                token_item = char_ts[ts_idx]
                                # 🔥 核心修复：Qwen 返回的是 "text" 字段，不是 "char"
                                token_word = token_item.get("text", "")
                                
                                # 检查 text 当前位置是否正好是这个词
                                # 简单前缀匹配 (忽略大小写)
                                if token_word and text[text_idx:].lower().startswith(token_word.lower()):
                                    # --- 匹配成功！生成卡拉OK标签 ---
                                    token_start = token_item.get("abs_start", start_ms) / 1000.0 if "abs_start" in token_item else token_item.get("start", prev_end_sec)
                                    token_end = token_item.get("abs_end", start_ms + 100) / 1000.0 if "abs_end" in token_item else token_item.get("end", token_start + 0.1)
                                    
                                    # 1. 等待时间 \k (仅在词前插入，且需有显著空隙)
                                    gap_sec = token_start - prev_end_sec
                                    if gap_sec > 0.05:
                                        gap_cs = int(gap_sec * 100)
                                        karaoke_text += f"{{\\k{gap_cs}}}"
                                        prev_end_sec = token_start
                                    
                                    # 2. 持续时间 \kf
                                    dur_sec = token_end - prev_end_sec
                                    dur_cs = int(dur_sec * 100)
                                    if dur_cs < 1: dur_cs = 1
                                    
                                    # 使用原始文本中的词（保留大小写），确保显示的是原句
                                    original_word = text[text_idx : text_idx + len(token_word)]
                                    karaoke_text += f"{{\\kf{dur_cs}}}{original_word}"
                                    
                                    prev_end_sec = token_end
                                    text_idx += len(token_word)
                                    ts_idx += 1
                                    matched = True
                            
                            if not matched:
                                # 未匹配（标点符号、空格、或无法对应的字符），直接纯文本输出
                                karaoke_text += text[text_idx]
                                text_idx += 1
                        
                        # 时间格式转换
                        start_str = format_ass_time(start_ms)
                        end_str = format_ass_time(end_ms)
                        
                        ass_events.append(f"Dialogue: 0,{start_str},{end_str},Karaoke,,0,0,0,,{karaoke_text}")
                    
                    # 写入 ASS 文件
                    with open(ass_path, "w", encoding="utf-8") as f:
                        f.write(ass_header)
                        f.write("\n".join(ass_events))
                    
                    log_ui(f"🎉 处理完成！\nSRT: {srt_path}\n🎤 ASS (卡拉OK): {ass_path}")

            except Exception as e:
                log_ui(f"❌ 处理过程中发生致命错误: {e}")
                import traceback
                log_ui(traceback.format_exc())
            
            finally:
                # 清理临时文件
                if os.path.exists(temp_wav):
                    try: os.remove(temp_wav)
                    except: pass
                for f in segment_files:
                    try: os.remove(f)
                    except: pass
                    
            return

        # =========================================================
        # 🛤️ 轨道一：手动流水线 (Nano / SenseVoice / Whisper)
        # =========================================================
        if self.backend in ["faster", "openai"] or (self.backend == "funasr" and self.sub_mode in ["emotion", "nano", "sensevoice"]):
            
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
            is_funasr_batch = (self.backend == "funasr" and self.sub_mode in ["nano", "emotion", "sensevoice", "seaco"])
            
            # [分支 C] Qwen3-vLLM 后端 (Docker) - 🔥 优化版：VAD切分 + 共享卷 + 批处理
            if self.backend == "qwen_vllm":
                log_ui("🚀 调用 Qwen3-vLLM 服务端 (Docker + Shared Volume)...")
                try:
                    from .qwen_client import QwenASRClient
                except ImportError:
                    import qwen_client
                    from qwen_client import QwenASRClient

                client = QwenASRClient()
                
                # 🆕 Qwen3 语言参数映射 (与上面分支一致)
                QWEN3_LANG_MAP = {
                    "auto": None, "zh": "Chinese", "en": "English", "ja": "Japanese",
                    "ko": "Korean", "yue": "Cantonese", "ar": "Arabic", "de": "German",
                    "fr": "French", "es": "Spanish", "pt": "Portuguese", "id": "Indonesian",
                    "it": "Italian", "ru": "Russian", "th": "Thai", "vi": "Vietnamese",
                    "tr": "Turkish", "hi": "Hindi", "ms": "Malay", "nl": "Dutch",
                }
                qwen_language = QWEN3_LANG_MAP.get(language, language if language in QWEN3_LANG_MAP.values() else None)
                
                # 🔥 1. 利用 VAD 切分结果，保存到共享目录
                import shutil
                import uuid
                SHARED_HOST_DIR = r"E:\FunClip\qwen_server\shared_tmp"
                
                if not os.path.exists(SHARED_HOST_DIR):
                    os.makedirs(SHARED_HOST_DIR, exist_ok=True)
                
                chunk_paths = []
                chunk_times = []  # 保存原始时间用于 SRT
                session_id = str(uuid.uuid4())[:8]  # 防止文件名冲突
                
                log_ui(f"⏳ 准备 {len(opt_segs)} 个音频片段到共享目录...")
                
                for i, (start_ms, end_ms) in enumerate(opt_segs):
                    s_idx = int(start_ms * 16)
                    e_idx = int(end_ms * 16)
                    # 加 padding
                    chunk = audio[max(0, s_idx-800):min(len(audio), e_idx+800)]
                    
                    if len(chunk) < 1600:
                        continue
                    
                    # 保存到共享目录
                    chunk_filename = f"qwen_{session_id}_{i:04d}.wav"
                    chunk_path = os.path.join(SHARED_HOST_DIR, chunk_filename)
                    
                    import soundfile as sf
                    sf.write(chunk_path, chunk, 16000)
                    
                    chunk_paths.append(chunk_path)
                    chunk_times.append((start_ms, end_ms))
                
                log_ui(f"✅ 已保存 {len(chunk_paths)} 个片段到共享卷")
                
                # 🔥 2. 分批发送 (防止显存溢出!)
                QWEN_BATCH_SIZE = 4  # 🔥 Qwen2-Audio 很重，每批只处理 4 个
                t_start = time.time()
                batch_results = []
                
                total_batches = (len(chunk_paths) + QWEN_BATCH_SIZE - 1) // QWEN_BATCH_SIZE
                log_ui(f"🚀 开始分批处理 ({total_batches} 批, 每批 {QWEN_BATCH_SIZE} 个)...")
                
                for batch_idx in range(0, len(chunk_paths), QWEN_BATCH_SIZE):
                    batch_chunk_paths = chunk_paths[batch_idx:batch_idx + QWEN_BATCH_SIZE]
                    current_batch_num = batch_idx // QWEN_BATCH_SIZE + 1
                    
                    log_ui(f"   ⚡ 批次 [{current_batch_num}/{total_batches}] 处理 {len(batch_chunk_paths)} 个片段...")
                    
                    try:
                        results = client.transcribe_batch(
                            batch_chunk_paths, 
                            language=qwen_language,  # 🆕 使用 Qwen3 格式的语言名称 
                            return_timestamps=True,
                            preprocess=False,
                            verbose=False  # 减少日志噪音
                        )
                        if results:
                            batch_results.extend(results)
                        else:
                            # 如果这批失败，填充 None
                            batch_results.extend([None] * len(batch_chunk_paths))
                    except Exception as e:
                        log_ui(f"   ⚠️ 批次 {current_batch_num} 失败: {e}")
                        batch_results.extend([None] * len(batch_chunk_paths))
                
                total_process_time = time.time() - t_start
                log_ui(f"✅ Qwen3 批量转写完成，耗时: {total_process_time:.2f}s")
                
                # 🔥 3. 合并结果生成 SRT
                if batch_results:
                    for j, res in enumerate(batch_results):
                        if not res:
                            continue
                        
                        seg_text = res.get("text", "").strip()
                        if not seg_text:
                            continue
                        
                        start_ms, end_ms = chunk_times[j]
                        text_res += seg_text + "\n"
                        srt_res += self._make_srt_block(srt_idx, start_ms/1000.0, end_ms/1000.0, seg_text)
                        srt_idx += 1
                    
                    log_ui(f"🎉 生成 {srt_idx-1} 条字幕")
                else:
                    log_ui("❌ Qwen3 服务端返回空结果")
                    
                # 🔥 4. 清理临时文件
                for p in chunk_paths:
                    try: os.remove(p)
                    except: pass
                
                # 🔥 5. 保存结果
                if text_res:
                    txt_path = os.path.join(final_out_dir, f"{file_stem}.txt")
                    srt_path = os.path.join(final_out_dir, f"{file_stem}.srt")
                    with open(txt_path, "w", encoding="utf-8") as f: f.write(text_res)
                    with open(srt_path, "w", encoding="utf-8") as f: f.write(srt_res)
                    log_ui(f"🎉 Qwen3 处理完成！\nSRT: {srt_path}")
                else:
                    log_ui("⚠️ 警告: 识别结果为空！")
                
                return  # 🔥 显式终止，防止掉入其他逻辑

            # [分支 A] FunASR 并发模式 (🔥 修复：移到正确位置，不再嵌套在 if text_res 里)
            elif is_funasr_batch:
                from concurrent.futures import ThreadPoolExecutor  # 🔥 确保导入
                
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
                        # 开启 ITN 找回标点 + 热词注入
                        res_batch = self.model.generate(
                            input=chunk_tensors,
                            batch_size_s=0,
                            disable_pbar=True,
                            language=language,
                            use_itn=True,
                            hotwords=hotwords  # 🔥 热词注入
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
                # 🔥 SPK 离线聚类：一次性分析所有片段，全局分组
                # =========================================================
                spk_cache = {}  # key: segment_index, value: speaker_id
                
                if enable_spk and self.spk_engine is not None:
                    log_ui(f"   🎭 Step 2: 离线聚类说话人识别 ({len(all_tensors)} 个片段)...")
                    spk_start_time = time.time()
                    
                    # 🔥 新版：一次性全局聚类
                    spk_cache = self.spk_engine.cluster_speakers(all_tensors)
                    
                    log_ui(f"   ✅ SPK 聚类完成！耗时: {time.time() - spk_start_time:.2f}s")
                
                # 现在 process_batch 可以直接查询 spk_cache
                # ========================================================
                # 🔥 ASYNC PIPELINE: GPU Producer -> Queue -> CPU Consumer
                # ========================================================
                pipeline_queue = Queue()
                
                def result_consumer_worker():
                    nonlocal srt_idx, text_res, srt_res, processed_count, total_infer_dur_s, total_audio_dur_s
                    while True:
                        item = pipeline_queue.get()
                        if item is None: break
                        
                        res_batch, time_ranges, batch_offset, batch_cost = item
                        
                        # Metrics Update
                        processed_count += len(time_ranges)
                        batch_audio_dur = sum((e-s) for s,e in time_ranges) / 1000.0
                        total_infer_dur_s += batch_cost
                        total_audio_dur_s += batch_audio_dur
                        rtf = batch_audio_dur / batch_cost if batch_cost > 0.001 else 0.0
                        
                        q_size = pipeline_queue.qsize()
                        progress_pct = (processed_count / total_segments) * 100
                        log_ui(f"   ⚡ [{progress_pct:.1f}%] 已处理 {processed_count}/{total_segments} | 耗时: {batch_cost:.2f}s | Speed: {rtf:.1f}x | Q剩余: {q_size}")

                        try:
                            for idx, res in enumerate(res_batch):
                                global_seg_idx = batch_offset + idx
                                s_ms, e_ms = time_ranges[idx]
                                raw_text = res.get('text', '').strip()
                                
                                if idx < 3:
                                    log_ui(f"      片段{idx}: '{raw_text[:50]}...' " if len(raw_text) > 50 else f"      片段{idx}: '{raw_text}'")
                                
                                if self.sub_mode == "emotion":
                                    raw_text = self._clean_sensevoice_tags(raw_text)
                                if not raw_text: continue
                                
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
                            log_ui(f"❌ Consumer Error: {e}")
                            import traceback
                            traceback.print_exc()
                        finally:
                            pipeline_queue.task_done()

                # Start Consumer Thread
                consumer_thread = threading.Thread(target=result_consumer_worker, daemon=True)
                consumer_thread.start()

                # Optimized Producer Function
                def process_batch_with_spk_cache(chunk_tensors, time_ranges, batch_offset):
                    if not chunk_tensors: return

                    t_start = time.time()
                    try:
                        # 纯推理 (GPU) + 热词注入
                        res_batch = self.model.generate(
                            input=chunk_tensors,
                            batch_size_s=0,
                            disable_pbar=True,
                            language=language,
                            use_itn=True,
                            hotwords=hotwords  # 🔥 热词注入
                        )
                        batch_cost = time.time() - t_start
                        
                        # 立即放入队列 (CPU处理)
                        pipeline_queue.put((res_batch, time_ranges, batch_offset, batch_cost))
                        
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

                # Wait for consumer to finish
                log_ui("⏳ 等待后台处理完成...")
                pipeline_queue.join()
                pipeline_queue.put(None)
                consumer_thread.join()
                
                log_ui(f"🏁 FunASR 处理完毕！总耗时: {time.time() - global_start_time:.2f}s")
                
                # 🔥 必须在这里保存，否则 return 就没了！
                if text_res:
                    txt_path = os.path.join(final_out_dir, f"{file_stem}.txt")
                    srt_path = os.path.join(final_out_dir, f"{file_stem}.srt")
                    with open(txt_path, "w", encoding="utf-8") as f: f.write(text_res)
                    with open(srt_path, "w", encoding="utf-8") as f: f.write(srt_res)
                    log_ui(f"🎉 处理完成！\nSRT: {srt_path}")
                else:
                    log_ui("⚠️ 警告: 识别结果为空！")

                return # 🔥 显式终止，防止掉入 SeACo 逻辑

            # [分支 B] Whisper 串行模式
            else:
                log_ui("🐢 运行在串行模式 (Whisper)...")
                # ... (rest of whisper logic unchanged, no need to repeat all if just adding return)
                buffer_text = ""
                # ...
                
                # 由于 replace_file_content 限制，我必须完整替换块或精准定位
                # 为了安全，我只替换尾部 return

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
                # SeACo 自带 VAD 和推理
                # 🔥 动态调整 batch_size_s 以防止显存爆炸
                # 默认 batch_size_s=60秒 (比较安全)
                safe_batch_s = 60
                if batch_size and int(batch_size) > 0:
                     safe_batch_s = int(batch_size) * 5
                     if safe_batch_s > 300: safe_batch_s = 300
                
                log_ui(f"   ⚙️ [Legacy] 动态批处理: {safe_batch_s}秒/批 (防止OOM)")

                res = self.model.generate(
                    input=input_path, 
                    batch_size_s=safe_batch_s, 
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
#         if self.backend in ["faster", "openai"] or (self.backend == "funasr" and self.sub_mode in ["emotion", "nano", "sensevoice", "seaco"]):
            
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

    def _build_srt_from_qwen_timestamps(self, timestamps, max_chars=25, max_gap=1.0):
        """将 Qwen 的字级别时间戳聚合成字幕行"""
        srt_content = ""
        idx = 1
        
        if not timestamps: return ""
        
        current_line_text = ""
        current_start = timestamps[0]['start']
        current_end = timestamps[0]['end']
        current_line_text = timestamps[0]['text']
        
        for i in range(1, len(timestamps)):
            item = timestamps[i]
            gap = item['start'] - current_end
            
            # 判断是否换行：
            # 1. 标点符号 (且之前已有一定长度)
            is_punctuation = current_line_text[-1] in ".?!。？！，,、" if current_line_text else False
            # 2. 字数超限 (防止单行太长)
            is_long = len(current_line_text) >= max_chars
            # 3. 这里的 Gap 是指字与字之间的停顿，超过 1秒 通常意味着换气/新句子
            is_big_gap = gap > max_gap
            
            if (is_punctuation and len(current_line_text) > 5) or is_long or is_big_gap:
                # 结算当前行
                srt_content += self._make_srt_block(idx, current_start, current_end, current_line_text)
                idx += 1
                # 开启新行
                current_start = item['start']
                current_line_text = item['text']
            else:
                # 追加到当前行
                current_line_text += item['text']
                
            current_end = item['end']
            
        # 结算最后一行
        if current_line_text:
            srt_content += self._make_srt_block(idx, current_start, current_end, current_line_text)
            
        return srt_content

    def _make_srt_block(self, idx, start, end, text):
        if not text.strip(): return ""
        if end <= start: end = start + 0.01
        def fmt(t):
            h, r = divmod(t, 3600)
            m, s = divmod(r, 60)
            return f"{int(h):02}:{int(m):02}:{int(s):02},{int((t%1)*1000):03}"
        return f"{idx}\n{fmt(start)} --> {fmt(end)}\n{text}\n\n"

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
    # 🔥 新增：热词参数
    parser.add_argument("--hotwords", default="", help="热词列表，用逗号分隔，例如: FunASR,语音识别,人工智能")
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
    if tasks:
        print(f"🐛 DEBUG: tasks[0] type={type(tasks[0])} content={tasks[0]}", flush=True)

    for i, file_obj in enumerate(tasks):
        file_path = str(file_obj)
        
        # 兼容性修复：防止 file_path 是 tuple
        if isinstance(file_obj, tuple):
             # 尝试在 tuple 中找到真正的路径 (字符串且存在)
             found_path = None
             for item in file_obj:
                 if isinstance(item, str) and (os.path.exists(item) or os.path.isabs(item)):
                     found_path = item
                     break
             
             if found_path:
                 file_path = found_path
             else:
                 # 没找到就把 tuple 强转 string (虽然可能还是错的但能看出来)
                 file_path = str(file_obj)
                 # 如果 tuple 里有 string 但没 exists? 可能是第二个
                 if len(file_obj) >= 2 and isinstance(file_obj[1], str):
                     file_path = file_obj[1]
                 elif len(file_obj) >= 1 and isinstance(file_obj[0], str):
                      file_path = file_obj[0]

        print(f"\n=======================================================", flush=True)
        # 用 exists 判断一下减少报错刷屏
        if not os.path.exists(file_path):
             print(f"⚠️ [Error] 文件不存在或路径错误: {file_path}", flush=True)
             continue
             
        print(f"▶️ [{i+1}/{len(tasks)}] 正在处理: {os.path.basename(file_path)}", flush=True)
        print(f"=======================================================", flush=True)
        
        try:
            # 解析热词列表
            hotwords = [w.strip() for w in args.hotwords.split(",") if w.strip()] if args.hotwords else []
            
            engine.run(
                file_path, 
                args.output_dir, 
                language=args.language,
                enable_spk=args.enable_spk,
                batch_size=args.batch_size,
                hotwords=hotwords  # 🔥 传递热词
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