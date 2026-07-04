import os
import sys
import queue
import time
import numpy as np
import threading
import importlib.util
import torch
import argparse
import tkinter as tk
from tkinter import font as tkfont
import ctypes

# ================= 📦 依赖检查 =================
try:
    import onnxruntime
except ImportError:
    print("❌ 缺少依赖: 请运行 pip install onnxruntime 安装")
    sys.exit(1)
# ===============================================

# 0. 强制 UTF-8 输出
sys.stdout.reconfigure(encoding='utf-8')

# ================= ⚙️ 用户参数配置区 =================
VAD_START_THRESHOLD = 0.5  # 【启动门槛】只有超过 0.5 才认为开始说话 (抗噪)
VAD_END_THRESHOLD = 0.3    # 【维持门槛】只要不低于 0.3 就认为还在说话 (防断连)

PAUSE_LIMIT_SEC = 0.15      # 句尾停顿 0.15 秒算结束（恢复原值，减少碎片）
MAX_SENTENCE_SEC = 6.0    # 最长 6 秒强制切断（恢复原值）
MIN_SENTENCE_SEC = 0.5     # 最短有效长度
VOLUME_BOOST = 3.0         # 声音增益

# 🔥 VAD 模式选择: "silero" 或 "fsmn"
# ⚠️ FSMN VAD 不适合实时场景（设计用于有限长度音频流）
# ✅ Silero VAD 推荐用于实时字幕（逐帧判断，无限流友好）
VAD_MODE = "silero"

# Silero VAD 模型路径
VAD_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "models", "silero_vad.onnx")
# =======================================================

# 1. 控制台选择：保留文本选择能力，避免禁用快速编辑模式
ENABLE_CONSOLE_SELECTION = True
if not ENABLE_CONSOLE_SELECTION:
    def disable_quick_edit():
        if os.name == 'nt':
            try:
                kernel32 = ctypes.windll.kernel32
                hStdIn = kernel32.GetStdHandle(-10)
                mode = ctypes.c_ulong()
                kernel32.GetConsoleMode(hStdIn, ctypes.byref(mode))
                mode.value &= ~0x0040
                kernel32.SetConsoleMode(hStdIn, mode)
            except: pass
    disable_quick_edit()

# 2. DPI 感知
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
    print("✅ DPI Awareness Set")
except Exception as e:
    print(f"⚠️ DPI Set Failed: {e}")

# ================= 🔧 加载魔改版 Nano 模型 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 尝试加载提取出来的本地修改版 model.py
manual_model_path = os.path.join(CURRENT_DIR, "custom_nano_model.py")
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

try:
    import pyaudiowpatch as pyaudio
    HAS_LOOPBACK = True
except ImportError:
    import pyaudio
    HAS_LOOPBACK = False

from funasr import AutoModel

# ================= 🤖 VAD 封装类 (修复版) =================
class SileroVAD:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"VAD 模型未找到: {model_path}")
        
        opts = onnxruntime.SessionOptions()
        opts.log_severity_level = 3
        
        # 强制使用 CPU
        self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'], sess_options=opts)
        
        self.input_names = [x.name for x in self.session.get_inputs()]
        
        self.reset_states()
        print(f"✅ VAD Loaded: {os.path.basename(model_path)} (CPU Mode)")

    def reset_states(self):
        if 'state' in self.input_names:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def __call__(self, audio_chunk):
        input_tensor = audio_chunk[np.newaxis, :]
        ort_inputs = {
            'input': input_tensor,
            'sr': np.array([16000], dtype='int64'),
        }
        
        if 'state' in self.input_names:
            ort_inputs['state'] = self._state
            out, new_state = self.session.run(None, ort_inputs)
            self._state = new_state
        elif 'h' in self.input_names and 'c' in self.input_names:
            ort_inputs['h'] = self._h
            ort_inputs['c'] = self._c
            out, h, c = self.session.run(None, ort_inputs)
            self._h, self._c = h, c
        else:
            out = self.session.run(None, ort_inputs)[0]

        return out[0][0]

# ================= 🤖 FSMN VAD Streaming 封装类 (能量门控版) =================
class FsmnVadStreaming:
    """
    FunASR FSMN VAD 流式封装
    
    🔥 关键改进：使用能量门控，只在有声音时才调用 FSMN
    静音时发送 is_final=True 来刷新状态
    """
    # 能量门控阈值
    ENERGY_THRESHOLD = 0.001  # 低于此值视为静音
    SILENCE_CHUNKS_TO_FLUSH = 10  # 连续 10 个静音块后刷新 VAD
    
    def __init__(self, chunk_size_ms=200):
        print("📦 加载 FSMN VAD Streaming 模型 (能量门控版)...")
        self.model = AutoModel(
            model="fsmn-vad",
            model_revision="v2.0.4",
            device="cpu",
            disable_update=True,
            disable_pbar=True
        )
        self.chunk_size = chunk_size_ms
        self.cache = {}
        self.sample_rate = 16000
        self.chunk_stride = int(chunk_size_ms * self.sample_rate / 1000)
        
        # 状态跟踪
        self.accumulated_audio = []  # 累积的有声音音频
        self.silence_counter = 0     # 静音块计数
        self.is_speaking = False     # 是否正在说话
        self.session_start_ms = 0    # 当前会话开始时间
        self._debug_counter = 0
        self._last_audio = None      # 🔥 最后完成的音频段
        
        print(f"✅ FSMN VAD Loaded! Chunk: {chunk_size_ms}ms | Energy Threshold: {self.ENERGY_THRESHOLD}")
    
    def reset(self):
        """重置状态"""
        self.cache = {}
        self.accumulated_audio = []
        self.silence_counter = 0
        self.is_speaking = False
        self.session_start_ms = 0
    
    def process_chunk(self, audio_chunk):
        """
        处理一块音频
        返回格式: [(start_ms, end_ms), ...] 或 [] 如果没有完成的段落
        
        策略：
        1. 有声音时累积音频
        2. 静音超过阈值时，用 is_final=True 刷新 FSMN 获取段落
        """
        energy = np.abs(audio_chunk).mean()
        completed_segments = []
        
        self._debug_counter += 1
        if self._debug_counter % 30 == 0:  # 每秒输出一次
            status = "🎤 Speaking" if self.is_speaking else "🔇 Silent"
            print(f"\r{status} | energy={energy:.4f} | buf={len(self.accumulated_audio)} chunks", end="")
        
        if energy > self.ENERGY_THRESHOLD:
            # ===== 有声音 =====
            self.silence_counter = 0
            self.accumulated_audio.append(audio_chunk)
            
            if not self.is_speaking:
                self.is_speaking = True
                self.session_start_ms = 0  # 重置时间基准
                self.cache = {}  # 重置 FSMN 缓存
                print(f"\n🎤 检测到声音，开始录制...")
        
        else:
            # ===== 静音 =====
            self.silence_counter += 1
            
            # 如果之前在说话，且静音够久，就刷新 VAD
            if self.is_speaking and self.silence_counter >= self.SILENCE_CHUNKS_TO_FLUSH:
                if len(self.accumulated_audio) > 0:
                    print(f"\n⏹️ 静音检测，处理累积的 {len(self.accumulated_audio)} 块音频...")
                    
                    # 合并音频并保存
                    full_audio = np.concatenate(self.accumulated_audio)
                    self._last_audio = full_audio  # 🔥 保存音频供主循环使用
                    
                    duration_ms = len(full_audio) / self.sample_rate * 1000
                    if duration_ms > 300:  # 至少 300ms
                        completed_segments.append((0, int(duration_ms)))
                        print(f"✅ 语音段: [0s - {duration_ms/1000:.1f}s]")
                    
                    # 重置状态
                    self.accumulated_audio = []
                
                self.is_speaking = False
                self.cache = {}
        
        return completed_segments, self.accumulated_audio if self.is_speaking else None

# ================= 🖥️ 悬浮窗 UI =================
class SubtitleOverlay:
    def __init__(self, root, target_width=1800):
        self.root = root
        self.root.title("FunASR Overlay")
        
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        w = target_width
        h = 280  # 🔥 增加高度以容纳 4 行历史 + 当前行
        x = (screen_w - w) // 2
        y = screen_h - h - 150  # 稍微上移一点
        
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.88)  # 稍微更不透明
        self.root.configure(bg='#1a1a2e')  # 深蓝黑色背景，更高级
        
        # 🔥 字体优化：当前行更大，历史行更小但清晰
        self.font_main = tkfont.Font(family="Microsoft YaHei UI", size=24, weight="bold")
        self.font_sub = tkfont.Font(family="Microsoft YaHei UI", size=14)
        self.font_typing = tkfont.Font(family="Microsoft YaHei UI", size=20)  # 打字中的字体
        
        self.text_box = tk.Text(root, bg='#1a1a2e', fg='white', font=self.font_main,
                               bd=0, highlightthickness=0, wrap="word", cursor="arrow",
                               padx=15, pady=8)  # 内边距
        self.text_box.pack(expand=True, fill='both', padx=25, pady=15)
        
        # 🔥 左对齐样式
        self.text_box.tag_config("hist1", foreground="#666666", font=self.font_sub, justify='left')  # 最旧，最淡
        self.text_box.tag_config("hist2", foreground="#888888", font=self.font_sub, justify='left')
        self.text_box.tag_config("hist3", foreground="#aaaaaa", font=self.font_sub, justify='left')
        self.text_box.tag_config("hist4", foreground="#cccccc", font=self.font_sub, justify='left')  # 最新历史，较亮
        self.text_box.tag_config("curr", foreground="#00ff88", font=self.font_main, justify='left')  # 当前行：亮绿色
        self.text_box.tag_config("typing", foreground="#ffcc00", font=self.font_typing, justify='left')  # 打字中：金黄色
        
        self.update_content([], "🚀 引擎启动中...")
        self.root.bind("<Button-1>", self.start_move)
        self.root.bind("<B1-Motion>", self.do_move)
        self.root.bind("<Double-Button-1>", lambda e: sys.exit(0))

    def start_move(self, event):
        self.x = event.x
        self.y = event.y

    def do_move(self, event):
        x = self.root.winfo_x() + (event.x - self.x)
        y = self.root.winfo_y() + (event.y - self.y)
        self.root.geometry(f"+{x}+{y}")
        
    def update_content(self, history, current, is_typing=False):
        """更新字幕内容，支持 4 行历史 + 渐变淡出效果"""
        self.text_box.config(state="normal")
        self.text_box.delete("1.0", "end")
        
        # 🔥 显示最近 4 行历史，从最旧到最新渐变
        hist_tags = ["hist1", "hist2", "hist3", "hist4"]  # 从暗到亮
        recent_history = history[-4:]  # 取最近 4 条
        
        # 根据实际条数分配 tag（越新越亮）
        for i, line in enumerate(recent_history):
            if line:
                # 计算这条记录应该用哪个亮度 tag
                tag_index = len(recent_history) - len(recent_history) + i
                tag_index = max(0, 4 - len(recent_history) + i)  # 确保最新的用 hist4
                tag = hist_tags[min(tag_index, 3)]
                self.text_box.insert("end", "  " + line + "\n", tag)
        
        # 🔥 当前行：根据是否正在打字用不同样式
        if current:
            prefix = "▶ " if is_typing else "● "
            tag = "typing" if is_typing else "curr"
            self.text_box.insert("end", prefix + current, tag)
        
        self.text_box.config(state="disabled")

# ================= 🎧 音频流基类 =================

# class BaseStream:
#     def __init__(self):
#         self.p = pyaudio.PyAudio()
#         self.q = queue.Queue() # 无限队列，防止阻塞
#         self.rate = 16000
#         self.target_chunk_size = 512
#     # 回调函数：由声卡驱动在后台线程自动调用
#     def callback(self, in_data, frame_count, time_info, status):
#         self.q.put(in_data) # 只管往里塞，耗时极短
#         return (None, pyaudio.paContinue)
        
#     # 读取函数：现在只是从队列里拿，不会阻塞硬件录音
#     def read(self):
#         # 如果主线程卡太久，队列里积压了太多数据，我们直接把积压的数据全部拿出来拼在一起？
#         # 或者为了实时性，丢弃旧数据？(不，转写不能丢数据)
#         # 现在的 read 只拿一个 chunk，但如果队列里有很多，说明我们滞后了
#         return self.process_data(self.q.get(), self.src_channels, self.src_rate)
    
#     # 新增一个方法：检查是否积压
#     def get_lag(self):
#         return self.q.qsize()
    
#     def process_data(self, data, src_channels, src_rate):
#         audio = np.frombuffer(data, dtype=np.int16)
#         audio = audio.astype(np.float32) / 32768.0
        
#         # 多声道转单声道
#         if src_channels > 1: 
#             audio = audio.reshape(-1, src_channels).mean(axis=1)
            
#         # 重采样
#         if src_rate != 16000:
#             # 这里的 len(audio) 已经是根据 src_rate 算好的 32ms 数据量
#             # 所以直接重采样到目标点数即可
#             num_samples = int(len(audio) * 16000 / src_rate)
#             indices = np.linspace(0, len(audio)-1, num_samples)
#             audio = audio[indices.astype(int)]
            
#         # 🔊 声音增益：防止内录声音太小
#         audio = audio * VOLUME_BOOST
        
#         # 裁剪到 -1.0 ~ 1.0 防止爆音
#         audio = np.clip(audio, -1.0, 1.0)
        
#         return audio

# class LoopbackStream(BaseStream):
#     def start(self):
#         if not HAS_LOOPBACK: raise Exception("No pyaudiowpatch library")
#         print("🔍 正在寻找 Loopback 设备...")
#         target = None
#         try:
#             # wasapi = next(i for i in range(self.p.get_host_api_count()) if self.p.get_host_api_info_by_index(i)["type"] == pyaudio.paWASAPI)
#             # default_out = self.p.get_device_info_by_index(wasapi["defaultOutputDevice"])
#             wasapi_index = next(i for i in range(self.p.get_host_api_count()) if self.p.get_host_api_info_by_index(i)["type"] == pyaudio.paWASAPI)
#             wasapi_info = self.p.get_host_api_info_by_index(wasapi_index) # ✅ 先获取详情字典
#             default_out = self.p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
#             print(f"🔈 系统默认输出: {default_out['name']}")
            
#             # 打印所有 Loopback 供调试
#             print("📋 可用 Loopback 设备:")
#             for loopback in self.p.get_loopback_device_info_generator():
#                 print(f"   - [{loopback['index']}] {loopback['name']}")
#                 if default_out["name"] in loopback["name"]:
#                     target = loopback
#         except Exception as e: print(f"⚠️ 自动匹配逻辑异常: {e}")

#         if not target:
#             print("⚠️ 未找到精确匹配，尝试使用首个可用 Loopback...")
#             loopbacks = list(self.p.get_loopback_device_info_generator())
#             if loopbacks: target = loopbacks[0]
#             else: raise Exception("系统未发现任何 Loopback 设备！")
            
#         print(f"🎤 [Loopback] 锁定: {target['name']} (ID: {target['index']})")
#         self.src_rate = int(target["defaultSampleRate"])
#         self.src_channels = target["maxInputChannels"]
        
#         # 🔥 关键修复：计算源采样率下，对应 32ms 需要多少个点
#         # 目标: 16000Hz 下 512 个点 (0.032s)
#         # 源头: src_rate 下 x 个点 -> x = src_rate * 0.032
#         chunk_src = int(self.src_rate * (self.target_chunk_size / 16000))
        
#         self.stream = self.p.open(
#             format=pyaudio.paInt16, channels=self.src_channels, rate=self.src_rate,
#             input=True, input_device_index=target["index"],
#             frames_per_buffer=chunk_src, stream_callback=self.callback
#         )
#         return self

#     def read(self):
#         return self.process_data(self.q.get(), self.src_channels, self.src_rate)

#     def stop(self):
#         try: self.stream.stop_stream(); self.stream.close(); self.p.terminate()
#         except: pass

# class MicStream(BaseStream):
#     def start(self):
#         print("🔍 正在寻找麦克风...")
#         target = None
#         try: target = self.p.get_default_input_device_info()
#         except: pass
        
#         if not target:
#             for i in range(self.p.get_device_count()):
#                 info = self.p.get_device_info_by_index(i)
#                 if info['maxInputChannels'] > 0: target = info; break
        
#         if not target: raise Exception("未找到任何输入设备")
            
#         print(f"🎤 [Mic] 锁定: {target['name']} (ID: {target['index']})")
#         self.src_rate = int(target["defaultSampleRate"])
#         self.src_channels = target["maxInputChannels"]
        
#         chunk_src = int(self.src_rate * (self.target_chunk_size / 16000))
        
#         self.stream = self.p.open(
#             format=pyaudio.paInt16, channels=self.src_channels, rate=self.src_rate,
#             input=True, input_device_index=target['index'],
#             frames_per_buffer=chunk_src, stream_callback=self.callback
#         )
#         return self

#     def read(self):
#         return self.process_data(self.q.get(), self.src_channels, self.src_rate)

#     def stop(self):
#         try: self.stream.stop_stream(); self.stream.close(); self.p.terminate()
#         except: pass

# class MixedStream:
#     def __init__(self):
#         self.mic = MicStream()
#         self.loop = LoopbackStream()
#     def start(self):
#         print("🔀 启动混合模式 (Mix Mode)...")
#         loop_ok = False
#         try: self.loop.start(); loop_ok = True
#         except Exception as e: print(f"⚠️ 内录启动失败: {e}"); self.loop = None
#         try: self.mic.start()
#         except Exception as e:
#             print(f"⚠️ 麦克风启动失败: {e}")
#             if not loop_ok: raise Exception("内录和麦克风全都启动失败！")
#         return self
#     def get_queue_size(self):
#         s1 = self.mic.q.qsize() if hasattr(self.mic, 'q') else 0
#         s2 = self.loop.q.qsize() if self.loop and hasattr(self.loop, 'q') else 0
#         return max(s1, s2)
#     def read(self):
#         chunk_mic = None
#         try: chunk_mic = self.mic.read()
#         except: pass
#         chunk_loop = None
#         if self.loop:
#             try:
#                 raw_loop = self.loop.q.get(timeout=0.05) 
#                 chunk_loop = self.loop.process_data(raw_loop, self.loop.src_channels, self.loop.src_rate)
#             except queue.Empty: pass
#         if chunk_mic is not None and chunk_loop is not None:
#             min_len = min(len(chunk_mic), len(chunk_loop))
#             return (chunk_mic[:min_len] + chunk_loop[:min_len]) / 2
#         elif chunk_mic is not None: return chunk_mic
#         elif chunk_loop is not None: return chunk_loop
#         else: return np.zeros(512, dtype=np.float32) 
#     def stop(self):
#         if self.loop: self.loop.stop()
#         self.mic.stop()
# ================= 🎧 音频流基类 (完整修正版) =================
class BaseStream:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.q = queue.Queue() # 无限队列，防止阻塞
        self.rate = 16000
        # 🔥 修复点 1: 必须定义这个变量，否则 start() 会报错
        self.target_chunk_size = 512 
        # 默认值，防止未启动时调用报错
        self.src_channels = 1
        self.src_rate = 16000
    
    # 回调函数：由声卡驱动在后台线程自动调用
    def callback(self, in_data, frame_count, time_info, status):
        self.q.put(in_data) # 只管往里塞，耗时极短
        return (None, pyaudio.paContinue)
        
    # 读取函数：现在只是从队列里拿，不会阻塞硬件录音
    def read(self):
        # 如果队列为空，这里会阻塞等待，但这没关系，因为我们在主循环里检查了 get_queue_size
        return self.process_data(self.q.get(), self.src_channels, self.src_rate)
    
    # 🔥 修复点 2: 必须定义这个方法，否则主循环报错
    def get_queue_size(self):
        return self.q.qsize()
    
    def process_data(self, data, src_channels, src_rate):
        audio = np.frombuffer(data, dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0
        
        # 多声道转单声道
        if src_channels > 1: 
            audio = audio.reshape(-1, src_channels).mean(axis=1)
            
        # 重采样
        if src_rate != 16000:
            num_samples = int(len(audio) * 16000 / src_rate)
            indices = np.linspace(0, len(audio)-1, num_samples)
            audio = audio[indices.astype(int)]
            
        audio = audio * VOLUME_BOOST
        audio = np.clip(audio, -1.0, 1.0)
        return audio

class LoopbackStream(BaseStream):
    def start(self):
        if not HAS_LOOPBACK: raise Exception("No pyaudiowpatch library")
        print("🔍 正在寻找 Loopback 设备...")
        target = None
        try:
            # 🔥 修复点 3: 修复 wasapi 寻找逻辑 (解决截图黄字报错)
            wasapi_index = next(i for i in range(self.p.get_host_api_count()) if self.p.get_host_api_info_by_index(i)["type"] == pyaudio.paWASAPI)
            wasapi_info = self.p.get_host_api_info_by_index(wasapi_index)
            default_out = self.p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
            
            print(f"🔈 系统默认输出: {default_out['name']}")
            
            # 打印所有 Loopback 供调试
            print("📋 可用 Loopback 设备:")
            for loopback in self.p.get_loopback_device_info_generator():
                print(f"   - [{loopback['index']}] {loopback['name']}")
                if default_out["name"] in loopback["name"]:
                    target = loopback
        except Exception as e: print(f"⚠️ 自动匹配逻辑异常: {e}")

        if not target:
            print("⚠️ 未找到精确匹配，尝试使用首个可用 Loopback...")
            loopbacks = list(self.p.get_loopback_device_info_generator())
            if loopbacks: target = loopbacks[0]
            else: raise Exception("系统未发现任何 Loopback 设备！")
            
        print(f"🎤 [Loopback] 锁定: {target['name']} (ID: {target['index']})")
        self.src_rate = int(target["defaultSampleRate"])
        self.src_channels = target["maxInputChannels"]
        
        chunk_src = int(self.src_rate * (self.target_chunk_size / 16000))
        
        self.stream = self.p.open(
            format=pyaudio.paInt16, channels=self.src_channels, rate=self.src_rate,
            input=True, input_device_index=target["index"],
            frames_per_buffer=chunk_src, stream_callback=self.callback
        )
        return self

    def stop(self):
        try: self.stream.stop_stream(); self.stream.close(); self.p.terminate()
        except: pass

class MicStream(BaseStream):
    def start(self):
        print("🔍 正在寻找麦克风...")
        target = None
        try: target = self.p.get_default_input_device_info()
        except: pass
        
        if not target:
            for i in range(self.p.get_device_count()):
                info = self.p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0: target = info; break
        
        if not target: raise Exception("未找到任何输入设备")
            
        print(f"🎤 [Mic] 锁定: {target['name']} (ID: {target['index']})")
        self.src_rate = int(target["defaultSampleRate"])
        self.src_channels = target["maxInputChannels"]
        
        chunk_src = int(self.src_rate * (self.target_chunk_size / 16000))
        
        self.stream = self.p.open(
            format=pyaudio.paInt16, channels=self.src_channels, rate=self.src_rate,
            input=True, input_device_index=target['index'],
            frames_per_buffer=chunk_src, stream_callback=self.callback
        )
        return self

    def stop(self):
        try: self.stream.stop_stream(); self.stream.close(); self.p.terminate()
        except: pass

class MixedStream:
    def __init__(self):
        self.mic = MicStream()
        self.loop = LoopbackStream()
    def start(self):
        print("🔀 启动混合模式 (Mix Mode)...")
        loop_ok = False
        try: self.loop.start(); loop_ok = True
        except Exception as e: print(f"⚠️ 内录启动失败: {e}"); self.loop = None
        try: self.mic.start()
        except Exception as e:
            print(f"⚠️ 麦克风启动失败: {e}")
            if not loop_ok: raise Exception("内录和麦克风全都启动失败！")
        return self
        
    # 🔥 修复点 2: 混合模式也需要这个方法
    def get_queue_size(self):
        s1 = self.mic.q.qsize() if hasattr(self.mic, 'q') else 0
        s2 = self.loop.q.qsize() if self.loop and hasattr(self.loop, 'q') else 0
        return max(s1, s2)
        
    def read(self):
        chunk_mic = None
        try: chunk_mic = self.mic.read()
        except: pass
        chunk_loop = None
        if self.loop:
            try:
                # 尝试非阻塞读取，如果没有数据就算了
                raw_loop = self.loop.q.get(timeout=0.005) 
                chunk_loop = self.loop.process_data(raw_loop, self.loop.src_channels, self.loop.src_rate)
            except queue.Empty: pass
        if chunk_mic is not None and chunk_loop is not None:
            min_len = min(len(chunk_mic), len(chunk_loop))
            return (chunk_mic[:min_len] + chunk_loop[:min_len]) / 2
        elif chunk_mic is not None: return chunk_mic
        elif chunk_loop is not None: return chunk_loop
        else: return np.zeros(512, dtype=np.float32) 
    def stop(self):
        if self.loop: self.loop.stop()
        self.mic.stop()


# ================= 🧠 主逻辑 =================
def run_engine(model_dir, lang, device_mode, gui_queue):
    if gui_queue: gui_queue.put("⏳ 正在加载 AI 模型...")
    print(f"🚀 Model: {os.path.basename(model_dir)} | Mode: {device_mode}")
    
    # 初始化 VAD（根据配置选择模式）
    use_fsmn = VAD_MODE.lower() == "fsmn"
    try:
        if use_fsmn:
            vad_model = FsmnVadStreaming(chunk_size_ms=200)
            print(f"📡 使用 FSMN VAD Streaming 模式")
        else:
            vad_model = SileroVAD(VAD_MODEL_PATH)
            print(f"📡 使用 Silero VAD 模式")
    except Exception as e:
        print(f"❌ VAD Init Failed: {e}")
        if gui_queue: gui_queue.put(f"❌ VAD加载失败: {e}")
        return

    # 初始化 ASR
    try:
        model = AutoModel(model=model_dir, trust_remote_code=True, device="cuda", 
                         disable_update=True, disable_pbar=True)
    except Exception as e:
        print(f"❌ ASR Model Init Failed: {e}")
        if gui_queue: gui_queue.put(f"❌ ASR模型加载失败: {e}")
        return

    if device_mode == "loopback": stream = LoopbackStream()
    elif device_mode == "mic": stream = MicStream()
    else: stream = MixedStream()

    try:
        stream.start()
    except Exception as e:
        print(f"❌ Audio Init Failed: {e}")
        if gui_queue: gui_queue.put(f"❌ 音频初始化失败: {e}")
        return
    
    buffer = []
    silence_cnt = 0
    is_speaking = False
    history = []
    
    CHUNK_DURATION = 0.032
    pause_limit_count = int(PAUSE_LIMIT_SEC / CHUNK_DURATION)
    min_len_points = int(16000 * MIN_SENTENCE_SEC)
    
    # print("✅ Engine Ready")
    # if gui_queue: gui_queue.put(f"✨ 就绪 | 模式: {device_mode}")
    
    # # 调试计数器
    # debug_tick = 0
    
    # while True:
    #     try:
    #         chunk = stream.read()
            
    #         if len(chunk) != 512:
    #             # 理论上现在不需要了，但为了保险还是留着
    #             if len(chunk) > 512: chunk = chunk[:512]
    #             else: chunk = np.pad(chunk, (0, 512 - len(chunk)))
    #         speech_prob = vad_model(chunk)
    #         active_threshold = VAD_END_THRESHOLD if is_speaking else VAD_START_THRESHOLD

    #         # 调试显示
    #         debug_tick += 1
    #         if debug_tick % 10 == 0:
    #             energy = np.abs(chunk).mean()
    #             # 状态图标：🔴=正在录入(哪怕概率稍低)  ⚪=高能预警(超过启动阈值)  🔇=静音
    #             if is_speaking:
    #                 status_icon = "🔴" # 录制中
    #             else:
    #                 status_icon = "⚪" if speech_prob > VAD_START_THRESHOLD else "🔇"
                
    #             # 打印当前生效的阈值
    #             # sys.stdout.write(f"\r{status_icon} Prob:{speech_prob:.2f} (>{active_threshold}) | Vol:{energy:.4f}   ")
    #             sys.stdout.flush()

    #         # 判断逻辑
    #         if speech_prob > active_threshold:
    #             is_speaking = True
    #             silence_cnt = 0
    #             buffer.append(chunk)

    #             if is_speaking:

    #                 duration = len(buffer) * 0.05
    #                 # bar = "▂▃▅▆▇" [min(int(energy * 1000), 4)]
    #                 sys.stdout.write(f"\r[{duration:.1f}s] ")
    #                 sys.stdout.flush()
    #             # 超时强制截断逻辑
    #             if len(buffer) * CHUNK_DURATION > MAX_SENTENCE_SEC:
    #                 sys.stdout.write("\n⚠️ [超时强制截断]\n")
    #                 sentence = np.concatenate(buffer)
    #                 if len(sentence) > min_len_points:
    #                     # ... (ASR 推理代码不变) ...
    #                     res = model.generate(input=[torch.from_numpy(sentence)], language=lang, use_itn=True, batch_size_s=0)
    #                     if res and res[0]['text'].strip(): 
    #                         txt = res[0]['text'].strip()
    #                         print(f"\n📝 {txt}")
    #                         history.append(txt)
    #                         if len(history) > 2: history.pop(0)
    #                         if gui_queue: gui_queue.put({"hist": history[:-1], "curr": history[-1]})
    #                 buffer = []; is_speaking = False; vad_model.reset_states()
    #         else:
    #             if is_speaking:
    #                 # 只有在说话状态下，且概率低于维持门槛 (0.3) 时，才开始计数停顿
    #                 silence_cnt += 1
    #                 buffer.append(chunk)
                    
    #                 if silence_cnt > pause_limit_count:
    #                     # 停顿超时，进行识别
    #                     sentence = np.concatenate(buffer)
    #                     if len(sentence) > min_len_points:
    #                         # ... (ASR 推理代码不变) ...
    #                         res = model.generate(input=[torch.from_numpy(sentence)], language=lang, use_itn=True, batch_size_s=0)
    #                         if res and res[0]['text'].strip(): 
    #                             txt = res[0]['text'].strip()
    #                             print(f"\n📝 {txt}")
    #                             history.append(txt)
    #                             if len(history) > 2: history.pop(0)
    #                             if gui_queue: gui_queue.put({"hist": history[:-1], "curr": history[-1]})
    #                     buffer = []; is_speaking = False; silence_cnt = 0; vad_model.reset_states()
    #     except Exception as e:
    #         print(f"\n❌ Stream Loop Error: {e}")
    #         break
    # ... (前面的初始化代码保持不变) ...
    
# ================= 🚀 新版智能循环 (支持双 VAD 模式) =================
    print(f"✅ Engine Ready | VAD: {VAD_MODE}")
    if gui_queue: gui_queue.put(f"✨ 就绪 | VAD: {VAD_MODE} | 模式: {device_mode}")
    
    # 状态变量
    last_preview_time = 0
    PREVIEW_MIN_INTERVAL = 0.3
    
    # FSMN VAD 专用变量
    fsmn_audio_buffer = []  # 累积音频用于 FSMN
    fsmn_temp_buffer = []   # 攒够 200ms 的临时缓冲
    last_segment_end = 0    # 上次处理的段落结束位置
    
    while True:
        try:
            # --- 1. 智能积压检测 ---
            lag_count = stream.get_queue_size()
            is_lagging = lag_count > 5
            
            if is_lagging and lag_count % 10 == 0:
                status_line = f"\r🚀 追赶进度中... (积压: {lag_count})"
                sys.stderr.write(status_line)
                sys.stderr.flush()

            # --- 2. 读取音频 ---
            chunk = stream.read()
            
            # 补齐或截断到 512 点
            if len(chunk) != 512:
                if len(chunk) > 512: chunk = chunk[:512]
                else: chunk = np.pad(chunk, (0, 512 - len(chunk)))
            
            # ========================================
            # 🔀 根据 VAD 模式分支处理
            # ========================================
            
            if use_fsmn:
                # ========== FSMN VAD 模式 (能量门控) ==========
                # 新接口返回 (completed_segments, accumulated_audio)
                completed_segments, accumulated_audio = vad_model.process_chunk(chunk)
                
                # 处理完成的语音段
                for start_ms, end_ms in completed_segments:
                    # accumulated_audio 已经在 VAD 内部被清空
                    # 需要从 VAD 获取完整音频
                    pass  # 此处不再需要，因为音频已在 VAD 内部处理
                
                # 如果检测到语音结束，从 VAD 获取累积的音频并识别
                if completed_segments:
                    # 获取之前累积的音频进行 ASR
                    for start_ms, end_ms in completed_segments:
                        # 使用 VAD 返回的时间信息，音频从 VAD 内部获取
                        # 由于我们用能量门控，实际音频在 vad_model 内部
                        # 需要在检测到段落结束前保存音频
                        pass
                
                # 🔥 新逻辑：直接使用 VAD 累积的音频
                if completed_segments and hasattr(vad_model, '_last_audio'):
                    segment_audio = vad_model._last_audio
                    if segment_audio is not None and len(segment_audio) > int(16000 * MIN_SENTENCE_SEC):
                        try:
                            res = model.generate(
                                input=[torch.from_numpy(segment_audio)],
                                language=lang, use_itn=True, batch_size_s=0
                            )
                            if res and res[0]['text'].strip():
                                txt = res[0]['text'].strip()
                                duration = len(segment_audio) / 16000
                                print(f"\n📝 [{duration:.1f}s] {txt}")
                                history.append(txt)
                                if len(history) > 4: history.pop(0)
                                if gui_queue:
                                    gui_queue.put({"hist": history[:-1], "curr": history[-1]})
                        except Exception as e:
                            print(f"⚠️ ASR Error: {e}")
            
            else:
                # ========== Silero VAD 模式 (原有逻辑) ==========
                speech_prob = vad_model(chunk)
                active_threshold = VAD_END_THRESHOLD if is_speaking else VAD_START_THRESHOLD

                if speech_prob > active_threshold:
                    is_speaking = True
                    silence_cnt = 0
                    buffer.append(chunk)

                    # 实时预览
                    current_time = time.time()
                    if not is_lagging and \
                       (current_time - last_preview_time > PREVIEW_MIN_INTERVAL) and \
                       len(buffer) > 5:
                        last_preview_time = current_time
                        try:
                            temp_sentence = np.concatenate(buffer)
                            res = model.generate(input=[torch.from_numpy(temp_sentence)], 
                                               language=lang, use_itn=True, batch_size_s=0)
                            if res and res[0]['text'].strip():
                                preview_text = res[0]['text'].strip() + " 🟢"
                                if gui_queue: 
                                    gui_queue.put({"hist": history, "curr": preview_text, "typing": True})
                        except Exception: pass

                    # 超时强制截断
                    if len(buffer) * CHUNK_DURATION > MAX_SENTENCE_SEC:
                        sys.stdout.write("\n⚠️ [超时强制截断]\n")
                        sentence = np.concatenate(buffer)
                        if len(sentence) > min_len_points:
                            res = model.generate(input=[torch.from_numpy(sentence)], language=lang, use_itn=True, batch_size_s=0)
                            if res and res[0]['text'].strip(): 
                                txt = res[0]['text'].strip()
                                print(f"\n📝 {txt}")
                                history.append(txt)
                                if len(history) > 4: history.pop(0)
                                if gui_queue: gui_queue.put({"hist": history[:-1], "curr": history[-1]})
                        buffer = []; is_speaking = False; vad_model.reset_states()

                else:
                    if is_speaking:
                        silence_cnt += 1
                        buffer.append(chunk)
                        
                        if silence_cnt > pause_limit_count:
                            sentence = np.concatenate(buffer)
                            if len(sentence) > min_len_points:
                                res = model.generate(input=[torch.from_numpy(sentence)], language=lang, use_itn=True, batch_size_s=0)
                                if res and res[0]['text'].strip(): 
                                    txt = res[0]['text'].strip()
                                    print(f"\n📝 定稿: {txt}")
                                    history.append(txt)
                                    if len(history) > 4: history.pop(0)
                                    if gui_queue: gui_queue.put({"hist": history[:-1], "curr": history[-1]})
                            
                            buffer = []; is_speaking = False; silence_cnt = 0; vad_model.reset_states()

        except Exception as e:
            print(f"\n❌ Loop Error: {e}")
            import traceback
            traceback.print_exc()
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "models", "iic", "SenseVoiceSmall"))
    parser.add_argument("--language", default="auto")
    parser.add_argument("--device_mode", default="loopback") 
    parser.add_argument("--overlay", action="store_true")
    args = parser.parse_args()
    
    q = queue.Queue() if args.overlay else None
    
    t = threading.Thread(target=run_engine, args=(args.model_dir, args.language, args.device_mode, q), daemon=True)
    t.start()
    
    if args.overlay:
        print("🖥️ Starting GUI...")
        root = tk.Tk()
        app = SubtitleOverlay(root, target_width=1800)
        
        def update_ui():
            try:
                while True:
                    data = q.get_nowait()
                    if isinstance(data, str):
                        app.update_content([], data)
                    elif isinstance(data, dict):
                        is_typing = data.get("typing", False)
                        app.update_content(data["hist"], data["curr"], is_typing)
            except queue.Empty: pass
            root.after(50, update_ui)  # 🔥 更快的 UI 刷新
        
        root.after(100, update_ui)
        root.lift()
        root.attributes('-topmost',True)
        root.mainloop()
    else:
        print("🖥️ Headless Mode (CLI only)")
        while True: time.sleep(1)