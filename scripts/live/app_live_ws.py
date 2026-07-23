import os
import sys
import time
import numpy as np
import threading
import argparse
import tkinter as tk
from tkinter import font as tkfont
import ctypes
import json
import asyncio
import websockets
import wave
from typing import List

from funclip_pro.core.audio import LoopbackStream, MicStream, MixedStream

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# DPI Awareness
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
    # 🔥 开启 Windows 虚拟终端支持 (ANSI Escape Codes)
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
except: pass

# --- Configuration ---
SAMPLE_RATE = 16000
CHUNK_MS = 500  # 🔥 与服务端 chunk_size_sec=0.5 对齐，减少通信开销
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)
SERVER_WS_URL = "ws://127.0.0.1:28000/ws/asr"

# ================= 🧠 智能分段器 =================
class SubtitleSegmenter:
    """将 Qwen3 的累积文本智能切分为句子，用于分行显示"""
    STRONG_ENDINGS = {'。', '！', '？', '…', '.', '!', '?', '；', '\n'}
    WEAK_ENDINGS = {'，', ','}  # 仅在超过 MAX_LINE_CHARS 时触发切分
    MAX_LINE_CHARS = 25
    RESET_CHAR_LIMIT = 200
    RESET_SEC_LIMIT = 30.0
    HIGHLIGHT_SEC = 1.5

    def __init__(self):
        self.history: List[str] = []
        self.session_char_count = 0
        self.session_duration = 0.0
        self.prev_history_count = 0 
        self.highlight_until = 0.0

    def process(self, full_text: str, duration: float = 0.0):
        """
        返回 (new_finalized_sentences, current_tail, highlight_last, need_reset)
        """
        now = time.time()
        self.session_duration = duration
        
        if not full_text:
            return [], "", now < self.highlight_until, False

        # 1. 动态标点切分：强标点直接切，弱标点(逗号)仅在超25字时切
        sentences = []
        current = ""
        # reset_tail: 仅按强标点判定，用于重置决策（不受逗号影响）
        reset_current = ""
        
        for i, char in enumerate(full_text):
            current += char
            reset_current += char
            
            is_strong = char in self.STRONG_ENDINGS
            
            # 🔥 特殊处理：防止 0.2 这种小数点被误认为英文句号
            if char == '.' and i > 0 and i < len(full_text) - 1:
                if full_text[i-1].isdigit() and full_text[i+1].isdigit():
                    is_strong = False
            
            if is_strong:
                s = current.strip()
                if s: sentences.append(s)
                current = ""
                reset_current = ""
            elif char in self.WEAK_ENDINGS and len(current) >= self.MAX_LINE_CHARS:
                # 逗号切分：仅影响显示，不清空 reset_current
                s = current.strip()
                if s: sentences.append(s)
                current = ""
        
        tail = current.strip()           # 显示用的尾巴（受逗号切分影响）
        reset_current.strip()  # 重置判断用的尾巴（仅受强标点影响）

        # 2. 决定当前大字幕和定稿历史
        if tail:
            # 有未完成的尾巴 → 所有完整句子进历史
            display_history = sentences
            display_current = tail
        elif sentences:
            # 没有尾巴（句号结尾）→ 最后一句可能还没说完，不往历史里放
            display_history = sentences[:-1]
            display_current = sentences[-1]
        else:
            display_history = []
            display_current = ""

        # 3. 找出本轮新定稿的句子
        new_finalized = []
        if len(display_history) > self.prev_history_count:
            new_finalized = display_history[self.prev_history_count:]
            self.highlight_until = now + self.HIGHLIGHT_SEC
            self.prev_history_count = len(display_history)
            
            for s in new_finalized:
                self.history.append(s)
                if len(self.history) > 20:
                    self.history.pop(0)

        highlight_last = now < self.highlight_until
        
        # 4. 重置检测（基于 Silero VAD 静音检测）
        self.session_char_count = len(full_text)
        threshold_reached = (self.session_char_count > self.RESET_CHAR_LIMIT or 
                            self.session_duration > self.RESET_SEC_LIMIT)
        
        # 🔥 优雅重置：不再看句号，改为由外部 VAD 静音信号控制
        # need_reset 默认 False，由 ASRClient 根据 VAD 静音状态来决定
        graceful_reset = False  # 由外部 VAD 覆盖
        
        # 硬性兜底：如果超过 45 秒仍未优雅重置，强制执行
        HARD_RESET_SEC = 45.0
        hard_reset = self.session_duration > HARD_RESET_SEC
        
        need_reset = graceful_reset or hard_reset

        return new_finalized, display_current, highlight_last, need_reset, threshold_reached

    def reset(self):
        self.session_char_count = 0
        self.session_duration = 0.0
        self.prev_history_count = 0
        self.highlight_until = 0.0
        self.history = []  # 🔥 必须清空历史，否则 UI 会显示重复内容


# ================= 🖥️ 可拖拽+缩放的悬浮字幕窗 =================
class SubtitleOverlay:
    EDGE_SIZE = 8  # 边缘拖拽区域像素
    MIN_W, MIN_H = 400, 120

    def __init__(self, root, target_width=1800):
        self.root = root
        self.root.title("Qwen3-ASR Live")

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        w = min(target_width, screen_w - 100)
        h = 220
        x = (screen_w - w) // 2
        y = screen_h - h - 120

        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.88)
        self.root.configure(bg='#1a1a2e')

        # 字体
        self.font_main = tkfont.Font(family="Microsoft YaHei UI", size=22, weight="bold")
        self.font_sub = tkfont.Font(family="Microsoft YaHei UI", size=13)
        self.font_highlight = tkfont.Font(family="Microsoft YaHei UI", size=18, weight="bold")

        # Text widget + Scrollbar
        self.container = tk.Frame(root, bg='#1a1a2e')
        self.container.pack(expand=True, fill='both', padx=10, pady=5)

        self.scrollbar = tk.Scrollbar(self.container)
        self.scrollbar.pack(side="right", fill="y")

        self.text_box = tk.Text(self.container, bg='#1a1a2e', fg='white', font=self.font_main,
                               bd=0, highlightthickness=0, wrap="word", cursor="arrow",
                               padx=10, pady=5, yscrollcommand=self.scrollbar.set)
        self.text_box.pack(side="left", expand=True, fill='both')
        self.scrollbar.config(command=self.text_box.yview)

        # 样式 tag
        self.text_box.tag_config("hist", foreground="#999999", font=self.font_sub, justify='left')
        self.text_box.tag_config("hist_hl", foreground="#ffffff", font=self.font_highlight, justify='left')
        self.text_box.tag_config("curr", foreground="#00ff88", font=self.font_main, justify='left')
        
        self.text_box.tag_raise("curr") # 确保当前行始终最亮
        self.auto_scroll = True

        self.update_content([], "🚀 Qwen3-ASR 就绪...")

        # --- 拖拽 & 缩放 & 滚动 ---
        self._drag_data = {"action": None, "x": 0, "y": 0}
        self.root.bind("<Button-1>", self._on_press)
        self.root.bind("<B1-Motion>", self._on_drag)
        self.root.bind("<Motion>", self._on_hover)
        self.root.bind("<MouseWheel>", self._on_mousewheel)
        self.root.bind("<Double-Button-1>", lambda e: self._close_app())

    def _get_edge(self, event):
        """判断鼠标在窗口的哪个边缘"""
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x, y = event.x, event.y
        e = self.EDGE_SIZE

        on_left = x < e
        on_right = x > w - e
        on_top = y < e
        on_bottom = y > h - e

        if on_bottom and on_right: return "br"
        if on_bottom and on_left: return "bl"
        if on_top and on_right: return "tr"
        if on_top and on_left: return "tl"
        if on_right: return "r"
        if on_left: return "l"
        if on_bottom: return "b"
        if on_top: return "t"
        return "move"

    def _on_hover(self, event):
        edge = self._get_edge(event)
        cursor_map = {
            "br": "size_nw_se", "tl": "size_nw_se",
            "bl": "size_ne_sw", "tr": "size_ne_sw",
            "r": "size_we", "l": "size_we",
            "b": "size_ns", "t": "size_ns",
            "move": "fleur"
        }
        self.root.config(cursor=cursor_map.get(edge, "arrow"))

    def _on_press(self, event):
        self._drag_data["action"] = self._get_edge(event)
        self._drag_data["x"] = event.x_root
        self._drag_data["y"] = event.y_root
        self._drag_data["win_x"] = self.root.winfo_x()
        self._drag_data["win_y"] = self.root.winfo_y()
        self._drag_data["win_w"] = self.root.winfo_width()
        self._drag_data["win_h"] = self.root.winfo_height()

    def _on_drag(self, event):
        action = self._drag_data["action"]
        dx = event.x_root - self._drag_data["x"]
        dy = event.y_root - self._drag_data["y"]

        ox = self._drag_data["win_x"]
        oy = self._drag_data["win_y"]
        ow = self._drag_data["win_w"]
        oh = self._drag_data["win_h"]

        if action == "move":
            self.root.geometry(f"+{ox + dx}+{oy + dy}")
        else:
            nx, ny, nw, nh = ox, oy, ow, oh
            if "r" in action: nw = max(self.MIN_W, ow + dx)
            if "b" in action: nh = max(self.MIN_H, oh + dy)
            if "l" in action:
                nw = max(self.MIN_W, ow - dx)
                if nw != ow: nx = ox + dx
            if "t" in action:
                nh = max(self.MIN_H, oh - dy)
                if nh != oh: ny = oy + dy
            self.root.geometry(f"{nw}x{nh}+{nx}+{ny}")

    def _on_mousewheel(self, event):
        self.text_box.yview_scroll(int(-1*(event.delta/120)), "units")
        # 如果手动向上滚动，则停止自动滚动；如果滚到底部，则恢复
        if self.text_box.yview()[1] < 1.0:
            self.auto_scroll = False
        else:
            self.auto_scroll = True
        return "break"

    def _close_app(self):
        """双击关闭字幕窗口"""
        print("\n👋 双击关闭字幕窗口")
        self.root.destroy()
        sys.exit(0)

    def update_content(self, history: List[str], current: str, highlight_last=False):
        """更新字幕：历史定稿 + 高亮 + 当前流式"""
        self.text_box.config(state="normal")
        self.text_box.delete("1.0", "end")

        for i, line in enumerate(history):
            is_last = (i == len(history) - 1)
            tag = "hist_hl" if is_last and highlight_last else "hist"
            prefix = "● " if is_last and highlight_last else "  "
            self.text_box.insert("end", prefix + line + "\n", tag)

        if current:
            self.text_box.insert("end", "▶ " + current, "curr")

        if self.auto_scroll:
            self.text_box.see("end")

        self.text_box.config(state="disabled")


# ================= 🔗 WebSocket 客户端 =================
class ASRClient:
    def __init__(self, overlay: SubtitleOverlay, save_on=False, save_dir="", audio_format="wav"):
        self.overlay = overlay
        self.loop = asyncio.new_event_loop()
        self.audio_queue = asyncio.Queue()
        self.is_running = True
        self.segmenter = SubtitleSegmenter()
        self.last_text = ""
        self.printed_sentences = []
        self.save_on = save_on
        self.save_dir = save_dir
        self.audio_format = audio_format
        self.save_file = None
        self.wav_file = None
        self.session_start_time = time.time()
        self.waiting_for_reset = False
        self.last_detected_language = None  # 语言锁定：记住模型检测到的语言
        
        # 🎙️ 文本空窗期检测（替代 Silero VAD，零 CPU 开销）
        self.last_text_change_time = time.time()  # 文本最后一次变化的时刻
        self.last_speech_time = time.time()  # 最近一次检测到能量的时刻（RMS）
        self.is_speaking = False          # RMS 能量状态标记
        self.chunk_send_time = time.time()    # 记录发包时刻，用于计算端到端 RTF
        self.current_tail = ""               # 当前未定稿的流式尾巴
        self.pending_chunks = []             # 🔥 重置期间积压的音频缓存
        
        if self.save_on and self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            
            # --- 保存文本 ---
            self.save_path = os.path.join(self.save_dir, f"subtitle_{timestamp}.txt")
            print(f"📝 文本保存: {self.save_path}")
            self.save_file = open(self.save_path, "w", encoding="utf-8")
            
            # --- 保存音频 (.wav) ---
            self.wav_path = os.path.join(self.save_dir, f"audio_{timestamp}.wav")
            print(f"🎙️ 音频录制: {self.wav_path}")
            self.wav_file = wave.open(self.wav_path, "wb")
            self.wav_file.setnchannels(1)
            self.wav_file.setsampwidth(2)
            self.wav_file.setframerate(SAMPLE_RATE)

    def start(self, mode, language):
        if mode == "mic":
            self.stream = MicStream().start()
        elif mode == "loopback":
            self.stream = LoopbackStream().start()
        else:
            self.stream = MixedStream().start()
            
        threading.Thread(target=self._worker_thread, args=(mode, language), daemon=True).start()
        threading.Thread(target=self._capture_thread, daemon=True).start()

    def _capture_thread(self):
        buf = []
        buf_samples = 0
        while self.is_running:
            try:
                frame = self.stream.read()
            except Exception:
                time.sleep(0.005)
                continue

            if frame is None or len(frame) == 0:
                time.sleep(0.005)
                continue

            buf.append(frame)
            buf_samples += len(frame)

            if buf_samples >= CHUNK_SAMPLES:
                chunk = np.concatenate(buf)[:CHUNK_SAMPLES]
                if self.wav_file:
                    audio_int16 = (chunk * 32767).astype(np.int16)
                    self.wav_file.writeframes(audio_int16.tobytes())
                self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, chunk)
                buf = []
                buf_samples = 0

    def _worker_thread(self, mode, language):
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._run_ws(language))
        finally:
            self.stop()

    def stop(self):
        self.is_running = False
        if self.save_file:
            self.save_file.close()
            self.save_file = None
        if self.wav_file:
            self.wav_file.close()
            self.wav_file = None
            
            # --- MP3 转码逻辑 ---
            if self.audio_format == "mp3" and os.path.exists(self.wav_path):
                mp3_path = self.wav_path.replace(".wav", ".mp3")
                print(f"🎵 正在转码为 MP3: {mp3_path}")
                try:
                    import subprocess
                    subprocess.run(["ffmpeg", "-y", "-i", self.wav_path, "-acodec", "libmp3lame", mp3_path], 
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if os.path.exists(mp3_path):
                        os.remove(self.wav_path)
                        print("✅ MP3 转码完成，已清理临时 WAV 文件。")
                except Exception as e:
                    print(f"⚠️ MP3 转码失败: {e}")

        print("🛑 ASR Client Stopped.")


    async def _do_reset(self, ws, reason="", current_chunk=None):
        """执行重置：保存末句 → 发送 reset 命令 → 立即补发当前块"""
        # 0. 强制换行并清除当前 RTF 行
        sys.stdout.write("\n")
        sys.stdout.flush()
        
        self.waiting_for_reset = True
        
        # 1. 保存当前的 tail 到历史
        if self.current_tail:
            self.segmenter.history.append(self.current_tail)
            if len(self.segmenter.history) > 20:
                self.segmenter.history.pop(0)
            sys.stdout.write(f"✅ {self.current_tail:<80}\n")
        
        # 2. 发送重置指令
        print(f"🔄 [Session Reset: {reason}]")
        reset_cmd = {"command": "reset"}
        if self.last_detected_language:
            reset_cmd["language"] = self.last_detected_language
        await ws.send(json.dumps(reset_cmd))
        
        # 3. 立即补发触发重置的当前 chunk
        if current_chunk is not None:
            await ws.send(current_chunk.astype(np.float32).tobytes())
            self.prev_chunk = current_chunk
            self.chunk_send_time = time.time()
            
        # 4. 清理状态
        self.segmenter.reset()
        self.last_text = ""
        self.current_tail = ""
        self.prev_chunk = None
        self.pending_chunks = []

    async def _run_ws(self, language):
        url = f"{SERVER_WS_URL}"
        if language: url += f"?language={language}"

        while self.is_running:
            try:
                print(f"🔗 Connecting to {url}...")
                async with websockets.connect(url, ping_interval=20, ping_timeout=60) as ws:
                    print("✅ Connected to Qwen3 Backend!")
                    self.segmenter.reset()
                    self.last_text = ""
                    self.current_tail = ""
                    self.prev_chunk = None
                    self.last_resp_time = time.time()
                    self.session_start_time = time.time()
                    self.last_speech_time = time.time()
                    self.last_text_change_time = time.time()
                    self.waiting_for_reset = False
                    self.is_gated = False         # 🔥 门禁状态初始化
                    self.is_speaking = False
                    self.pending_chunks = []      # 🔥 连接时清空缓存

                    async def receiver():
                        try:
                            while self.is_running:
                                resp = await ws.recv()
                                data = json.loads(resp)
                                if "error" in data:
                                    print(f"\n❌ Server Error: {data['error']}")
                                    break
                                    
                                if "status" in data and data["status"] == "reset_ok":
                                    self.waiting_for_reset = False
                                    self.last_resp_time = time.time()
                                    self.session_start_time = time.time()
                                    
                                    # 🔥 移除原有的清空队列逻辑，保留音频以便补发
                                    
                                    self.last_speech_time = time.time()
                                    self.silence_start_time = None
                                    
                                    sys.stdout.write("\n")
                                    if self.is_gated:
                                        sys.stdout.write("✅ Server Ready. Entering Sleep Mode... 💤\n")
                                    else:
                                        sys.stdout.write("✅ Server Ready. Session Reset OK\n------------------------------------------\n")
                                    sys.stdout.flush()
                                    continue

                                if "text" in data and data["text"] and not self.waiting_for_reset:
                                    self.last_resp_time = time.time()  # 🔥 刷新回复时间戳
                                    new_text = data["text"]
                                    if new_text != self.last_text:
                                        self.last_text = new_text
                                        self.last_text_change_time = time.time()  # 🔥 文本变化时刷新
                                        self.last_speech_time = time.time()       # 🔥 只要有新文字识别出来，就刷新说话时间，防止声音小被误休眠
                                        
                                        duration = time.time() - self.session_start_time
                                        new_finals, tail, highlight_last, need_reset, threshold_reached = self.segmenter.process(new_text, duration)
                                        self.current_tail = tail or ""
                                        latency = data.get("latency", 0)
                                        
                                        detected_lang = data.get("language")
                                        if detected_lang:
                                            self.last_detected_language = detected_lang
                                        
                                        self.overlay.root.after(0, self.overlay.update_content, self.segmenter.history, tail, highlight_last)
                                        
                                        if new_finals:
                                            # 先把当前的 RTF 行彻底抹掉，确保定稿行干干净净地出来
                                            sys.stdout.write("\r" + " " * 100 + "\r")
                                            for new_sentence in new_finals:
                                                sys.stdout.write(f"✅ {new_sentence}\n")
                                                self.printed_sentences.append(new_sentence)
                                                if self.save_file:
                                                    self.save_file.write(f"[{time.strftime('%H:%M:%S')}] {new_sentence}\n")
                                                    self.save_file.flush()
                                            sys.stdout.flush()
                                        
                                        if tail:
                                            display_tail = tail[-25:] if len(tail) > 25 else tail
                                            e2e_latency = time.time() - self.chunk_send_time
                                            chunk_sec = CHUNK_MS / 1000
                                            rtf = e2e_latency / chunk_sec
                                            server_rtf = latency / chunk_sec
                                            vad_icon = "\U0001f7e2" if self.is_speaking else "\u26ab"
                                            sys.stdout.write(f"\r{vad_icon} [RTF: {rtf:.2f} | GPU: {server_rtf:.2f}] {display_tail}\033[K")
                                        sys.stdout.flush()
                                        
                                        # 🔥 文本空窗期断句：文本超过 0.8s 没变化 = 没人在说话了
                                        TEXT_IDLE_SEC = 0.8
                                        if threshold_reached and not need_reset and not self.is_gated:
                                            text_idle = (time.time() - self.last_text_change_time) > TEXT_IDLE_SEC
                                            if text_idle:
                                                need_reset = True
                                                reason = "Text Idle (No New Speech)"
                                            elif len(new_text) > self.segmenter.RESET_CHAR_LIMIT:
                                                need_reset = True
                                                reason = "Text Length Limit"
                                        
                                        if need_reset:
                                            if 'reason' not in locals():
                                                reason = "Segmenter Internal"
                                            await self._do_reset(ws, reason=reason)
                                
                        except websockets.exceptions.ConnectionClosed:
                            print("\n🔌 Connection closed")
                        except Exception as e:
                            print(f"\n❌ Receiver Error: {e}")

                    receiver_task = asyncio.create_task(receiver())

                    try:
                        while self.is_running:
                            try:
                                chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=1.0)
                                now = time.time()
                                self.chunk_send_time = now
                                elapsed = now - self.session_start_time
                                
                                # 2. 60秒硬重置（仅在非休眠状态下判断，兜底防膨胀）
                                if elapsed > 60.0 and not self.is_gated:
                                    await self._do_reset(ws, reason="Hard Reset (60s limit)", current_chunk=chunk)
                                    continue

                                # 🎙️ 数学 VAD (RMS 能量检测) - 零功耗，用于门禁
                                rms = np.sqrt(np.mean(chunk**2))
                                RMS_THRESHOLD = 0.002  # 能量阈值，降至 0.002 适应低电平数字麦克风
                                
                                # 🔥 唤醒逻辑：只要能量超标，立刻唤醒
                                if self.is_gated and rms > RMS_THRESHOLD:
                                    self.is_gated = False
                                    self.last_speech_time = now # 唤醒时刷新语音时间
                                    self.session_start_time = now # 🔥 唤醒瞬间重置计时器，给足 60s 空间
                                    sys.stdout.write(f"\n🚀 [Energy Wakeup: RMS={rms:.4f}]\n")
                                    if self.prev_chunk is not None:
                                        await ws.send(self.prev_chunk.astype(np.float32).tobytes())

                                # 🎙️ RMS 能量追踪（替代 Silero VAD，零 CPU 开销）
                                if rms > RMS_THRESHOLD:
                                    self.last_speech_time = now
                                    self.is_speaking = True
                                else:
                                    self.is_speaking = False
                                
                                # 🔥 发包逻辑
                                if not self.is_gated:
                                    await ws.send(chunk.astype(np.float32).tobytes())
                                    self.prev_chunk = chunk
                                    self.chunk_send_time = now
                                else:
                                    self.prev_chunk = chunk
                                
                                # 🔥 重置与休眠逻辑
                                silence_dur = now - self.last_speech_time
                                # (A) 闲置休眠：变回 10秒 没声音才睡觉
                                if silence_dur > 10.0 and not self.is_gated:
                                    await self._do_reset(ws, reason="Idle Sleep (RMS)", current_chunk=chunk)
                                    self.is_gated = True
                                    continue
                                # (B) 定时重置（文本空窗期检测）
                                TEXT_IDLE_RESET_SEC = 0.8
                                text_idle_dur = now - self.last_text_change_time
                                if elapsed > self.segmenter.RESET_SEC_LIMIT and not self.is_gated:
                                    if text_idle_dur >= TEXT_IDLE_RESET_SEC:
                                        await self._do_reset(ws, reason=f"Text Idle ({int(elapsed)}s)", current_chunk=chunk)
                                        continue
                                    
                            except asyncio.TimeoutError:
                                continue

                            if receiver_task.done():
                                break
                    finally:
                        receiver_task.cancel()

            except Exception as e:
                print(f"\n❌ WS Error: {e}, retrying in 3s...")
                await asyncio.sleep(3)



# ================= 🚀 入口 =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default=None, help="Language: zh/en/ja/auto(None)")
    parser.add_argument("--mode", default="loopback", help="Audio mode: loopback/mic")
    parser.add_argument("--save_on", action="store_true", help="Auto save to file")
    parser.add_argument("--save_dir", default=".", help="Directory to save subtitles")
    parser.add_argument("--audio_format", default="wav", choices=["wav", "mp3"], help="Audio format")
    args = parser.parse_args()

    root = tk.Tk()
    overlay = SubtitleOverlay(root)
    client = ASRClient(overlay, save_on=args.save_on, save_dir=args.save_dir, audio_format=args.audio_format)
    client.start(mode=args.mode, language=args.lang)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        client.stop()
        sys.exit(0)
