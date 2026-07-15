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

# ================= 📦 Core 包导入 =================
# 不再直接 import funasr 的 AutoModel
# 全量替换为：
from funclip_pro.core.audio import (
    process_audio_frame,
    LoopbackStream,
    MicStream,
    MixedStream,
)
from funclip_pro.core.streaming_asr import (
    FunAsrStreamingEngine,
)
from funclip_pro.config.loader import resolve_model_path
# ===================================================

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

# ================= 🧠 主逻辑（薄壳） =================
def run_engine(model_dir, lang, device_mode, gui_queue):
    """薄壳版 run_engine：只做编排，不做算法。"""
    # 1. 导入 core 包（函数内延迟导入，保持模块级简洁）
    from funclip_pro.core.streaming_asr import FunAsrStreamingEngine
    from funclip_pro.core.audio import LoopbackStream, MicStream, MixedStream
    from funclip_pro.config.loader import resolve_model_path

    if gui_queue:
        gui_queue.put("⏳ 正在初始化流式引擎...")

    # 2. 选择音频流
    if device_mode == "loopback":
        stream = LoopbackStream()
    elif device_mode == "mic":
        stream = MicStream()
    else:
        stream = MixedStream()

    # 3. 启动音频采集
    try:
        stream.start()
    except Exception as e:
        print(f"❌ Audio Init Failed: {e}")
        if gui_queue:
            gui_queue.put(f"❌ 音频初始化失败: {e}")
        return

    # 4. 创建流式引擎（惰性加载模型），传递 VAD 模式配置
    engine = FunAsrStreamingEngine(config={
        "model_dir": model_dir,
        "vad_mode": VAD_MODE,
        "vad_model_path": VAD_MODEL_PATH,
        "vad_start_threshold": VAD_START_THRESHOLD,
        "vad_end_threshold": VAD_END_THRESHOLD,
        "pause_limit_sec": PAUSE_LIMIT_SEC,
        "max_sentence_sec": MAX_SENTENCE_SEC,
        "min_sentence_sec": MIN_SENTENCE_SEC,
    })
    session_id = engine.create_session()

    if gui_queue:
        gui_queue.put("✨ 就绪")

    # 5. 状态变量
    history = []
    CHUNK_DURATION = 0.032
    pause_limit_count = int(PAUSE_LIMIT_SEC / CHUNK_DURATION)
    min_len_points = int(16000 * MIN_SENTENCE_SEC)
    last_preview_time = 0
    PREVIEW_MIN_INTERVAL = 0.3
    buffer = []
    silence_cnt = 0
    is_speaking = False

    print("✅ Engine Ready")

    # 6. 主循环（使用 engine.feed_chunk 代替直接调用 model.generate）
    while True:
        try:
            chunk = stream.read()

            # 补齐或截断到 512 点
            if len(chunk) != 512:
                if len(chunk) > 512:
                    chunk = chunk[:512]
                else:
                    chunk = np.pad(chunk, (0, 512 - len(chunk)))

            # 调用流式引擎
            results = engine.feed_chunk(session_id, chunk)

            for seg in results:
                text = seg.get("text", "").strip()
                if not text or len(text) <= 1:
                    continue

                if not seg.get("is_final", True):
                    # 🟢 预览文本：typing 模式显示，不记入 history
                    if gui_queue:
                        gui_queue.put({
                            "hist": history[:],
                            "curr": text + " 🟢",
                            "typing": True,
                        })
                else:
                    # 📝 定稿文本：记入 history
                    print(f"\n📝 {text}")
                    history.append(text)
                    if len(history) > 4:
                        history.pop(0)
                    if gui_queue:
                        gui_queue.put({"hist": history[:-1], "curr": history[-1]})

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
