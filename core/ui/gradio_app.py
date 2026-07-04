import gradio as gr
import os
import sys

# Add Root to Path (to allow 'import core...')
# Current: .../FunClip/core/ui/gradio_app.py
# Target: .../FunClip/
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from core.asr.funasr import FunASREngine
from core.asr.qwen import QwenEngine

# Globals
engine = None
current_model_key = None

def load_engine(model_selection):
    global engine, current_model_key
    
    if engine and current_model_key == model_selection:
        return f"✅ 模型已加载: {model_selection}"
        
    try:
        if model_selection == "Qwen3-Docker":
            engine = QwenEngine()
        elif model_selection == "SenseVoice":
            engine = FunASREngine(sub_mode="sensevoice")
        elif model_selection == "FunASR-Nano":
            engine = FunASREngine(sub_mode="nano")
        elif model_selection == "SeACo (Legacy)":
            engine = FunASREngine(sub_mode="seaco")
        else:
            return "❌ 未知模型"
            
        engine.set_log_callback(lambda x: print(f"[UI] {x}")) # Capture logs if needed
        engine.load_model()
        current_model_key = model_selection
        return f"✅ 成功加载: {model_selection}"
    except Exception as e:
        return f"❌ 加载失败: {e}"

def start_process(audio_file, mic_input, manual_path, model_sel, lang, batch_size, hotwords):
    if not engine:
        msg = load_engine(model_sel)
        if "失败" in msg: return msg, ""
    
    # Priority: Manual Path > Audio Upload > Mic
    target_path = None
    if manual_path and manual_path.strip():
        # 🔥 Fix: Strip quotes from Windows "Copy as Path"
        target_path = manual_path.strip().strip('"').strip("'")
    elif audio_file:
        target_path = audio_file
    elif mic_input:
        target_path = mic_input
    
    if not target_path:
        return "❌ 请提供音频输入 (上传、录音或手动路径)", ""

    # Redirect logs to UI
    log_buffer = []
    def ui_log(msg):
        log_buffer.append(msg)
        print(msg)
        
    engine.set_log_callback(ui_log)
    
    try:
        # Check if directory
        import glob
        files_to_process = []
        if os.path.isdir(target_path):
            exts = ['*.wav', '*.mp3', '*.m4a', '*.mp4', '*.flv', '*.mkv']
            for ext in exts:
                files_to_process.extend(glob.glob(os.path.join(target_path, ext)))
            if not files_to_process:
                return "❌ 目录中未找到音频/视频文件", ""
        else:
            files_to_process = [target_path]

        ui_log(f"📋 共找到 {len(files_to_process)} 个任务")
        
        final_srt = ""
        for idx, fpath in enumerate(files_to_process):
            ui_log(f"\n▶️ [{idx+1}/{len(files_to_process)}] 处理: {os.path.basename(fpath)}")
            res = engine.transcribe(
                audio_path=fpath,
                language=lang,
                batch_size=int(batch_size),
                hotwords=hotwords
            )
            
            # Save files
            out_dir = os.path.dirname(fpath)
            name = os.path.splitext(os.path.basename(fpath))[0]
            
            with open(os.path.join(out_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
                f.write(res["text"])
            with open(os.path.join(out_dir, f"{name}.srt"), "w", encoding="utf-8") as f:
                f.write(res["srt"])
                
            final_srt += f"=== {name} ===\n{res['srt']}\n\n"
            
        return final_srt, "\n".join(log_buffer)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ 错误: {e}", "\n".join(log_buffer)

# UI Layout
with gr.Blocks(title="FunClip Pro (Clean)") as demo:
    gr.Markdown("## 🎬 FunClip Pro - ASR Studio")
    
    with gr.Row():
        with gr.Column():
            with gr.Tab("文件/文件夹"):
                audio_input = gr.Audio(type="filepath", label="📤 上传文件")
                manual_path = gr.Textbox(label="📁 手动输入路径 (支持文件或文件夹)", placeholder="E:\\Video\\test.mp4")
            with gr.Tab("录音"):
                mic_input = gr.Microphone(type="filepath", label="🎤 麦克风输入")
            
            with gr.Accordion("⚙️ 模型设置", open=True):
                model_sel = gr.Dropdown(
                    choices=["SenseVoice", "FunASR-Nano", "SeACo (Legacy)", "Qwen3-Docker"],
                    value="SenseVoice",
                    label="选择模型"
                )
                batch_size = gr.Slider(minimum=1, maximum=64, value=12, step=1, label="Batch Size (Nano/Sense专用)")
                lang = gr.Dropdown(choices=["auto", "zh", "en", "ja"], value="auto", label="语言")
                hotwords = gr.Textbox(label="热词注入 (逗号分隔)")
                
            btn_run = gr.Button("🚀 开始转换", variant="primary")
            
        with gr.Column():
            status_box = gr.Textbox(label="运行日志", lines=10, interactive=False)
            srt_output = gr.Textbox(label="SRT 预览", lines=15, show_copy_button=True)

    btn_run.click(
        fn=start_process,
        inputs=[audio_input, mic_input, manual_path, model_sel, lang, batch_size, hotwords],
        outputs=[srt_output, status_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
