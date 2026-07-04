import gradio as gr
import subprocess
import os
import sys
import time
from pathlib import Path
import logging
import shutil
import tempfile
import traceback
import json

# 0. 强制 UTF-8 输出 (解决 GBK 报错)
sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [WebUI] - %(message)s')

# ================= ⚙️ 核心配置区 (动态路径) =================
# 1. 自动获取当前脚本所在的根目录 (e.g. E:\FunClip)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 2. 项目逻辑目录
PROJECT_PATH = PROJECT_ROOT

# 3. Conda 安装根目录 (如果你的 Conda 不在项目内，这个建议保留原样或改为环境变量)
CONDA_ROOT = r"D:\program files\Miniconda" 

# 4. 【离线任务】使用的 Python
# 注意：如果这个环境不在项目内，建议保留绝对路径
OFFLINE_PYTHON = r"E:\conda\envs\funclip_final\python.exe"

# 5. 【实时任务】环境名称
LIVE_ENV_NAME = "asr_ui_env"

# 尝试读取 config.json 更新配置
config_path = os.path.join(PROJECT_ROOT, "config.json")
if os.path.exists(config_path):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
            if "conda_root" in config_data:
                CONDA_ROOT = config_data["conda_root"]
            if "offline_python" in config_data:
                OFFLINE_PYTHON = config_data["offline_python"]
            if "live_env_name" in config_data:
                LIVE_ENV_NAME = config_data["live_env_name"]
    except Exception as e:
        logging.error(f"Error loading config.json: {e}")

# 6. 脚本路径自动定位
OFFLINE_SCRIPT = os.path.join(PROJECT_PATH, "funclip", "asr1.py")
LIVE_SCRIPT = os.path.join(PROJECT_PATH, "app_live_local.py")
QWEN_LIVE_SCRIPT = os.path.join(PROJECT_ROOT, "app_live_ws.py")
# =========================================================

live_proc = None

# =========================================================
# 📂 模块一：离线文件转写 (逻辑找回)
# =========================================================
# def run_offline_asr(uploaded_files, path_input, engine, output_dir, whisper_size, funasr_mode, lang, whisper_backend, spk_on, batch_size):
#     # 检查 Python
#     if not os.path.exists(OFFLINE_PYTHON):
#         yield f"❌ 错误：找不到离线 Python: {OFFLINE_PYTHON}", "启动失败", None
#         return

#     # 路径处理
#     target_path_str = ""
#     is_batch = False
#     if uploaded_files:
#         p = Path(uploaded_files[0].name).resolve()
#         target_path_str = str(p) if len(uploaded_files) == 1 else str(p.parent)
#         is_batch = len(uploaded_files) > 1
#     elif path_input:
#         target_path_str = path_input.strip()
#         is_batch = os.path.isdir(target_path_str)
#     else:
#         yield "❌ 请提供输入文件或路径", "Error", None
#         return

#     # 输出目录
#     tgt = Path(target_path_str)
#     out_dir = Path(output_dir) if output_dir else (tgt / "out" if is_batch else tgt.parent / tgt.stem)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # 参数映射
#     backend = "faster" if engine == "Whisper" and "Faster" in whisper_backend else ("openai" if engine == "Whisper" else "funasr")
#     sub_mode = "precision"
#     if engine == "FunASR":
#         if "SenseVoice" in funasr_mode: sub_mode = "emotion"
#         elif "Nano" in funasr_mode: sub_mode = "nano"
#         elif "SeACo" in funasr_mode: sub_mode = "seaco"

#     lang_code = {"自动 (Auto)": "auto", "中文 (Chinese)": "zh", "英文 (English)": "en", 
#                  "日文 (Japanese)": "ja", "粤语 (Yue)": "yue", "韩语 (Korean)": "ko"}.get(lang, "auto")

#     # 构建命令
#     cmd = [
#         "cmd.exe", "/c", 
#         OFFLINE_PYTHON, OFFLINE_SCRIPT,
#         "--file", target_path_str,
#         "--output_dir", str(out_dir),
#         "--backend", backend,
#         "--model_size", whisper_size,
#         "--sub_mode", sub_mode,
#         "--language", lang_code,
#         "--batch_size", str(batch_size)  # 🔥🔥🔥【新增】传入 batch_size
#     ]
#     if spk_on: cmd.append("--enable_spk")

#     yield f"🚀 离线任务启动...", "运行中...", None

#     try:
#         subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE).wait()
        
#         found = None
#         if is_batch:
#             f = out_dir / "processing_list.txt"
#             if f.exists(): found = f
#         else:
#             srts = list(out_dir.rglob("*.srt"))
#             if srts: found = max(srts, key=os.path.getctime)
            
#         if found:
#             dest = os.path.join(tempfile.gettempdir(), f"{int(time.time())}_{found.name}")
#             shutil.copy2(found, dest)
#             yield f"✅ 完成: {found.name}", "完成", dest
#         else:
#             yield "⚠️ 未找到输出文件", "结束", None
#     except Exception as e:
#         yield f"❌ 异常: {e}", "Error", None
# =========================================================
# 📂 模块一：离线文件转写 (增强版：支持多行路径、去引号、混合输入)
# =========================================================
# =========================================================
# 📂 模块一：离线文件转写 (V3.0: 文件夹递归展开 + 格式过滤)
# =========================================================

# 定义支持的媒体格式 (防止把 txt 或 jpg 当视频处理)
SUPPORTED_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg'}

def get_all_media_files(path_str):
    """
    递归遍历文件夹，返回所有支持的媒体文件路径列表
    """
    media_files = []
    p = Path(path_str)
    
    if p.is_file():
        if p.suffix.lower() in SUPPORTED_EXTS:
            media_files.append(str(p))
    elif p.is_dir():
        # os.walk 可以递归遍历子文件夹
        for root, dirs, files in os.walk(p):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in SUPPORTED_EXTS:
                    media_files.append(str(file_path))
    
    return media_files

def parse_text_paths(text_input):
    """
    解析文本框：去引号、去空行
    """
    raw_paths = []
    if not text_input:
        return raw_paths
    
    lines = text_input.split('\n')
    for line in lines:
        clean_line = line.strip().strip('"').strip("'") # 核心：去引号
        if clean_line:
            raw_paths.append(clean_line)
    return raw_paths

def run_offline_asr(uploaded_files, mic_audio, path_input, engine, output_dir, rec_save_dir, whisper_size, funasr_mode, lang, whisper_backend, spk_on, batch_size, folder_mode, hotwords_str):
    # 0. 检查环境
    if not os.path.exists(OFFLINE_PYTHON):
        yield f"❌ 错误：找不到离线 Python: {OFFLINE_PYTHON}", "启动失败", None
        return
    
    # 🔥 热词预处理
    hotwords = hotwords_str.strip() if hotwords_str else ""

    # 1. 第一阶段：收集任务
    raw_inputs = []
    if uploaded_files:
        for f in uploaded_files: raw_inputs.append(f.name)
    text_paths = parse_text_paths(path_input)
    raw_inputs.extend(text_paths)
    
    # 新增：处理麦克风录音
    if mic_audio:
        try:
            # 1. 确保保存目录存在
            save_root = rec_save_dir if rec_save_dir else os.path.join(PROJECT_PATH, "recordings")
            os.makedirs(save_root, exist_ok=True)
            
            # 2. 生成文件名 (时间戳)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            # Gradio 录音通常是 wav，但也可能是 flac，保留后缀或默认 wav
            ext = os.path.splitext(mic_audio)[-1]
            if not ext: ext = ".wav"
            
            new_filename = f"rec_{timestamp}{ext}"
            save_path = os.path.join(save_root, new_filename)
            
            # 3. 移动文件到目标目录
            shutil.copy2(mic_audio, save_path)
            yield f"🎙️ 录音已保存至: {save_path}", "录音保存成功", None
            
            # 4. 加入任务队列
            raw_inputs.append(save_path)
            
        except Exception as e:
            yield f"❌ 录音保存失败: {e}", "Error", None
            return

    if not raw_inputs:
        yield "❌ 未检测到有效输入", "Error", None
        return

    yield "🔍 正在扫描文件...", "扫描中", None
    
    # 递归展开文件夹
    final_tasks = []
    for raw_path in raw_inputs:
        if not os.path.exists(raw_path): continue
        files_found = get_all_media_files(raw_path)
        final_tasks.extend(files_found)

    final_tasks = list(set(final_tasks))
    final_tasks.sort()
    
    if not final_tasks:
        yield "⚠️ 未找到支持的媒体文件", "无任务", None
        return

    # ========================================================
    # 🔥🔥🔥 智能路径计算逻辑 (核心修改) 🔥🔥🔥
    # ========================================================
    # 基础输出目录 (默认为 output)
    base_out = output_dir if output_dir else "output"
    final_out_dir = base_out

    if folder_mode:
        # 如果开启了文件夹归档模式，尝试提取输入文件夹的名字
        # 例如：输入 E:\JavaCourse -> 输出 output\JavaCourse
        folder_name = ""
        if path_input and os.path.isdir(path_input.strip('"')):
            folder_name = os.path.basename(path_input.strip('"').rstrip("\\/"))
        
        # 如果找到了文件夹名，拼接到输出路径后
        if folder_name:
            final_out_dir = os.path.join(base_out, folder_name)
        
        # 提前创建这个总文件夹
        if not os.path.exists(final_out_dir):
            os.makedirs(final_out_dir)
            
        yield f"📂 [文件夹模式] 开启！字幕将统一归档至: {final_out_dir}", "准备中", None
    
    # 2. 第二阶段：生成任务清单文件
    task_list_file = os.path.join(PROJECT_PATH, "temp_task_list.txt")
    try:
        with open(task_list_file, "w", encoding="utf-8") as f:
            for p in final_tasks:
                f.write(p + "\n")
    except Exception as e:
        yield f"❌ 无法创建任务清单: {e}", "Error", None
        return

    yield f"🚀 启动引擎... (共 {len(final_tasks)} 个文件)", "启动中", None

    # 3. 第三阶段：构建命令
    backend = "funasr" # Default
    if engine == "Whisper":
        backend = "faster" if "Faster" in whisper_backend else "openai"
    elif engine == "Qwen3 (Docker)":
        backend = "qwen_vllm"
    
    sub_mode = "precision"
    if engine == "FunASR":
        if "SenseVoice" in funasr_mode: sub_mode = "emotion"
        elif "Nano" in funasr_mode: sub_mode = "nano"
        elif "SeACo" in funasr_mode: sub_mode = "seaco"

    lang_code = {"自动 (Auto)": "auto", "中文": "zh", "英文": "en", 
                 "日文": "ja", "粤语": "yue", "韩语": "ko"}.get(lang, "auto")
    
    cmd = [
        "cmd.exe", "/k",  # 🔥 改成 /k 让窗口保持打开，方便看错误
        OFFLINE_PYTHON, OFFLINE_SCRIPT,
        "--file_list", task_list_file,
        "--output_dir", str(final_out_dir), # 🔥 使用计算好的新路径
        "--backend", backend,
        "--model_size", whisper_size,
        "--sub_mode", sub_mode,
        "--language", lang_code,
        "--batch_size", str(batch_size)
    ]
    if spk_on: cmd.append("--enable_spk")
    
    # 传递开关给后端
    if folder_mode: cmd.append("--folder_mode")
    
    # 🔥 热词传递
    if hotwords:
        cmd.extend(["--hotwords", hotwords])

    try:
        proc = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
        
        yield f"✅ 批量任务已启动！\n输出目录: {final_out_dir}\n\n请查看弹出的黑色命令窗口查看实时进度...", "运行中", None
        
        proc.wait()
        
        if os.path.exists(task_list_file):
            os.remove(task_list_file)

        # ========================================================
        # 🔥🔥🔥 结果收集逻辑 (适配文件夹模式) 🔥🔥🔥
        # ========================================================
        found_srts = []
        # 注意：这里我们要去 final_out_dir 找文件
        search_base = Path(final_out_dir)
        
        for input_file in final_tasks:
            p = Path(input_file)
            stem = p.stem
            
            # 根据模式预测路径
            if folder_mode:
                 # 扁平模式: output/JavaCourse/01.srt
                 expected_srt = search_base / f"{stem}.srt"
            else:
                 # 默认模式: output/01/01.srt
                 expected_srt = search_base / stem / f"{stem}.srt"
            
            if expected_srt.exists():
                found_srts.append(str(expected_srt.resolve()))
        
        if found_srts:
            path_list_str = "\n".join(found_srts)
            success_msg = (
                f"🎉 处理完毕！共生成 {len(found_srts)} 个字幕文件。\n"
                f"========================================\n"
                f"📂 SRT 绝对路径列表 (请直接复制):\n"
                f"========================================\n"
                f"{path_list_str}\n"
                f"========================================\n"
            )
            yield success_msg, "完成", found_srts[-1]
        else:
            yield "⚠️ 任务结束，但未检测到生成的 SRT 文件。", "结束", None

    except Exception as e:
        yield f"❌ 运行异常: {e}", "Error", None

# def run_offline_asr(uploaded_files, path_input, engine, output_dir, whisper_size, funasr_mode, lang, whisper_backend, spk_on, batch_size):
#     # 0. 检查环境
#     if not os.path.exists(OFFLINE_PYTHON):
#         yield f"❌ 错误：找不到离线 Python: {OFFLINE_PYTHON}", "启动失败", None
#         return

#     # 1. 第一阶段：收集任务
#     raw_inputs = []
#     if uploaded_files:
#         for f in uploaded_files: raw_inputs.append(f.name)
#     text_paths = parse_text_paths(path_input)
#     raw_inputs.extend(text_paths)

#     if not raw_inputs:
#         yield "❌ 未检测到有效输入", "Error", None
#         return

#     yield "🔍 正在扫描文件...", "扫描中", None
    
#     # 递归展开文件夹
#     final_tasks = []
#     for raw_path in raw_inputs:
#         if not os.path.exists(raw_path): continue
#         files_found = get_all_media_files(raw_path) # 使用之前定义的 V3.0 函数
#         final_tasks.extend(files_found)

#     final_tasks = list(set(final_tasks))
#     final_tasks.sort()
    
#     if not final_tasks:
#         yield "⚠️ 未找到支持的媒体文件", "无任务", None
#         return

#     # 2. 第二阶段：生成任务清单文件 (Task List)
#     # -------------------------------------------------
#     # 创建一个临时 txt 文件，把所有要处理的视频路径写进去，一行一个
#     task_list_file = os.path.join(PROJECT_PATH, "temp_task_list.txt")
#     try:
#         with open(task_list_file, "w", encoding="utf-8") as f:
#             for p in final_tasks:
#                 f.write(p + "\n")
#     except Exception as e:
#         yield f"❌ 无法创建任务清单: {e}", "Error", None
#         return

#     yield f"🚀 启动引擎... (共 {len(final_tasks)} 个文件，模型仅加载一次)", "启动中", None

#     # 3. 第三阶段：一次性调用后台
#     # -------------------------------------------------
#     # 参数映射
#     backend = "faster" if engine == "Whisper" and "Faster" in whisper_backend else ("openai" if engine == "Whisper" else "funasr")
#     sub_mode = "precision"
#     if engine == "FunASR":
#         if "SenseVoice" in funasr_mode: sub_mode = "emotion"
#         elif "Nano" in funasr_mode: sub_mode = "nano"
#         elif "SeACo" in funasr_mode: sub_mode = "seaco"

#     lang_code = {"自动 (Auto)": "auto", "中文 (Chinese)": "zh", "英文 (English)": "en", 
#                  "日文 (Japanese)": "ja", "粤语 (Yue)": "yue", "韩语 (Korean)": "ko"}.get(lang, "auto")
    
#     # 确定输出目录 (如果没有指定，就传空字符串，backend里会处理)
#     final_out_dir = output_dir if output_dir else ""

#     cmd = [
#         "cmd.exe", "/c", 
#         OFFLINE_PYTHON, OFFLINE_SCRIPT,
#         "--file_list", task_list_file,  # <--- 关键变化：传列表文件，而不是单文件
#         "--output_dir", str(final_out_dir),
#         "--backend", backend,
#         "--model_size", whisper_size,
#         "--sub_mode", sub_mode,
#         "--language", lang_code,
#         "--batch_size", str(batch_size)
#     ]
#     if spk_on: cmd.append("--enable_spk")
#     try:
#         # 启动唯一的进程
#         proc = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
        
#         yield f"✅ 批量任务已启动！\n\n请查看弹出的黑色命令窗口查看实时进度。\n\n处理逻辑：\n1. 加载模型 (约5-10秒)\n2. 连续处理 {len(final_tasks)} 个视频\n3. 处理完成后窗口会自动关闭", "运行中", None
        
#         # 等待所有任务完成
#         proc.wait()
        
#         # 完成后清理临时文件
#         if os.path.exists(task_list_file):
#             os.remove(task_list_file)

#         # ========================================================
#         # 🔥🔥🔥 新增：自动收集生成的 SRT 路径 🔥🔥🔥
#         # ========================================================
#         found_srts = []
        
#         # 基础输出目录 (如果用户没填，默认为当前目录)
#         base_out = Path(output_dir) if output_dir else Path(".")
        
#         for input_file in final_tasks:
#             # 根据 asr1.py 的逻辑预测路径: output_dir/文件名/文件名.srt
#             p = Path(input_file)
#             stem = p.stem
#             expected_srt = base_out / stem / f"{stem}.srt"
            
#             # 检查文件是否存在
#             if expected_srt.exists():
#                 # 获取绝对路径，方便跨盘符复制
#                 found_srts.append(str(expected_srt.resolve()))
        
#         if found_srts:
#             # 拼接成字符串
#             path_list_str = "\n".join(found_srts)
            
#             success_msg = (
#                 f"🎉 所有视频处理完毕！共生成 {len(found_srts)} 个字幕文件。\n"
#                 f"========================================\n"
#                 f"📂 SRT 绝对路径列表 (请直接复制下面的路径):\n"
#                 f"========================================\n"
#                 f"{path_list_str}\n"
#                 f"========================================\n"
#                 f"👉 提示：复制上方路径，直接粘贴到【Qwen翻译端】的批量输入框即可。"
#             )
#             # 返回最后一个文件作为示例下载，但日志里显示全部
#             yield success_msg, "完成", found_srts[-1]
#         else:
#             yield "⚠️ 任务结束，但未能自动检测到生成的 SRT 文件 (请检查黑框内是否有报错)", "结束", None

#     except Exception as e:
#         yield f"❌ 运行异常: {e}", "Error", None



    # try:
    #     # 启动唯一的进程
    #     proc = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
        
    #     yield f"✅ 批量任务已启动！\n\n请查看弹出的黑色命令窗口查看实时进度。\n\n处理逻辑：\n1. 加载模型 (约5-10秒)\n2. 连续处理 {len(final_tasks)} 个视频\n3. 处理完成后窗口会自动关闭", "运行中", None
        
    #     # 等待所有任务完成
    #     proc.wait()
        
    #     # 完成后清理临时文件
    #     if os.path.exists(task_list_file):
    #         os.remove(task_list_file)
            
    #     yield "🎉 所有视频处理完毕！", "完成", None

    # except Exception as e:
    #     yield f"❌ 运行异常: {e}", "Error", None

# =========================================================
# 🎤 模块二：实时系统听写 (修复参数传递)
# =========================================================
# 🔥 核心修复：这里接收 5 个参数，与 UI 对应
def run_live_asr(model_name, lang_label, device_mode_label, overlay_on, force_console, save_on, save_dir, audio_format):
    global live_proc
    if live_proc: stop_live_asr()

    # 1. 检查文件
    if not os.path.exists(LIVE_SCRIPT):
        return f"❌ 找不到引擎脚本: {LIVE_SCRIPT}"

    # 2. 参数处理
    model_root = os.path.join(PROJECT_ROOT, "model", "models")
    
    if "Qwen3" in model_name:
        # Qwen3 专用逻辑
        target_script = QWEN_LIVE_SCRIPT
        lang_code = {"中文": "zh", "英文": "en", "自动": "auto"}.get(lang_label, "auto")
        mode_arg = "loopback" if "Loopback" in device_mode_label else "mic"
        cmd_str = f'python "{target_script}" --mode {mode_arg} --lang {lang_code}'
        if save_on:
            cmd_str += f' --save_on --save_dir "{save_dir}" --audio_format {audio_format}'
        title_str = "Qwen3 Real-time Subtitles"
    else:
        # 原有 FunASR 逻辑
        target_script = LIVE_SCRIPT
        model_dir = os.path.join(model_root, "FunAudioLLM", "Fun-ASR-Nano-2512") if "Nano" in model_name else os.path.join(model_root, "iic", "SenseVoiceSmall")
        lang_code = {"中文": "zh", "英文": "en", "自动": "auto"}.get(lang_label, "auto")
        
        mode_map = {
            "系统内录 (Loopback)": "loopback",
            "麦克风 (Microphone)": "mic",
            "混合模式 (Mix - 实验性)": "mix"
        }
        mode_arg = mode_map.get(device_mode_label, "loopback")
        cmd_str = f'python "{target_script}" --model_dir "{model_dir}" --language {lang_code} --device_mode {mode_arg}'
        if overlay_on: cmd_str += " --overlay"
        title_str = f"FunASR Live Engine ({mode_arg})"

    # 3. 生成启动脚本
    bat_file = os.path.join(PROJECT_PATH, "start_live_engine.bat")
    activate_script = os.path.join(CONDA_ROOT, "Scripts", "activate.bat")
    
    # 🔥 Bat 内容：先激活环境，再运行
    bat_content = f"""@echo off
chcp 65001 >nul
title {title_str}
echo ==========================================
echo  🚀 Activating Env: {LIVE_ENV_NAME}...
echo ==========================================
call "{activate_script}" {LIVE_ENV_NAME}

echo.
echo  🚀 Launching Engine...
echo  Command: {cmd_str}
echo ==========================================
{cmd_str}

if %errorlevel% neq 0 (
    echo.
    echo ❌ Error occurred! Press any key to exit...
    pause
)
"""
    
    with open(bat_file, "w", encoding="utf-8") as f:
        f.write(bat_content)

    try:
        # 启动 Bat
        live_proc = subprocess.Popen(
            ["cmd.exe", "/c", bat_file],
            creationflags=subprocess.CREATE_NEW_CONSOLE, # 强制弹窗
            cwd=PROJECT_PATH
        )
        return f"✅ 实时引擎已启动！\n模式: {mode_arg}\n环境: {LIVE_ENV_NAME}\n\n请查看新弹出的黑框和屏幕下方的字幕条。"
    except Exception as e:
        return f"❌ 启动失败: {e}"

def stop_live_asr():
    os.system('taskkill /f /fi "WINDOWTITLE eq FunASR Live Engine*" /im python.exe')
    return "🛑 已发送停止指令"


# =========================================================
# 🖥️ UI 界面
# =========================================================
with gr.Blocks(title="SuperASR Final Fixed", theme=gr.themes.Soft()) as app:
    gr.Markdown("## ⚡ SuperASR Final Integration (Fixed)")
    
    # ================= 🚀 Qwen3 Docker 控制函数 (UI 元素移到下方) =================
    def start_qwen_service():
        root_dir = PROJECT_ROOT
        bat_path = os.path.join(root_dir, "start_qwen_backend.bat")
        if not os.path.exists(bat_path): return f"❌ 找不到启动脚本: {bat_path}"
        subprocess.Popen(["cmd", "/c", "start", bat_path], shell=True)
        return "✅ 已发送启动命令，请等待10-20秒..."

    def stop_qwen_service():
        root_dir = PROJECT_ROOT
        bat_path = os.path.join(root_dir, "stop_qwen_backend.bat")
        if not os.path.exists(bat_path): return f"❌ 找不到停止脚本: {bat_path}"
        subprocess.Popen(["cmd", "/c", "start", bat_path], shell=True)
        return "✅ 已发送停止命令，显存已释放。"
    # =========================================================================
    
    with gr.Tabs():
        # Tab 1: 离线
        with gr.TabItem("📂 离线文件转写"):
            with gr.Row():
                with gr.Column():
                    files = gr.File(label="文件输入 (Batch Input)", file_count="multiple")
                    with gr.Row():
                        mic_audio = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ 麦克风录音 (Record)", interactive=True)
                        with gr.Column():
                            rec_save_dir = gr.Textbox(label="💾 录音保存目录 (Save Record To)", value=os.path.join(PROJECT_PATH, "recordings"))
                            btn_clear_mic = gr.Button("🗑️ 清空录音 (Clear)", size="sm", variant="secondary")
                    
                    path_txt = gr.Textbox(label="或 本地路径 (Local Paths)")
                    default_out = os.path.join(PROJECT_PATH, "output")
                    out_path = gr.Textbox(label="输出目录 (Output Dir)", value=default_out)
                    chk_folder_mode = gr.Checkbox(label="📂 文件夹归档模式", value=False, info="扁平输出：不创建子文件夹")
                    with gr.Row():
                        # 🔥 新增 Qwen3 (Docker) 选项
                        engine = gr.Radio(["FunASR", "Whisper", "Qwen3 (Docker)"], label="引擎", value="FunASR")
                        funasr_mode = gr.Radio(["SeACo", "SenseVoice", "Nano"], label="FunASR模式", value="SeACo")
                    
                    # --- Whisper 面板 ---
                    whisper_grp = gr.Group(visible=False)
                    with whisper_grp:
                        w_backend = gr.Radio(["Faster", "Official"], label="Whisper后端", value="Faster")
                        w_size = gr.Dropdown(["turbo", "large-v3"], label="尺寸", value="turbo")
                        
                    # --- Qwen3 面板 (新增) ---
                    qwen_grp = gr.Group(visible=False)
                    with qwen_grp:
                        gr.Markdown("#### 🐳 Qwen3 后端控制")
                        with gr.Row():
                            btn_start_qwen = gr.Button("🚀 启动 (Start)", variant="primary", size="sm")
                            btn_stop_qwen = gr.Button("🛑 停止 (Stop)", variant="stop", size="sm")
                        gr.Markdown("> 首次使用请先点击「启动」，等待黑框加载完毕再点击「开始转写」。")

                    lang = gr.Dropdown(["自动 (Auto)", "中文", "英文", "日文"], label="语言", value="自动 (Auto)")
                    spk = gr.Checkbox(label="说话人区分 (SPK)", value=False)
                    batch_size = gr.Slider(minimum=4, maximum=48, value=24, step=4, label="批处理量 (Batch Size)", info="4080 推荐 24-32，显存不足可降低")
                    # 🔥 新增：热词输入框
                    hotwords = gr.Textbox(label="🔥 热词 (Hotwords)", placeholder="例如: FunASR,语音识别,GPT", info="用逗号分隔，提升专业术语识别准确率 (仅Nano模式生效)")
                    btn_run = gr.Button("🚀 开始转写", variant="primary")
                with gr.Column():
                    log_out = gr.Textbox(label="日志", lines=10)
                    status_out = gr.Textbox(label="状态")
                    file_out = gr.File(label="结果")
            
            # 🔥 动态显隐逻辑
            def on_engine_change(x):
                # 返回 [whisper_visible, qwen_visible, funasr_mode_interactive]
                # FunASR 模式下，右边的 funasr_mode 应该生效；其他模式下其实可以禁用，但为了简单暂不禁用
                return [
                    gr.update(visible=(x == "Whisper")), 
                    gr.update(visible=(x == "Qwen3 (Docker)"))
                ]

            engine.change(on_engine_change, engine, [whisper_grp, qwen_grp])
            
            # 🔥 新增：清空录音按钮逻辑
            btn_clear_mic.click(lambda: None, None, mic_audio)

            # btn_run.click(run_offline_asr, [files, path_txt, engine, out_path, w_size, funasr_mode, lang, w_backend, spk], [log_out, status_out, file_out])
            btn_run.click(
                run_offline_asr, 
                [files, mic_audio, path_txt, engine, out_path, rec_save_dir, w_size, funasr_mode, lang, w_backend, spk, batch_size, chk_folder_mode, hotwords], 
                [log_out, status_out, file_out]
            )
            
            # 🔥 绑定 Qwen 按钮事件
            btn_start_qwen.click(start_qwen_service, inputs=[], outputs=[status_out])
            btn_stop_qwen.click(stop_qwen_service, inputs=[], outputs=[status_out])

        # Tab 2: 实时 (这里修复了参数)
        with gr.TabItem("🎤 实时系统听写"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ⚙️ 实时配置")
                    l_model = gr.Radio(["Nano (推荐)", "SenseVoice", "Qwen3 (实时字幕)"], label="模型", value="Nano (推荐)")
                    l_lang = gr.Dropdown(["自动", "中文", "英文", "日文"], label="语言", value="自动")
                    
                    # 🔥 把你丢掉的 device_mode 加回来了
                    l_mode = gr.Dropdown(
                        ["系统内录 (Loopback)", "麦克风 (Microphone)", "混合模式 (Mix - 实验性)"], 
                        label="音频源", value="系统内录 (Loopback)"
                    )
                    l_overlay = gr.Checkbox(label="开启悬浮字幕框", value=True)
                    with gr.Row():
                        l_save_on = gr.Checkbox(label="自动保存 (Save)", value=False)
                        l_audio_fmt = gr.Dropdown(["wav", "mp3"], label="格式", value="wav")
                    default_save_dir = os.path.join(PROJECT_ROOT, "qwen_server", "output")
                    l_save_dir = gr.Textbox(label="保存路径", value=default_save_dir)
                    l_console = gr.Checkbox(label="强制弹出黑框 (调试)", value=True, visible=False) # 隐藏但保留参数位
                    
                    with gr.Row():
                        btn_live_start = gr.Button("▶️ 启动引擎 (黑框)", variant="primary")
                        btn_live_stop = gr.Button("🛑 停止", variant="stop")
                
                with gr.Column():
                    live_status = gr.Textbox(label="引擎反馈", lines=5)

            # 🔥 核心修复：这里的参数列表 [l_mode, l_overlay, l_console] 必须和 run_live_asr 对应
            btn_live_start.click(
                run_live_asr,
                [l_model, l_lang, l_mode, l_overlay, l_console, l_save_on, l_save_dir, l_audio_fmt],
                [live_status]
            )
            btn_live_stop.click(stop_live_asr, None, live_status)

if __name__ == "__main__":
    app.launch(inbrowser=True)