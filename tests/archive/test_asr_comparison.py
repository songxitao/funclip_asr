import os
import io
import re
import sys
import time
import socket
import requests
import subprocess

# 强制使用 UTF-8 编码输出以防 Windows 控制台/重定向乱码
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 字符对齐辅助工具，处理中文字符占用 2 个字符宽度的问题
def get_display_width(s):
    width = 0
    for c in str(s):
        if '\u4e00' <= c <= '\u9fa5':
            width += 2
        else:
            width += 1
    return width

def pad_right(s, width):
    cur_w = get_display_width(s)
    if cur_w >= width:
        return str(s)
    return str(s) + ' ' * (width - cur_w)

def print_row(col1, col2, col3):
    w1, w2, w3 = 30, 22, 22
    r1 = pad_right(col1, w1)
    r2 = pad_right(col2, w2)
    r3 = pad_right(col3, w3)
    print(f"{r1} | {r2} | {r3}")

# 端口连通性检测
def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(('127.0.0.1', port)) == 0

# 文本清理：只保留核心文字（中文字符、英文字母、数字），过滤掉所有标点和空白符
def clean_text(text):
    return re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", text)

# Levenshtein 动态规划编辑距离算法
def get_edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,    # 删除
                    dp[i][j-1] + 1,    # 插入
                    dp[i-1][j-1] + 1   # 替换
                )
    return dp[m][n]

# 计算字符错误率 (CER)
def calculate_cer(text_p, text_o):
    clean_p = clean_text(text_p)
    clean_o = clean_text(text_o)
    max_len = max(len(clean_p), len(clean_o))
    if max_len == 0:
        return 0, clean_p, clean_o, 0.0
    dist = get_edit_distance(clean_p, clean_o)
    cer = dist / max_len
    return dist, clean_p, clean_o, cer

def main():
    python_path = r"E:\conda\envs\asr_ui_env\python.exe"
    audio_path = r"E:\下载\下载\李雪花2.wav"
    
    if not os.path.exists(audio_path):
        print(f"错误: 默认测试音频文件不存在，路径为: {audio_path}")
        sys.exit(1)
        
    pytorch_proc = None
    onnx_proc = None
    
    # 获取 funclip-pro 的根目录
    # 由于此脚本在 tests/ 文件夹下，因此其父目录为项目根目录
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # 1. 自动启停与等待机制
        pytorch_port_already_open = is_port_open(8001)
        onnx_port_already_open = is_port_open(8002)
        
        # 启动 PyTorch-GPU 服务
        if not pytorch_port_already_open:
            print("检测到端口 8001 (PyTorch) 未在监听，正在后台拉起 PyTorch-GPU 服务...")
            pytorch_proc = subprocess.Popen(
                [python_path, "asr_service.py"],
                cwd=project_dir,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            print("端口 8001 (PyTorch) 已被监听，跳过自动拉起。")
            
        # 启动 ONNX-GPU 服务
        if not onnx_port_already_open:
            print("检测到端口 8002 (ONNX) 未在监听，正在后台拉起 ONNX-GPU 服务...")
            onnx_proc = subprocess.Popen(
                [python_path, "asr_onnx_service.py"],
                cwd=project_dir,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            print("端口 8002 (ONNX) 已被监听，跳过自动拉起。")
            
        # 等待服务就绪
        need_wait = (pytorch_proc is not None) or (onnx_proc is not None)
        if need_wait:
            print("正在等待服务拉起与模型载入（最长 15 秒）...")
            start_wait = time.time()
            while time.time() - start_wait < 15:
                p_ready = pytorch_port_already_open or is_port_open(8001)
                o_ready = onnx_port_already_open or is_port_open(8002)
                if p_ready and o_ready:
                    print("检测到双服务端口均已处于可用状态。")
                    break
                time.sleep(1)
            
            # 额外预留 3 秒让 Fastapi 完成 startup 模型加载与资源调度
            time.sleep(3)
        
        # 验证服务最终是否连通
        if not is_port_open(8001):
            print("错误: 端口 8001 服务连接失败。")
            sys.exit(1)
        if not is_port_open(8002):
            print("错误: 端口 8002 服务连接失败。")
            sys.exit(1)
            
        print("\n双服务均就绪，开始转写性能测试...")
        
        # 2. 向 PyTorch 服务发转写请求
        print("1/2. 正在请求 PyTorch-GPU (8001) 转写服务...")
        py_url = "http://127.0.0.1:8001/transcribe"
        py_text = ""
        py_time = 0.0
        
        try:
            with open(audio_path, "rb") as f:
                start_req = time.time()
                resp = requests.post(py_url, files={"file": f}, data={"vad_split": "true"})
                py_time = time.time() - start_req
                if resp.status_code == 200:
                    py_text = resp.json().get("text", "")
                else:
                    print(f"PyTorch 请求失败，HTTP 状态码: {resp.status_code}, 返回: {resp.text}")
                    sys.exit(1)
        except Exception as e:
            print(f"请求 PyTorch 服务时发生异常: {e}")
            sys.exit(1)
            
        # 3. 向 ONNX 服务发转写请求
        print("2/2. 正在请求 ONNX-GPU (8002) 转写服务...")
        onnx_url = "http://127.0.0.1:8002/transcribe"
        onnx_text = ""
        onnx_time = 0.0
        
        try:
            with open(audio_path, "rb") as f:
                start_req = time.time()
                resp = requests.post(onnx_url, files={"file": f}, data={"vad_split": "true"})
                onnx_time = time.time() - start_req
                if resp.status_code == 200:
                    onnx_text = resp.json().get("text", "")
                else:
                    print(f"ONNX 请求失败，HTTP 状态码: {resp.status_code}, 返回: {resp.text}")
                    sys.exit(1)
        except Exception as e:
            print(f"请求 ONNX 服务时发生异常: {e}")
            sys.exit(1)
            
        # 4. 计算与分析
        speedup = py_time / onnx_time if onnx_time > 0 else 0.0
        dist, clean_p, clean_o, cer = calculate_cer(py_text, onnx_text)
        
        # 5. 打印漂亮的对比表格
        print("\n" + "=" * 80)
        print_row("评估指标", "PyTorch-GPU (8001)", "ONNX-GPU (8002)")
        print("-" * 80)
        print_row("转写耗时 (秒)", f"{py_time:.4f}", f"{onnx_time:.4f}")
        print_row("性能加速比 (PyTorch/ONNX)", "1.00x", f"{speedup:.2f}x")
        print_row("原始总字数 (带标点)", f"{len(py_text)}", f"{len(onnx_text)}")
        print_row("核心字数 (去标点/空白)", f"{len(clean_p)}", f"{len(clean_o)}")
        print_row("字符差异数 (编辑距离)", "-", f"{dist}")
        print_row("字符错误率 (CER)", "-", f"{cer * 100:.2f}%")
        print_row("吻合度 (1 - CER)", "-", f"{(1.0 - cer) * 100:.2f}%")
        print("=" * 80)
        
        # 6. 转写文本前 100 字展示
        print("PyTorch 转写文本前 100 字展示:")
        print(py_text[:100] + ("..." if len(py_text) > 100 else ""))
        print("-" * 80)
        print("ONNX 转写文本前 100 字展示:")
        print(onnx_text[:100] + ("..." if len(onnx_text) > 100 else ""))
        print("=" * 80 + "\n")
        
    finally:
        # 7. 优雅地杀掉并清理两个后台进程
        for name, proc in [("PyTorch-GPU 服务", pytorch_proc), ("ONNX-GPU 服务", onnx_proc)]:
            if proc is not None:
                print(f"正在清理后台进程: {name} (PID: {proc.pid})...")
                try:
                    proc.terminate()
                    # 等待优雅终止，最多等待 5 秒
                    proc.wait(timeout=5)
                    print(f"{name} 进程已优雅终止。")
                except subprocess.TimeoutExpired:
                    print(f"{name} 未能在 5 秒内响应 terminate，正在强制杀掉 (kill)...")
                    try:
                        proc.kill()
                        proc.wait()
                    except Exception as ke:
                        print(f"强制杀掉 {name} 进程失败: {ke}")
                except Exception as e:
                    print(f"清理 {name} 进程时遇到未知错误: {e}")

if __name__ == "__main__":
    main()
