# -*- coding: utf-8 -*-
"""FunClip Pro ASR & 说话人分离命令行客户端 (CLI Client)

用法：
    E:\conda\envs\asr_ui_env\python.exe cli_transcribe.py <音频路径> [参数]

示例：
    E:\conda\envs\asr_ui_env\python.exe cli_transcribe.py test.wav --diarize
    E:\conda\envs\asr_ui_env\python.exe cli_transcribe.py test.wav --diarize --num_speakers 2
"""
import sys
import os
import argparse
import requests
import json

DEFAULT_URL = "http://127.0.0.1:8002/transcribe"

# 强制设置控制台编码，防止 Windows 打印乱码
try:
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except Exception:
    pass


def main():
    parser = argparse.ArgumentParser(description="FunClip Pro 命令行转写客户端")
    parser.add_argument("audio_path", help="音频文件路径 (WAV/MP3/FLAC 等)")
    parser.add_argument("--diarize", action="store_true", help="是否开启说话人角色分离")
    parser.add_argument("--strategy", default="seg_clustering",
                        choices=["seg_clustering", "vad_sliding", "spectral", "two_stage", "seg_cut_asr"],
                        help="说话人分离聚类策略 (默认: seg_clustering)")
    parser.add_argument("--num_speakers", type=int, default=None,
                        help="指定说话人人数 (可选)")
    parser.add_argument("--format", default="json", choices=["json", "text", "srt"],
                        help="输出格式：json(默认, 含详细段信息), text(纯文本), srt(字幕)")
    parser.add_argument("--url", default=DEFAULT_URL, help="API 服务地址")
    args = parser.parse_args()

    # 1. 路径预处理：去掉 Windows 复制路径可能带的双引号
    path = args.audio_path.strip().strip('"').strip("'")
    if not os.path.exists(path):
        print(f"❌ 错误：找不到音频文件 '{path}'")
        sys.exit(1)

    print(f"🎬 正在读取文件: {os.path.basename(path)}")
    print(f"⚙️  配置：diarize={args.diarize} | strategy={args.strategy} | num_speakers={args.num_speakers}")

    # 2. 检查后台服务
    try:
        requests.get(args.url.replace("/transcribe", "/"), timeout=2)
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到后台服务 ({args.url})。")
        print("💡 请先确保服务已启动 (双击运行 `一键启动_ASR_API服务.bat` 或在后台执行 `python asr_onnx_service.py`)。")
        sys.exit(1)
    except Exception:
        pass

    # 3. 发送转写请求
    print("🚀 正在发送音频至后台引擎进行推理，请稍候...")
    try:
        with open(path, "rb") as f:
            files = {"file": f}
            data = {
                "diarize": "true" if args.diarize else "false",
                "diarize_strategy": args.strategy,
                "response_format": args.format
            }
            if args.num_speakers is not None:
                data["num_speakers"] = str(args.num_speakers)

            resp = requests.post(args.url, files=files, data=data)
            
        if resp.status_code != 200:
            print(f"❌ 识别失败，服务返回 HTTP {resp.status_code}: {resp.text}")
            sys.exit(1)

        if args.format == "json":
            res = resp.json()
            print(f"✓ 推理成功！引擎: {res.get('engine')} | 端到端耗时: {res.get('latency_ms', 0):.1f}ms\n")

            print("=================== 🗣️  转写文本结果 ===================")
            if args.diarize:
                diarized_text = res.get("diarized_text", "")
                if diarized_text:
                    print(diarized_text)
                else:
                    # 兜底以 segments 渲染
                    for seg in res.get("segments", []):
                        spk = seg.get("speaker", "?")
                        txt = seg.get("text", "")
                        print(f"[说话人{spk}] {txt}")
            else:
                print(res.get("text", ""))
            print("======================================================")
        else:
            # text / srt: 服务返回纯文本
            print(f"✓ 推理成功！耗时: {resp.elapsed.total_seconds()*1000:.1f}ms\n")
            print(resp.text)

    except Exception as e:
        print(f"❌ 运行发生异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
