# -*- coding: utf-8 -*-
"""FunClip Pro ASR & 说话人分离命令行客户端 (CLI Thin Client)

薄客户端：仅负责命令行参数解析与最终输出渲染；实际转写委托给
funclip_pro.pipeline.OfflinePipeline（等价原 asr_onnx_service._run_inference
收口层，已下沉核心推理/对齐/SRT 逻辑到 funclip_pro.core / funclip_pro.utils）。

用法：
    E:\conda\envs\asr_ui_env\python.exe cli_transcribe.py <音频路径> [参数]

示例：
    E:\conda\envs\asr_ui_env\python.exe cli_transcribe.py test.wav --diarize
    E:\conda\envs\asr_ui_env\python.exe cli_transcribe.py test.wav --diarize --num_speakers 2

红线：
    - 时间戳统一 ms（由 core / pipeline 保证，客户端不重复实现）
    - 零硬编码盘符/绝对路径（src 经相对 __file__ 注入 sys.path）
    - 模块间绝对导入：from funclip_pro.x import Y
"""
import os
import sys
import argparse

# 2. DLL 补丁：动态点亮 onnxruntime / torch 推理（必须在首次加载重型库前）
from funclip_pro.config.loader import apply_dll_patch
apply_dll_patch()

# 3. 统一转写流水线（收口层：VAD 三态 + 引擎路由 + 说话人分离 + SRT 组装）
from funclip_pro.pipeline import OfflinePipeline
# SRT 响应组装复用 funclip_pro.utils 工具（等价于原 _segments_to_srt / _merge_same_speaker_segments）
from funclip_pro.utils import _segments_to_srt, _merge_same_speaker_segments

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
                        choices=["seg_clustering", "vad_sliding", "spectral", "two_stage"],
                        help="说话人分离聚类策略 (默认: seg_clustering)")
    parser.add_argument("--num_speakers", type=int, default=None,
                        help="指定说话人人数 (可选)")
    parser.add_argument("--format", default="json", choices=["json", "text", "srt"],
                        help="输出格式：json(默认, 含详细段信息), text(纯文本), srt(字幕)")
    parser.add_argument("--url", default="http://127.0.0.1:8002/transcribe",
                        help="(保留参数，兼容旧调用；本地引擎推理已不再依赖后台服务)")
    args = parser.parse_args()

    # 1. 路径预处理：去掉 Windows 复制路径可能带的双引号
    path = args.audio_path.strip().strip('"').strip("'")
    if not os.path.exists(path):
        print(f"❌ 错误：找不到音频文件 '{path}'")
        sys.exit(1)

    print(f"🎬 正在读取文件: {os.path.basename(path)}")
    print(f"⚙️  配置：diarize={args.diarize} | strategy={args.strategy} | num_speakers={args.num_speakers}")

    # 2. 实例化本地流水线（加载 Sherpa-ASR / VAD / PUNC 模型，等价原服务启动钩子）
    print("🚀 正在本地引擎推理，请稍候...")
    try:
        pipeline = OfflinePipeline(auto_load=True)

        start_time = __import__("time").time()
        # 返回四元组 (raw_text, engine_key, segments, diarized_text)
        text, engine_key, segments, diarized_text = pipeline.run(
            path,
            vad_strategy="auto",
            diarize=args.diarize,
            diarize_strategy=args.strategy,
            num_speakers=args.num_speakers,
        )
        latency = (__import__("time").time() - start_time) * 1000
    except Exception as e:
        print(f"❌ 运行发生异常: {e}")
        sys.exit(1)

    # 3. 组装输出（字段/排版等价原 cli_transcribe.py，原 HTTP 服务字段直接复用四元组）
    if args.format == "json":
        print(f"✓ 推理成功！引擎: {engine_key} | 端到端耗时: {latency:.1f}ms\n")

        print("=================== 🗣️  转写文本结果 ===================")
        if args.diarize:
            if diarized_text:
                print(diarized_text)
            else:
                # 兜底以 segments 渲染
                for seg in segments:
                    spk = seg.get("speaker", "?")
                    txt = seg.get("text", "")
                    print(f"[说话人{spk}] {txt}")
        else:
            print(text)
        print("======================================================")
    else:
        # text / srt: 纯文本输出（等价原服务 PlainTextResponse）
        print(f"✓ 推理成功！耗时: {latency:.1f}ms\n")

        if args.format == "text":
            out = diarized_text if (args.diarize and diarized_text) else text
        else:  # srt
            if args.diarize and segments:
                merged = _merge_same_speaker_segments(segments)
                out = _segments_to_srt(merged)
            else:
                # 非说话人分离模式：整段文字作为一条字幕（无时间戳信息）
                out = f"1\n00:00:00,000 --> 00:00:00,000\n{text.strip()}\n" if text.strip() else ""
        print(out)


if __name__ == "__main__":
    main()
