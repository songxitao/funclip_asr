# -*- coding: utf-8 -*-
"""FunClip Pro CLI 命令行入口 — 供 pyproject.toml [project.scripts] 反射调用。

用法（安装后）：
    funclip-pro transcribe <audio> [--diarize] [--format json|text|srt]
"""
import os
import sys
import argparse

# DLL 补丁：动态点亮 onnxruntime / torch 推理
from funclip_pro.config.loader import apply_dll_patch
apply_dll_patch()

from funclip_pro.pipeline import OfflinePipeline
from funclip_pro.utils import _segments_to_srt, _merge_same_speaker_segments


def _main():
    """CLI 核心逻辑（供 cli_transcribe.py 和 main() 共同调用）。"""
    parser = argparse.ArgumentParser(description="FunClip Pro 命令行转写客户端")
    parser.add_argument("audio_path", help="音频文件路径 (WAV/MP3/FLAC 等)")
    parser.add_argument("--diarize", action="store_true", help="是否开启说话人角色分离")
    parser.add_argument("--strategy", default="seg_clustering",
                        choices=["seg_clustering", "vad_sliding", "spectral", "two_stage"],
                        help="说话人分离聚类策略 (默认: seg_clustering)")
    parser.add_argument("--num_speakers", type=int, default=None,
                        help="指定说话人人数 (可选)")
    parser.add_argument("--format", default="json", choices=["json", "text", "srt"],
                        help="输出格式：json(默认), text(纯文本), srt(字幕)")
    args = parser.parse_args()

    path = args.audio_path.strip().strip('"').strip("'")
    if not os.path.exists(path):
        print(f"错误：找不到音频文件 '{path}'")
        sys.exit(1)

    print(f"正在读取文件: {os.path.basename(path)}")
    print(f"配置：diarize={args.diarize} | strategy={args.strategy} | num_speakers={args.num_speakers}")

    print("正在本地引擎推理，请稍候...")
    try:
        pipeline = OfflinePipeline(auto_load=True)
        start_time = __import__("time").time()
        text, engine_key, segments, diarized_text = pipeline.run(
            path,
            vad_strategy="auto",
            diarize=args.diarize,
            diarize_strategy=args.strategy,
            num_speakers=args.num_speakers,
        )
        latency = (__import__("time").time() - start_time) * 1000
    except Exception as e:
        print(f"运行发生异常: {e}")
        sys.exit(1)

    if args.format == "json":
        print(f"推理成功！引擎: {engine_key} | 端到端耗时: {latency:.1f}ms\n")
        print("转写文本结果:")
        if args.diarize:
            if diarized_text:
                print(diarized_text)
            else:
                for seg in segments:
                    spk = seg.get("speaker", "?")
                    txt = seg.get("text", "")
                    print(f"[说话人{spk}] {txt}")
        else:
            print(text)
    else:
        print(f"推理成功！耗时: {latency:.1f}ms\n")
        if args.format == "text":
            out = diarized_text if (args.diarize and diarized_text) else text
        else:
            if args.diarize and segments:
                merged = _merge_same_speaker_segments(segments)
                out = _segments_to_srt(merged)
            else:
                out = f"1\n00:00:00,000 --> 00:00:00,000\n{text.strip()}\n" if text.strip() else ""
        print(out)


def main():
    """CLI 入口 — 供 pyproject.toml [project.scripts] 反射调用。"""
    _main()


if __name__ == "__main__":
    main()
