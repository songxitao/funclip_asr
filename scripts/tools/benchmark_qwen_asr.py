#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QwenASR 离线推理性能基准测试
用法: python benchmark_qwen_asr.py
"""

import base64
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

# ============ 配置 ============
API_HOST = "http://127.0.0.1:28000"
TESTSET_DIR = Path(r"E:\project\funclip-pro\testset")

# 寻找短音频: 兼容不同的 BAC 命名规则，取前10个
SHORT_FILES = (sorted(TESTSET_DIR.rglob("BAC009S0916W0*.wav")) or sorted(TESTSET_DIR.rglob("BAC*.wav")))[:10]
# 中等音频: ~60s mixed file
MEDIUM_FILES = sorted(TESTSET_DIR.rglob("R8002_M8002_mixed.wav"))
MEDIUM_FILE = MEDIUM_FILES[0] if MEDIUM_FILES else None

DIVIDER = "=" * 70


def get_audio_duration(filepath):
    """用文件大小估算 WAV 时长 (16bit 16kHz mono)"""
    size = os.path.getsize(filepath)
    # WAV header ~44 bytes, 16bit=2bytes, 16kHz
    return max(0, (size - 44)) / (16000 * 2)


def encode_b64(filepath):
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode()


def docker_snapshot():
    """采集容器内 GPU + 内存快照"""
    try:
        gpu = subprocess.run(
            ["docker", "exec", "qwen3-asr", "nvidia-smi",
             "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        mem = subprocess.run(
            ["docker", "exec", "qwen3-asr", "bash", "-c",
             "cat /proc/$(pgrep -f custom_server.py | head -n 1)/status 2>/dev/null | grep -E 'VmRSS|Threads' || echo 'N/A'"],
            capture_output=True, text=True, timeout=5
        )
        gpu_line = gpu.stdout.strip() if gpu.returncode == 0 else "N/A"
        mem_line = mem.stdout.strip().replace("\n", " | ") if mem.returncode == 0 else "N/A"
        return f"GPU: [{gpu_line}]  Process: [{mem_line}]"
    except Exception as e:
        return f"snapshot error: {e}"


def test_single(filepath, label=""):
    """单文件推理测试"""
    duration = get_audio_duration(filepath)
    b64 = encode_b64(filepath)

    print(f"\n{'─' * 50}")
    print(f"📄 {label} | {filepath.name} | 音频时长: {duration:.2f}s")

    snap_before = docker_snapshot()
    print(f"   ⏱️ 推理前: {snap_before}")

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{API_HOST}/v1/audio/transcriptions",
            json={"audio_base64": b64, "language": "Chinese", "return_timestamps": True},
            timeout=120,
        )
        elapsed = time.perf_counter() - t0
        resp.raise_for_status()
        data = resp.json()
        text = data.get("text", "")[:80]
        rtf = elapsed / duration if duration > 0 else float("inf")

        snap_after = docker_snapshot()
        print(f"   ✅ 耗时: {elapsed:.3f}s | RTF: {rtf:.3f} | 文本: {text}")
        print(f"   ⏱️ 推理后: {snap_after}")
        return {"file": filepath.name, "duration": duration, "elapsed": elapsed, "rtf": rtf, "ok": True}
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"   ❌ 失败 ({elapsed:.2f}s): {e}")
        return {"file": filepath.name, "duration": duration, "elapsed": elapsed, "rtf": -1, "ok": False}


def test_batch(files, label=""):
    """批量推理测试 (Base64 传输)"""
    total_duration = sum(get_audio_duration(f) for f in files)
    b64_list = [encode_b64(f) for f in files]

    print(f"\n{'─' * 50}")
    print(f"📦 {label} | {len(files)} 文件 | 总音频时长: {total_duration:.2f}s")

    snap_before = docker_snapshot()
    print(f"   ⏱️ 推理前: {snap_before}")

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{API_HOST}/v1/audio/batch_transcriptions",
            json={"audio_batch_base64": b64_list, "language": "Chinese", "return_timestamps": True},
            timeout=300,
        )
        elapsed = time.perf_counter() - t0
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        rtf = elapsed / total_duration if total_duration > 0 else float("inf")

        snap_after = docker_snapshot()
        print(f"   ✅ 耗时: {elapsed:.3f}s | RTF: {rtf:.3f} | 返回 {len(results)} 条结果")
        for i, r in enumerate(results[:3]):
            print(f"      [{i}] {r.get('text', '')[:60]}")
        if len(results) > 3:
            print(f"      ... 还有 {len(results)-3} 条")
        print(f"   ⏱️ 推理后: {snap_after}")
        return {"label": label, "files": len(files), "duration": total_duration, "elapsed": elapsed, "rtf": rtf, "ok": True}
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"   ❌ 失败 ({elapsed:.2f}s): {e}")
        return {"label": label, "files": len(files), "duration": total_duration, "elapsed": elapsed, "rtf": -1, "ok": False}


def test_batch_shared(files, label=""):
    """批量推理测试 (共享卷直读模式)"""
    import shutil
    import uuid

    total_duration = sum(get_audio_duration(f) for f in files)
    
    # 物理共享目录
    shared_host_dir = Path(r"E:\project\funclip-pro\qwen_server\shared_tmp")
    shared_host_dir.mkdir(parents=True, exist_ok=True)
    shared_docker_dir = "/app/server/shared_tmp"
    
    copied_paths = []
    docker_paths = []
    
    for f in files:
        ext = f.suffix
        filename = f"bench_{uuid.uuid4()}{ext}"
        dest_path = shared_host_dir / filename
        shutil.copy2(f, dest_path)
        copied_paths.append(dest_path)
        
        d_path = f"{shared_docker_dir}/{filename}"
        docker_paths.append(d_path)
        
    print(f"\n{'─' * 50}")
    print(f"📦 {label} | {len(files)} 文件 | 总音频时长: {total_duration:.2f}s")

    snap_before = docker_snapshot()
    print(f"   ⏱️ 推理前: {snap_before}")

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{API_HOST}/v1/audio/batch_transcriptions",
            json={"audio_paths": docker_paths, "language": "Chinese", "return_timestamps": True},
            timeout=300,
        )
        elapsed = time.perf_counter() - t0
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        rtf = elapsed / total_duration if total_duration > 0 else float("inf")

        snap_after = docker_snapshot()
        print(f"   ✅ 耗时: {elapsed:.3f}s | RTF: {rtf:.3f} | 返回 {len(results)} 条结果")
        for i, r in enumerate(results[:3]):
            print(f"      [{i}] {r.get('text', '')[:60]}")
        if len(results) > 3:
            print(f"      ... 还有 {len(results)-3} 条")
        print(f"   ⏱️ 推理后: {snap_after}")
        return {"label": label, "files": len(files), "duration": total_duration, "elapsed": elapsed, "rtf": rtf, "ok": True}
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"   ❌ 失败 ({elapsed:.2f}s): {e}")
        return {"label": label, "files": len(files), "duration": total_duration, "elapsed": elapsed, "rtf": -1, "ok": False}
    finally:
        for p in copied_paths:
            if p.exists():
                try: p.unlink()
                except: pass


def test_long_pipeline(filepath):
    """端到端长音频 VAD + 共享卷直读转写测试"""
    if not filepath or not filepath.exists():
        print(f"\n❌ 未找到长音频文件: {filepath}")
        return None
        
    duration = get_audio_duration(filepath)
    print(f"\n{'─' * 50}")
    print(f"🎬 阶段 4: 端到端长音频 OfflinePipeline 测试 | {filepath.name} | 时长: {duration:.2f}s")
    
    # 将 SDK 的 src 加入 python 路径以使用我们刚优化的 QwenEngine
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from funclip_pro.pipeline.offline import OfflinePipeline
    
    snap_before = docker_snapshot()
    print(f"   ⏱️ 推理前: {snap_before}")

    t0 = time.perf_counter()
    try:
        pipeline = OfflinePipeline(auto_load=True)
        # 运行转写，默认启用 VAD 并采用 QwenEngine 直读
        raw_text, engine, segments, diarized = pipeline.run(
            str(filepath), vad_strategy="auto", engine="qwen", language="zh"
        )
        elapsed = time.perf_counter() - t0
        rtf = elapsed / duration if duration > 0 else float("inf")

        snap_after = docker_snapshot()
        print(f"   ✅ 耗时: {elapsed:.3f}s | RTF: {rtf:.3f} | 转写字数: {len(raw_text)} | 产生 VAD 对齐片段: {len(segments)}")
        print(f"   📢 识别结果示例(前80字): {raw_text[:80]}")
        print(f"   ⏱️ 推理后: {snap_after}")
        return {"label": "端到端长音频Pipeline", "duration": duration, "elapsed": elapsed, "rtf": rtf, "ok": True}
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"   ❌ 失败 ({elapsed:.2f}s): {e}")
        import traceback
        traceback.print_exc()
        return {"label": "端到端长音频Pipeline", "duration": duration, "elapsed": elapsed, "rtf": -1, "ok": False}


def main():
    print(DIVIDER)
    print("🔬 QwenASR 离线推理性能与共享卷优化对比测试")
    print(f"   API: {API_HOST}")
    print(f"   时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(DIVIDER)

    # 0. 健康检查
    print("\n🏥 健康检查...")
    try:
        r = requests.get(f"{API_HOST}/docs", timeout=5)
        print(f"   服务状态: HTTP {r.status_code}")
    except Exception as e:
        print(f"   ❌ 服务不可达: {e}")
        print("   请先启动 Qwen3-ASR Docker 服务!")
        sys.exit(1)

    all_results = []

    # 1. 预热
    print(f"\n{DIVIDER}")
    print("🔥 阶段 0: 预热 (首次推理)")
    if SHORT_FILES:
        warmup = test_single(SHORT_FILES[0], label="预热")
        all_results.append(warmup)
    else:
        print("   ❌ 没有找到短音频文件用于预热！")

    # 2. 单文件短音频
    print(f"\n{DIVIDER}")
    print("📋 阶段 1: 单文件短音频 (~3-7s)")
    for f in SHORT_FILES[1:6]:
        r = test_single(f, label="短音频")
        all_results.append(r)

    # 3. 单文件中等音频
    if MEDIUM_FILE and MEDIUM_FILE.exists():
        print(f"\n{DIVIDER}")
        print("📋 阶段 2: 单文件中等音频 (~60s)")
        r = test_single(MEDIUM_FILE, label="中等音频")
        all_results.append(r)

    # 4. 批量测试对比 (Base64 vs 共享卷直读)
    if SHORT_FILES:
        print(f"\n{DIVIDER}")
        print("📋 阶段 3: 批量测试对比 (10个短音频)")
        b64_res = test_batch(SHORT_FILES[:10], label="批量10 (Base64)")
        shared_res = test_batch_shared(SHORT_FILES[:10], label="批量10 (共享卷直读)")
        all_results.append(b64_res)
        all_results.append(shared_res)

    # 5. 端到端长音频转写测试
    # 尝试读取 34 分钟的中长音频，如果不存在，则尝试在 testset 下匹配任何大于 5MB 的 wav
    long_file = TESTSET_DIR / "ali_near_prep" / "R8002_M8002_mixed.wav"
    if not long_file.exists():
        import glob
        wavs = glob.glob(str(TESTSET_DIR / "**" / "*.wav"), recursive=True)
        large_wavs = [Path(w) for w in wavs if os.path.getsize(w) > 5*1024*1024]
        if large_wavs:
            long_file = large_wavs[0]
            
    if long_file and long_file.exists():
        long_res = test_long_pipeline(long_file)
        all_results.append(long_res)
    else:
        print(f"\n{DIVIDER}")
        print("⚠️ 阶段 4 跳过：未在 testset 中找到大于 5MB 的中长音频。")

    # 6. 汇总
    print(f"\n{DIVIDER}")
    print("📊 性能与吞吐对比汇总")
    print(f"{'─' * 50}")
    
    # 打印批量对比
    b64_item = next((r for r in all_results if r and r.get("label") == "批量10 (Base64)" and r.get("ok")), None)
    shared_item = next((r for r in all_results if r and r.get("label") == "批量10 (共享卷直读)" and r.get("ok")), None)
    if b64_item and shared_item:
        speedup = (b64_item["elapsed"] - shared_item["elapsed"]) / b64_item["elapsed"] * 100
        print(f"   批量10 (Base64) 耗时: {b64_item['elapsed']:.3f}s (RTF: {b64_item['rtf']:.3f})")
        print(f"   批量10 (共享卷) 耗时: {shared_item['elapsed']:.3f}s (RTF: {shared_item['rtf']:.3f})")
        print(f"   🚀 共享卷直读比 Base64 节省了: {speedup:.1f}% 的开销时间！")
        print(f"{'─' * 50}")
        
    long_item = next((r for r in all_results if r and r.get("label") == "端到端长音频Pipeline" and r.get("ok")), None)
    if long_item:
        print(f"   端到端长音频 (时长: {long_item['duration']:.1f}s) 推理耗时: {long_item['elapsed']:.3f}s")
        print(f"   🚀 最终端到端吞吐 RTF: {long_item['rtf']:.3f} (即约 {1/long_item['rtf']:.1f} 倍速转写！)")
    
    print(f"\n{DIVIDER}")
    print("✅ 基准对比测试完成")


if __name__ == "__main__":
    main()
