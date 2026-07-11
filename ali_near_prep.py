# -*- coding: utf-8 -*-
"""
AliMeeting near 预处理验证脚本（单场试跑）。
作用：每场 3 个领夹麦(near) wav -> 相加成单通道混合 wav（模拟1个近场麦收3人）
      每场 3 个 TextGrid -> 合并成参考 RTTM（谁在何时说话）
零第三方依赖，仅用标准库 wave/array/re。
用法：python ali_near_prep.py <场ID如 R8002_M8002>
"""
import wave
import array
import re
import sys
import os
from pathlib import Path

BASE = Path(r"E:\project\funclip-pro\testset\Test_Ali\Test_Ali\Test_Ali_near")
OUT = Path(r"E:\project\funclip-pro\testset\ali_near_prep")
OUT.mkdir(parents=True, exist_ok=True)


def read_wav_mono(path):
    """用标准库 wave 读单声道 16bit PCM，返回 (samples:list[int], sr:int)。"""
    with wave.open(str(path), "rb") as w:
        nch = w.getnchannels()
        sr = w.getframerate()
        sw = w.getsampwidth()
        n = w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise RuntimeError(f"{path} 非声明16bit(sampwidth={sw})，需 soundfile")
    arr = array.array("h")
    arr.frombytes(raw)
    if nch > 1:  # 多声道取平均
        mono = array.array("h", [0] * (len(arr) // nch))
        for i in range(len(mono)):
            s = sum(arr[i * nch + c] for c in range(nch)) // nch
            mono[i] = s
        return list(mono), sr
    return list(arr), sr


def mix_to_mono(wav_paths):
    """多路 wav 相加 -> 单通道（对齐到最长，短的补0，相加后除路数防瀑音）。"""
    signals = []
    sr = None
    for p in wav_paths:
        s, this_sr = read_wav_mono(p)
        if sr is None:
            sr = this_sr
        elif this_sr != sr:
            raise RuntimeError(f"采样率不一致 {sr} vs {this_sr}")
        signals.append(s)
    max_len = max(len(s) for s in signals)
    n = len(signals)
    mixed = [0] * max_len
    for s in signals:
        for i, v in enumerate(s):
            mixed[i] += v
    # 除路数平均 + clip 到 int16
    out = array.array("h")
    clip = 32767
    floor = -32768
    for v in mixed:
        avg = v // n
        out.append(clip if avg > clip else (floor if avg < floor else avg))
    return out, sr


def write_wav_mono(path, samples, sr):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(array.array("h", samples).tobytes())


def parse_textgrid(path):
    """解析 TextGrid，返回 [(xmin, xmax, text), ...]。只取 interval 块内紧邻的 xmin/xmax/text。"""
    txt = Path(path).read_text(encoding="utf-8")
    intervals = []
    # 关键：用 \s* 要求 xmin/xmax/text 三者紧邻（仅空白分隔），
    # 避免误匹配文件头/tier头的 xmin=0 / xmax=总时长。
    pat = re.compile(r"xmin\s*=\s*([\d.]+)\s*xmax\s*=\s*([\d.]+)\s*text\s*=\s*\"(.*?)\"",
                     re.DOTALL)
    for m in pat.finditer(txt):
        text = m.group(3)
        if not text.strip():  # 空文本=静音段，跳过
            continue
        intervals.append((float(m.group(1)), float(m.group(2)), text))
    return intervals


def spk_id_from_name(name):
    """R8002_M8002_N_SPK8005.wav -> SPK8005"""
    m = re.search(r"SPK(\d+)", name)
    return f"SPK{m.group(1)}" if m else name


def build_rttm(session_id, tg_paths, out_path):
    """合并多说话人 TextGrid -> 单个 RTTM。"""
    lines = []
    for tp in tg_paths:
        spk = spk_id_from_name(tp.name)
        for xmin, xmax, text in parse_textgrid(tp):
            if xmax <= xmin:
                continue
            dur = xmax - xmin
            # SPEAKER <file> 1 <start> <dur> <NA> <NA> <spk> <NA> <NA>
            lines.append(f"SPEAKER {session_id} 1 {xmin:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>")
    lines.sort(key=lambda x: float(x.split()[3]))
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main():
    session = sys.argv[1] if len(sys.argv) > 1 else "R8002_M8002"
    wav_dir = BASE / "audio_dir"
    tg_dir = BASE / "textgrid_dir"

    wavs = sorted(wav_dir.glob(f"{session}_N_SPK*.wav"))
    tgs = sorted(tg_dir.glob(f"{session}_N_SPK*.TextGrid"))
    print(f"场 {session}: 找到 {len(wavs)} 个 near wav, {len(tgs)} 个 TextGrid")
    if len(wavs) < 2:
        print("不足2路，跳过"); return

    # 1. 混音
    mixed, sr = mix_to_mono(wavs)
    mix_wav = OUT / f"{session}_mixed.wav"
    write_wav_mono(mix_wav, mixed, sr)
    dur = len(mixed) / sr
    print(f"混音 -> {mix_wav.name} | 采样率={sr} 时长={dur:.1f}s 帧数={len(mixed)}")

    # 2. TextGrid -> RTTM
    rttm_path = OUT / f"{session}.rttm"
    n_lines = build_rttm(session, tgs, rttm_path)
    # 统计说话人数
    spks = set()
    for p in tgs:
        spks.add(spk_id_from_name(p.name))
    print(f"RTTM -> {rttm_path.name} | 行数={n_lines} 说话人数={len(spks)} {sorted(spks)}")

    # 3. 抽样校验 RTTM 前5行
    print("--- RTTM 前5行 ---")
    for ln in Path(rttm_path).read_text(encoding="utf-8").splitlines()[:5]:
        print("  " + ln)
    print(f"\n验证通过。产物在 {OUT}")


if __name__ == "__main__":
    main()
