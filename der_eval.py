#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AISHELL-4 说话人分离 DER 评测器（纯 Python 实现，零额外依赖）

流程:
  1. 对每条会议音频：8 声道 flac -> 抽取 ch0 转单声道 flac（与 RTTM 参考信号对齐，且 <50MB 上限）
  2. POST /transcribe?diarize=true -> 取 segments（start/end 单位 ms，speaker 为整数）
  3. 解析参考 RTTM -> 标准 DER：0.25s collar + 贪心说话人映射，统计 Missed/FalseAlarm/Confusion
  4. 输出每文件 DER 与全局（按参考语音时长加权）DER

DER 定义: DER = (Missed + FalseAlarm + Confusion) / 总参考语音时长（collar 内帧不计入）

用法:
  # 单条会议（验证管线，约 40min 音频，推理数分钟）
  python der_eval.py \
      --audio_dir E:/project/funclip-pro/testset/dia-aishell4-test/audio/test \
      --rttm_dir  E:/project/funclip-pro/testset/dia-aishell4-test/rttm/test \
      --limit 1 --out test_results/der_single.json

  # 全量 20 条（可选，耗时约 1-2h）
  python der_eval.py ... --limit 0 --out test_results/der_full.json
"""
import argparse
import os
import sys
import glob
import json
import time
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def _select_best_channel(audio, sr):
    """从多通道音频中选择能量/SNR最高的通道。

    Args:
        audio: (samples, channels) numpy 数组
        sr: 采样率

    Returns:
        (channel_index, channel_data): 最优通道索引 (int) 和该通道的 (samples,) 数组
    """
    if audio.ndim == 1 or audio.shape[1] == 1:
        return 0, (audio if audio.ndim == 1 else audio[:, 0])

    energies = [np.mean(audio[:, ch] ** 2) for ch in range(audio.shape[1])]
    best_ch = int(np.argmax(energies))
    print(f"[der_eval] selected channel {best_ch}/{audio.shape[1]} "
          f"(energy={energies[best_ch]:.6f}, best SNR)")
    return best_ch, audio[:, best_ch]


def to_mono_ch0(path, target_sr=16000):
    """多声道音频 -> 自动选择能量最高通道转单声道，写临时 mono flac，返回 (path, duration_sec)。"""
    data, sr = sf.read(path, always_2d=True)
    if data.shape[1] > 1:
        _, mono = _select_best_channel(data, sr)
    else:
        mono = data[:, 0]
    if sr != target_sr:
        import librosa
        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    data_out = mono.reshape(-1, 1)
    tmp = tempfile.NamedTemporaryFile(suffix=".flac", delete=False)
    sf.write(tmp.name, data_out, sr, format="FLAC")
    return tmp.name, len(mono) / sr


def parse_rttm(path):
    segs = []
    spks = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 9 or p[0] != "SPEAKER":
                continue
            start = float(p[3])
            dur = float(p[4])
            spk = p[7]
            segs.append((start, start + dur, spk))
            spks.add(spk)
    return segs, spks


def greedy_map(hyp_segs, ref_segs):
    """贪心说话人映射：每个 hyp 说话人映射到与其重叠最多的 ref 说话人。"""
    hyp_spks = sorted({s for (_, _, s) in hyp_segs})
    ref_spks = sorted({s for (_, _, s) in ref_segs})
    overlap = {h: {r: 0.0 for r in ref_spks} for h in hyp_spks}
    for (hs, he, h) in hyp_segs:
        for (rs, re, r) in ref_segs:
            ov = max(0.0, min(he, re) - max(hs, rs))
            if ov > 0:
                overlap[h][r] += ov
    mapping = {}
    for h in hyp_spks:
        if ref_spks:
            mapping[h] = max(ref_spks, key=lambda r: overlap[h][r])
        else:
            mapping[h] = None
    return mapping


def compute_der(ref_segs, hyp_segs, duration, collar=0.25, step=0.01):
    mapping = greedy_map(hyp_segs, ref_segs)
    N = int(duration / step) + 1
    ref_arr = [-1] * N
    hyp_arr = [-1] * N
    ignore = [False] * N

    for (s, e, spk) in ref_segs:
        a, b = int(s / step), min(int(e / step), N)
        for i in range(a, b):
            ref_arr[i] = spk
        for bd in (s, e):
            ca, cb = int((bd - collar) / step), int((bd + collar) / step)
            for i in range(max(0, ca), min(cb, N)):
                ignore[i] = True

    for (s, e, spk) in hyp_segs:
        a, b = int(s / step), min(int(e / step), N)
        m = mapping.get(spk)
        for i in range(a, b):
            hyp_arr[i] = m if m is not None else -2

    FA = MISS = CONF = REF = 0
    for i in range(N):
        if ignore[i]:
            continue
        r = ref_arr[i]
        h = hyp_arr[i]
        if r == -1:
            if h != -1:
                FA += 1
        else:
            REF += 1
            if h == -1 or h == -2:
                MISS += 1
            elif h == r:
                pass
            else:
                CONF += 1
    der = (FA + MISS + CONF) / REF if REF > 0 else 0.0
    return der, {"FA": FA, "MISS": MISS, "CONF": CONF, "REF": REF,
                 "mapping": {str(k): v for k, v in mapping.items()}}


def main():
    ap = argparse.ArgumentParser(description="AISHELL-4 DER 评测（纯 Python）")
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--rttm_dir", required=True)
    ap.add_argument("--service_url", default="http://localhost:8002")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--collar", type=float, default=0.25)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--out", default="test_results/der_single.json")
    ap.add_argument("--diarize_strategy", default="spectral",
                    choices=["single", "two_stage", "spectral"],
                    help="说话人聚类策略（默认 spectral）")
    args = ap.parse_args()

    rttm_files = sorted(glob.glob(os.path.join(args.rttm_dir, "**", "*.rttm"), recursive=True))
    audio_files = sorted(
        glob.glob(os.path.join(args.audio_dir, "**", "*.flac"), recursive=True)
        + glob.glob(os.path.join(args.audio_dir, "**", "*.wav"), recursive=True)
    )
    rttm_map = {Path(f).stem: f for f in rttm_files}
    pairs = [(af, rttm_map[Path(af).stem]) for af in audio_files if Path(af).stem in rttm_map]
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"[info] 匹配 {len(pairs)} 对 audio/rttm | limit={args.limit}")

    import requests
    results = []
    gFA = gMISS = gCONF = gREF = 0
    for (af, rf) in pairs:
        print(f"[info] 处理 {Path(af).name} ...", flush=True)
        mono_path, dur = to_mono_ch0(af)
        ref_segs, ref_spks = parse_rttm(rf)  # 先解析参考，拿 oracle 说话人数
        t0 = time.time()
        try:
            with open(mono_path, "rb") as fh:
                r = requests.post(
                    args.service_url + "/transcribe",
                    files={"file": fh},
                    data={"diarize": "true", "response_format": "json",
                          "diarize_strategy": args.diarize_strategy,
                          "num_speakers": len(ref_spks)},  # oracle-K：用 RTTM 真实人数强制收敛
                    timeout=args.timeout,
                )
            segs = r.json().get("segments", [])
            hyp_segs = [(s["start"] / 1000.0, s["end"] / 1000.0, str(s["speaker"])) for s in segs]
            from collections import Counter
            spk_counter = Counter(str(s["speaker"]) for s in segs)
            print(f"  [debug] hyp speaker 分布: {dict(spk_counter)}", flush=True)
            max_end = max(
                [e for (_, e, _) in ref_segs] + [e for (_, e, _) in hyp_segs] + [dur]
            )
            der, detail = compute_der(ref_segs, hyp_segs, max_end, args.collar, args.step)
            gFA += detail["FA"]; gMISS += detail["MISS"]
            gCONF += detail["CONF"]; gREF += detail["REF"]
            rec = {
                "file": Path(af).name, "duration": round(dur, 1),
                "num_hyp_segs": len(hyp_segs), "num_ref_spks": len(ref_spks),
                "DER": der, "detail": {k: detail[k] for k in ("FA", "MISS", "CONF", "REF")},
                "mapping": detail["mapping"], "latency_s": round(time.time() - t0, 1),
            }
            results.append(rec)
            print(f"  -> DER={der*100:.2f}% | ref_spks={len(ref_spks)} hyp_segs={len(hyp_segs)} "
                  f"| FA={detail['FA']} MISS={detail['MISS']} CONF={detail['CONF']} REF={detail['REF']} "
                  f"| 耗时{rec['latency_s']}s", flush=True)
        except Exception as e:
            print(f"  [ERROR] {Path(af).name}: {e}", flush=True)
            results.append({"file": Path(af).name, "error": str(e)})
        finally:
            try:
                os.remove(mono_path)
            except Exception:
                pass

    gder = (gFA + gMISS + gCONF) / gREF if gREF > 0 else 0.0
    summary = {"global_DER": gder,
               "global": {"FA": gFA, "MISS": gMISS, "CONF": gCONF, "REF": gREF},
               "per_file": results}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n===== DER 评测结果 =====")
    print(f"全局 DER: {gder*100:.2f}%")
    print(f"FA={gFA} MISS={gMISS} CONF={gCONF} REF={gREF}")


if __name__ == "__main__":
    main()
