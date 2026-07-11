# -*- coding: utf-8 -*-
"""
AliMeeting near 单场 DER 评测（试跑）。
混合单通道 wav -> flac(规避50MB) -> POST :8002/transcribe?diarize=true -> DER
用 conda asr_ui_env python 跑。
用法: python ali_der_eval.py <场ID>
"""
import sys, json, time, requests, soundfile as sf
from pathlib import Path
from der_eval import compute_der, parse_rttm

PREP = Path(r"E:\project\funclip-pro\testset\ali_near_prep")
URL = "http://localhost:8002/transcribe"


def eval_one(session, strategy="spectral"):
    mixed_wav = PREP / f"{session}_mixed.wav"
    rttm_path = PREP / f"{session}.rttm"
    if not mixed_wav.exists():
        print(f"找不到 {mixed_wav}，先跑 ali_near_prep.py"); return

    # 1. wav -> flac（规避 50MB 上传上限）
    data, sr = sf.read(str(mixed_wav))
    flac = PREP / f"{session}_mixed.flac"
    sf.write(str(flac), data, sr, format="FLAC")
    print(f"[{session}] flac={flac.stat().st_size/1e6:.1f}MB 时长={len(data)/sr:.1f}s")

    # 2. 解析参考 RTTM
    ref_segs, ref_spks = parse_rttm(str(rttm_path))
    n_spk = len(ref_spks)
    print(f"[{session}] ref: {len(ref_segs)}段 {n_spk}人 {sorted(ref_spks)}")

    # 3. POST 服务
    t0 = time.time()
    with open(flac, "rb") as f:
        resp = requests.post(URL,
            files={"file": (flac.name, f, "audio/flac")},
            data={"diarize": "true", "num_speakers": str(n_spk),
                  "vad_strategy": "always", "diarize_strategy": strategy},
            timeout=3600)
    latency = time.time() - t0
    if resp.status_code != 200:
        print(f"HTTP {resp.status_code}: {resp.text[:300]}"); return
    result = resp.json()
    segs = result.get("segments", [])
    print(f"[{session}] hyp: {len(segs)}段 latency={latency:.1f}s engine={result.get('engine')}")

    # 4. hyp segments -> (start_sec, end_sec, speaker)，start/end 是 ms
    hyp_segs = []
    for s in segs:
        spk = str(s.get("speaker", "?"))
        if spk in ("?", "None", ""):
            continue
        hyp_segs.append((s["start"] / 1000.0, s["end"] / 1000.0, spk))

    # 5. 算 DER
    duration = len(data) / sr
    der, detail = compute_der(ref_segs, hyp_segs, duration, collar=0.25, step=0.01)
    print(f"\n===== [{session}] DER 结果 =====")
    print(f"DER = {der*100:.2f}%")
    print(f"FA={detail['FA']} MISS={detail['MISS']} CONF={detail['CONF']} REF={detail['REF']}")
    print(f"  FA占比={detail['FA']/detail['REF']*100:.1f}% "
          f"MISS占比={detail['MISS']/detail['REF']*100:.1f}% "
          f"CONF占比={detail['CONF']/detail['REF']*100:.1f}%")

    # 6. hyp 说话人分布
    from collections import Counter
    spk_dur = Counter()
    for st, en, sp in hyp_segs:
        spk_dur[sp] += (en - st)
    print(f"hyp 说话人时长分布: {dict(sorted(spk_dur.items()))}")

    return der, detail


if __name__ == "__main__":
    session = sys.argv[1] if len(sys.argv) > 1 else "R8002_M8002"
    eval_one(session)
