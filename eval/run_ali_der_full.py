# -*- coding: utf-8 -*-
"""AliMeeting near 全量 20 场 DER 评测。

流程：每场混音+RTTM(ali_near_prep) -> POST :8002?diarize_strategy=sliding -> DER -> 加权平均。

用法:
    python eval/run_ali_der_full.py [--strategy sliding|spectral|two_stage]

前置：
    1) :8002 服务在跑（含 sliding 改动）
    2) testset/Test_Ali 已下
"""
import sys
import json
import time
from pathlib import Path

from ali_near_prep import BASE as NEAR_BASE, OUT as PREP_OUT, mix_to_mono, write_wav_mono, build_rttm
from ali_der_eval import eval_one

# 迁移后：脚本在 eval/，项目根在其父目录
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

STRATEGY = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--strategy" else "sliding"


def list_sessions():
    wav_dir = NEAR_BASE / "audio_dir"
    sessions = sorted(set(p.name.split("_N_")[0] for p in wav_dir.glob("*_N_SPK*.wav")))
    return sessions


def prep_session(session):
    """混音 + RTTM（复用 ali_near_prep 逻辑），返回 (mixed_wav, rttm_path)。"""
    wavs = sorted((NEAR_BASE / "audio_dir").glob(f"{session}_N_SPK*.wav"))
    tgs = sorted((NEAR_BASE / "textgrid_dir").glob(f"{session}_N_SPK*.TextGrid"))
    if len(wavs) < 2:
        return None, None
    mixed, sr = mix_to_mono(wavs)
    mix_wav = PREP_OUT / f"{session}_mixed.wav"
    write_wav_mono(mix_wav, mixed, sr)
    rttm_path = PREP_OUT / f"{session}.rttm"
    build_rttm(session, tgs, rttm_path)
    return mix_wav, rttm_path


def eval_full():
    sessions = list_sessions()
    print(f"共 {len(sessions)} 场 | strategy={STRATEGY}")
    results = []
    gFA = gMISS = gCONF = gREF = 0
    for i, sess in enumerate(sessions, 1):
        print(f"\n[{i}/{len(sessions)}] {sess}")
        mix_wav, rttm = prep_session(sess)
        if not mix_wav:
            print("  跳过（不足2路）")
            continue
        # 调服务算 DER（strategy 透传，默认 sliding）
        out = eval_one(sess, strategy=STRATEGY)
        if not out:
            print("  跳过（评测失败）")
            continue
        der, detail = out
        results.append({"session": sess, "DER": der, "detail": detail})
        gFA += detail["FA"]
        gMISS += detail["MISS"]
        gCONF += detail["CONF"]
        gREF += detail["REF"]
        print(f"  DER={der * 100:.2f}%")
    gder = (gFA + gMISS + gCONF) / gREF if gREF > 0 else 0
    print(f"\n===== 全量 {len(results)} 场 加权平均 =====")
    print(f"global DER = {gder * 100:.2f}%")
    print(f"FA={gFA} MISS={gMISS} CONF={gCONF} REF={gREF}")
    out_dir = _PROJECT_ROOT / "test_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ali_der_full_{STRATEGY}.json"
    out = {
        "strategy": STRATEGY,
        "global_DER": gder,
        "global": {"FA": gFA, "MISS": gMISS, "CONF": gCONF, "REF": gREF},
        "per_session": results,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {out_path}")


if __name__ == "__main__":
    t0 = time.time()
    eval_full()
    print(f"\n总耗时 {time.time() - t0:.1f}s")
