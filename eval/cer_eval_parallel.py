#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AISHELL-1 CER 并行评测器（GPU 提速版）

- 强制 ASR 走 PyTorch-GPU 引擎（engine=gpu），多服务实例可逗号分隔轮询
- 支持随机采样（--sample N），避免只取前 N 条造成的分布偏差
- 断点续跑：每条结果写 JSONL，重跑自动跳过已完成 utt

用法:
  python eval/cer_eval_parallel.py \
      --wav_dir E:/project/funclip-pro/testset/aishell1_test_extracted/wav \
      --transcript E:/project/funclip-pro/testset/aishell1_test_extracted/transcript.txt \
      --engine gpu --sample 1000 --workers 4 \
      --base_urls http://localhost:8002,http://localhost:8003 \
      --out test_results/cer_sample.jsonl
"""
import argparse
import os
import sys
import glob
import json
import time
import threading
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# 迁移后：脚本在 eval/，项目根在 os.path.dirname(os.path.dirname(__file__))
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

PUNCT = set("，。！？、；：,.!?;:\u3000\t\n\r"
            "“”‘’\"'《》〈〉（）()【】[]「」『』"
            "…—·~～/\\-_=+*&^%@#$")


def normalize(text, keep_punct=False):
    s = text.replace(" ", "").replace("\u3000", "")
    if not keep_punct:
        s = "".join(ch for ch in s if ch not in PUNCT)
    return s


def load_transcript(path, keep_punct=False):
    ref = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            ref[parts[0]] = normalize("".join(parts[1:]), keep_punct)
    return ref


def cer_chars(ref_str, hyp_str):
    r, h = list(ref_str), list(hyp_str)
    n, m = len(r), len(h)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            if r[i - 1] == h[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[m]


def worker(wav_path, base_url, timeout, engine, keep_punct):
    uid = Path(wav_path).stem
    try:
        with open(wav_path, "rb") as fh:
            r = requests.post(
                base_url + "/transcribe",
                files={"file": fh},
                data={"response_format": "text", "engine": engine},
                timeout=timeout,
            )
        hyp = normalize(r.text.strip(), keep_punct)
        return uid, hyp, None
    except Exception as e:
        return uid, None, str(e)


def main():
    ap = argparse.ArgumentParser(description="AISHELL-1 CER 并行评测 (GPU)")
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--base_urls", default="http://localhost:8002",
                    help="逗号分隔的多个服务实例，客户端轮询以水平扩展")
    ap.add_argument("--engine", default="gpu", choices=["gpu", "cpu", "auto"])
    ap.add_argument("--sample", type=int, default=0,
                    help="随机采样条数（固定 seed，覆盖 --limit 的前 N 截取）")
    ap.add_argument("--limit", type=int, default=0, help="取前 N 条（与 --sample 二选一）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(_PROJECT_ROOT, "test_results", "cer_sample.jsonl"))
    ap.add_argument("--keep-punct", action="store_true")
    args = ap.parse_args()

    urls = [u.strip() for u in args.base_urls.split(",") if u.strip()]
    ref = load_transcript(args.transcript, args.keep_punct)
    wavs = sorted(glob.glob(os.path.join(args.wav_dir, "**", "*.wav"), recursive=True))
    jobs = [w for w in wavs if Path(w).stem in ref]
    if args.sample > 0:
        random.seed(args.seed)
        jobs = random.sample(jobs, min(args.sample, len(jobs)))
        print(f"[info] 随机采样 {len(jobs)} 条 (seed={args.seed})")
    elif args.limit:
        jobs = jobs[:args.limit]

    # 断点续跑：跳过已完成（含有效 dist）的 utt
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)
                    if "dist" in o:
                        done.add(o["uid"])
                except Exception:
                    pass
    jobs = [w for w in jobs if Path(w).stem not in done]
    print(f"[info] 待评测 {len(jobs)} 条 (已完成 {len(done)}) | engine={args.engine} "
          f"| workers={args.workers} | urls={urls}")

    lock = threading.Lock()
    total_dist = 0
    total_ref = 0
    done_count = 0
    fails = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex, \
         open(args.out, "a", encoding="utf-8") as outf:
        fut_map = {}
        for idx, w in enumerate(jobs):
            url = urls[idx % len(urls)]
            fut_map[ex.submit(worker, w, url, args.timeout, args.engine, args.keep_punct)] = w
        for fu in as_completed(fut_map):
            uid, hyp, err = fu.result()
            if err is None and hyp is not None:
                gt = ref[uid]
                d = cer_chars(gt, hyp)
                with lock:
                    total_dist += d
                    total_ref += len(gt)
                    done_count += 1
                outf.write(json.dumps({"uid": uid, "dist": d, "gt_len": len(gt)},
                                      ensure_ascii=False) + "\n")
            else:
                fails += 1
                outf.write(json.dumps({"uid": uid, "error": err}, ensure_ascii=False) + "\n")
            processed = done_count + fails
            if processed % 200 == 0 or processed == len(jobs):
                cer = (total_dist / total_ref * 100) if total_ref else 0.0
                print(f"[progress] {processed}/{len(jobs)} | CER_so_far={cer:.2f}% | "
                      f"失败{fails} | 耗时{time.time()-t0:.0f}s", flush=True)

    cer = total_dist / total_ref if total_ref else 0.0
    summary = {
        "cer": cer, "total_ref": total_ref, "total_dist": total_dist,
        "done": done_count, "fails": fails, "engine": args.engine,
        "workers": args.workers, "urls": urls, "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.splitext(args.out)[0] + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== CER 评测结果 =====")
    print(f"评测条数 : {done_count} (失败 {fails})")
    print(f"CER      : {cer * 100:.2f}%")
    print(f"耗时     : {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
