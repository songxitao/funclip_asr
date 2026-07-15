#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AISHELL-1 字错率(CER)评测脚本

流程: 读取标注 -> 遍历解压后的 wav -> 逐条调本地 8002 /transcribe -> 字符级 CER

前置: 先把 AISHELL-1 的 tar.gz 解压 (Git Bash):
  mkdir -p /e/project/funclip-pro/testset/AISHELL-1/extracted
  for f in /e/project/funclip-pro/testset/AISHELL-1/data_aishell/wav/*.tar.gz; do
    tar -xzf "$f" -C /e/project/funclip-pro/testset/AISHELL-1/extracted
  done

用法:
  # 先 dry-run 验证 wav 与标注匹配 (不调服务, 零依赖)
  python eval/cer_eval.py --wav_dir /e/project/funclip-pro/testset/AISHELL-1/extracted \
                     --transcript /e/project/funclip-pro/testset/AISHELL-1/data_aishell/transcript/aishell_transcript_v0.8.txt \
                     --dry-run

  # 正式评测 (限前 200 条先验证流程)
  python eval/cer_eval.py --wav_dir .../extracted --transcript .../aishell_transcript_v0.8.txt \
                     --limit 200 --base_url http://localhost:8002

说明:
  - AISHELL-1 标注每行 "utt_id 字 字 字", 按字符算 CER, 默认去空格+去标点(纯字错, 与业界一致)
  - 本地下载的 HF 版 AISHELL-1 仅含 train wav (无 test/dev), 故 CER 为 train 子集数字,
    非官方 test benchmark。要官方数字需补下 OpenSLR-33 的 test.tar.gz
"""
import argparse
import os
import sys
import glob
import time
from pathlib import Path


# 中英文标点集合（去标点后可让 CER 只反映纯字错，不被标点差异虚高）
PUNCT = set("，。！？、；：,.!?;:\u3000\t\n\r"
            "“”‘’\"'《》〈〉（）()【】[]「」『』"
            "…—·~～/\\-_=+*&^%@#$")


def normalize(text, keep_punct=False):
    """去空格；默认去中英文标点，让 CER 只算纯字符错误（与业界标准一致）。"""
    s = text.replace(" ", "").replace("\u3000", "")
    if not keep_punct:
        s = "".join(ch for ch in s if ch not in PUNCT)
    return s


def load_transcript(path, keep_punct=False):
    """每行: utt_id 字 字 字 ... -> {utt_id: 归一化后的字符序列}"""
    ref = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            uid = parts[0]
            text = normalize("".join(parts[1:]), keep_punct)  # 去分词空格+可选去标点
            ref[uid] = text
    return ref


def cer_chars(ref_str, hyp_str):
    """字符级编辑距离 (Levenshtein), 用于算字错率"""
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


def main():
    ap = argparse.ArgumentParser(description="AISHELL-1 CER 评测")
    ap.add_argument("--wav_dir", required=True, help="解压后的 wav 根目录 (递归找 *.wav)")
    ap.add_argument("--transcript", required=True, help="aishell_transcript_v0.8.txt")
    ap.add_argument("--base_url", default="http://localhost:8002", help="8002 服务地址")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条 (0=全部)")
    ap.add_argument("--timeout", type=int, default=60, help="单条请求超时秒")
    ap.add_argument("--keep-punct", action="store_true",
                    help="保留标点计算 CER（默认去标点，只算字错，与业界标准一致）")
    ap.add_argument("--dry-run", action="store_true", help="只统计 wav/标注匹配, 不调服务")
    args = ap.parse_args()

    ref = load_transcript(args.transcript, args.keep_punct)
    print(f"[info] 标注条数: {len(ref)}")

    wavs = sorted(glob.glob(os.path.join(args.wav_dir, "**", "*.wav"), recursive=True))
    print(f"[info] 找到 wav: {len(wavs)}")

    matched = sum(1 for w in wavs if Path(w).stem in ref)
    print(f"[info] wav 与标注匹配: {matched}/{len(wavs)}")

    if args.dry_run:
        print("[dry-run] 仅验证匹配, 未调用服务。去掉 --dry-run 正式评测。")
        return

    if matched == 0:
        sys.exit("[错误] 无匹配 wav, 检查 --wav_dir 与 --transcript 是否对应")

    # 正式评测才需要 requests
    try:
        import requests
    except ImportError:
        sys.exit("[错误] 缺少 requests 库。请先安装: pip install requests (需你确认后我可协助)")

    total_dist = 0
    total_ref = 0
    done = 0
    fails = 0
    t0 = time.time()
    for w in wavs:
        uid = Path(w).stem
        if uid not in ref:
            continue
        gt = ref[uid]
        try:
            with open(w, "rb") as fh:
                r = requests.post(
                    args.base_url + "/transcribe",
                    files={"file": fh},
                    data={"response_format": "text"},
                    timeout=args.timeout,
                )
            hyp = normalize(r.text.strip(), args.keep_punct)
        except Exception as e:
            fails += 1
            print(f"[warn] {uid} 调用失败: {e}")
            continue
        total_dist += cer_chars(gt, hyp)
        total_ref += len(gt)
        done += 1
        if args.limit and done >= args.limit:
            break

    cer = total_dist / total_ref if total_ref else 0
    print("\n===== CER 评测结果 =====")
    print(f"评测条数 : {done} (调用失败 {fails})")
    print(f"总字符数 : {total_ref}")
    print(f"总错字   : {total_dist}")
    print(f"CER      : {cer * 100:.2f}%")
    print(f"耗时     : {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
