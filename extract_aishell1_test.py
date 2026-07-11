"""从 xiaofff/omnievalkit-data-test 的 aishell1_test 子集抽取 wav + transcript，
供 cer_eval.py 直接消费。

用法（在 asr_ui_env 中）：
  # 1) 设镜像 + 装库（datasets 已确认，soundfile 为写 wav 配套）
  $env:HF_ENDPOINT = "https://hf-mirror.com"
  pip install datasets soundfile

  # 2) 先小批量验证（200 条）
  python extract_aishell1_test.py --out_dir E:/project/funclip-pro/testset/aishell1_test_extracted --limit 200

  # 3) 全量抽取
  python extract_aishell1_test.py --out_dir E:/project/funclip-pro/testset/aishell1_test_extracted
"""
import os
import argparse
import soundfile as sf
from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="xiaofff/omnievalkit-data-test",
                    help="HuggingFace 数据集 repo（走 HF_ENDPOINT 镜像）")
    ap.add_argument("--subset", default="aishell1_test",
                    help="子集名，AISHELL-1 的 7176 条 test")
    ap.add_argument("--out_dir", required=True,
                    help="输出目录，下含 wav/ 与 transcript.txt")
    ap.add_argument("--limit", type=int, default=0,
                    help="仅抽取前 N 条做验证（0=全量）")
    args = ap.parse_args()

    print(f"[*] 加载 {args.repo} / {args.subset}（首次会从 HF 镜像下载 ~1.1GB）...")
    ds = load_dataset(args.repo, args.subset, split="test")

    wav_dir = os.path.join(args.out_dir, "wav")
    os.makedirs(wav_dir, exist_ok=True)
    trans_path = os.path.join(args.out_dir, "transcript.txt")

    n = 0
    with open(trans_path, "w", encoding="utf-8") as ft:
        for i, sample in enumerate(ds):
            if args.limit and i >= args.limit:
                break
            audio = sample["audio"]
            text = sample.get("text", "").strip()
            # utt_id：优先用显式 id 字段，否则取音频文件名，最后 fallback 序号
            utt = (sample.get("utt_id") or sample.get("id") or
                   os.path.splitext(os.path.basename(audio.get("path", "")))[0] or
                   f"utt_{i:06d}")
            utt = utt.replace(" ", "_")  # 防止 id 含空格破坏 transcript 解析
            arr = audio["array"]
            sr = int(audio["sampling_rate"])
            sf.write(os.path.join(wav_dir, utt + ".wav"), arr, sr)
            ft.write(f"{utt} {text}\n")
            n += 1
            if n % 500 == 0:
                print(f"    ... 已抽取 {n} 条")

    print(f"[done] 共 {n} 条 -> wav: {wav_dir} | transcript: {trans_path}")


if __name__ == "__main__":
    main()
