"""对比评测：Sherpa-ONNX(CPU) vs PyTorch(GPU) 在 李雪花2.wav 上的表现。

控制变量：两条路径共用同一套 VAD 切分 + 同一套「剥原生标点 -> 拼全文 -> PUNC 一次」后处理，
仅 ASR 引擎不同（Sherpa INT8 ONNX CPU vs PyTorch FP32 GPU）。

输出：
  - 打印延迟、字符级一致度(CER)、标点统计
  - 验证「逐段标点 vs 全文 PUNC」差异（Sherpa 路径）
  - 保存两份转写 + JSON 报告到 tests/bench_report.json
"""
import os
import re
import sys
import io
import time
import json

ROOT = r"E:\project\funclip-pro"
sys.path.insert(0, ROOT)

for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[k] = "6"

import torch
torch.set_num_threads(6)

from funasr import AutoModel
import sherpa_onnx
import librosa

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

AUDIO = r"E:\下载\下载\李雪花2.wav"
SHERPA_MODEL = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmallOnnx"
MODEL_INT8 = os.path.join(SHERPA_MODEL, "model.int8.onnx")
PT_ASR = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmall"
VAD = r"E:\project\funclip-pro\model\models\damo\speech_fsmn_vad_zh-cn-16k-common-pytorch"
PUNC = r"E:\project\funclip-pro\model\models\damo\punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

PUNCS = set("，。！？、；：\"”‘’（）《》—…·,.!?;:")
TAG_RE = re.compile(r"<\|.*?\|>")
PUNC_RE = re.compile(r"[，。！？；：、…—·「」『』“”‘’（）《》〈〉【】\[\]\(\)\{\}\"'\.,!?;:\s]")


def strip_punc(s: str) -> str:
    return PUNC_RE.sub("", s).strip()


def strip_tags(s: str) -> str:
    return TAG_RE.sub("", s).strip()


def merge(segs, max_gap=300, max_dur=8000):
    if not segs:
        return []
    m = []
    cs, ce = segs[0]
    for ns, ne in segs[1:]:
        gap = ns - ce
        dur = (ce - cs) + (ne - ns)
        if gap < max_gap and dur < max_dur:
            ce = ne
        else:
            m.append([cs, ce])
            cs, ce = ns, ne
    m.append([cs, ce])
    return m


def get_vad_chunks():
    vad = AutoModel(model=VAD, trust_remote_code=True, device="cpu",
                    disable_update=True, disable_pbar=True)
    audio, sr = librosa.load(AUDIO, sr=16000)
    vad_out = vad.generate(input=AUDIO, batch_size_s=5000, max_single_segment_time=60000)
    raw = vad_out[0]["value"] if vad_out and vad_out[0].get("value") else [[0, len(audio) / 16 * 1000]]
    opt = merge(raw)
    chunks = []
    for s, e in opt:
        si, ei = int(s * 16), int(e * 16)
        c = audio[max(0, si - 800):min(len(audio), ei + 800)]
        if len(c) < 1600:
            continue
        chunks.append(c)
    return audio, chunks


def apply_punc(model, text):
    out = model.generate(input=text)
    return out[0].get("text", text) if out else text


def lev(a, b):
    if a == b:
        return 0
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            tmp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = tmp
    return dp[lb]


def cer(a, b):
    d = lev(a, b)
    base = max(len(a), 1)
    return d / base


def punc_positions(s):
    return [(i, c) for i, c in enumerate(s) if c in PUNCS]


def main():
    assert os.path.exists(AUDIO), f"音频不存在: {AUDIO}"
    audio, chunks = get_vad_chunks()
    print(f"[VAD] 有效 chunks: {len(chunks)}  | 音频时长 ≈ {len(audio)/16000:.1f}s")

    # 共享 PUNC 模型（CPU）
    punc_model = AutoModel(model=PUNC, trust_remote_code=True, device="cpu",
                           disable_update=True, disable_pbar=True)

    report = {"audio": AUDIO, "n_chunks": len(chunks),
              "duration_s": round(len(audio) / 16000, 1), "engines": {}}

    # ---------- 路径 A: Sherpa-ONNX (CPU) ----------
    print("\n=== [A] Sherpa-ONNX (CPU, num_threads=6) ===")
    from sherpa_engine import SherpaSenseVoice
    eng = SherpaSenseVoice(model_dir=SHERPA_MODEL, num_threads=6, use_itn=True)

    t0 = time.time()
    raw_texts = eng(chunks)                     # 逐段原生输出（含标点+ITN，无标签）
    t_asr = time.time() - t0

    # 逐段原生标点（as-is），用于验证「逐段标点」问题
    sherpa_asis = "\n".join(strip_tags(t) for t in raw_texts if strip_tags(t).strip())

    # 纠正路径：剥标点 -> 拼全文 -> PUNC 一次
    clean = [strip_punc(strip_tags(t)) for t in raw_texts]
    clean = [c for c in clean if c]
    full_text = "\n".join(clean)
    t1 = time.time()
    sherpa_punc = apply_punc(punc_model, full_text)
    t_punc = time.time() - t1
    t_total_a = t_asr + t_punc

    print(f"  ASR 耗时: {t_asr:.2f}s | PUNC 耗时: {t_punc:.2f}s | 总计: {t_total_a:.2f}s")
    print(f"  转写(前 200 字): {sherpa_punc[:200]}")

    # 逐段标点 vs 全文 PUNC 的差异（用户担心的点）
    pos_asis = punc_positions(sherpa_asis)
    pos_full = punc_positions(sherpa_punc)
    # 以全文 PUNC 文本长度为基准比对标点位置
    diff = len(pos_asis) - len(pos_full)
    print(f"  逐段标点数: {len(pos_asis)} | 全文PUNC标点数: {len(pos_full)} | 差: {diff}")

    report["engines"]["sherpa_cpu"] = {
        "device": "cpu", "asr_s": round(t_asr, 3), "punc_s": round(t_punc, 3),
        "total_s": round(t_total_a, 3),
        "n_punc_asis": len(pos_asis), "n_punc_full": len(pos_full),
        "text_full_punc": sherpa_punc,
        "text_asis": sherpa_asis,
    }

    # ---------- 路径 B: PyTorch SenseVoice (GPU) ----------
    print("\n=== [B] PyTorch SenseVoice (GPU) ===")
    assert torch.cuda.is_available(), "CUDA 不可用"
    torch.cuda.reset_peak_memory_stats()
    pt = AutoModel(model=PT_ASR, trust_remote_code=True, device="cuda:0",
                   disable_update=True, disable_pbar=True)

    t0 = time.time()
    pt_out = pt.generate(input=chunks, batch_size_s=0, language="auto", use_itn=True)
    t_asr = time.time() - t0
    pt_texts = [o.get("text", "") for o in pt_out]

    clean = [strip_punc(strip_tags(t)) for t in pt_texts]
    clean = [c for c in clean if c]
    full_text = "\n".join(clean)
    t1 = time.time()
    pt_punc = apply_punc(punc_model, full_text)
    t_punc = time.time() - t1
    t_total_b = t_asr + t_punc
    gpu_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"  ASR 耗时: {t_asr:.2f}s | PUNC 耗时: {t_punc:.2f}s | 总计: {t_total_b:.2f}s")
    print(f"  GPU 峰值显存: {gpu_mem:.0f} MiB")
    print(f"  转写(前 200 字): {pt_punc[:200]}")

    report["engines"]["pytorch_gpu"] = {
        "device": "cuda:0", "asr_s": round(t_asr, 3), "punc_s": round(t_punc, 3),
        "total_s": round(t_total_b, 3), "gpu_peak_mib": round(gpu_mem, 1),
        "text_full_punc": pt_punc,
    }

    # ---------- 评估 ----------
    print("\n=== 评估 ===")
    sherpa_dp = strip_punc(sherpa_punc)
    pt_dp = strip_punc(pt_punc)
    c = cer(sherpa_dp, pt_dp)
    print(f"  去标点后文本长度: Sherpa={len(sherpa_dp)} | PyTorch={len(pt_dp)}")
    print(f"  字符级一致度 CER(Sherpa vs PyTorch, 去标点): {c*100:.2f}%  (越低越一致)")
    print(f"  最终文本标点数: Sherpa={len(punc_positions(sherpa_punc))} | PyTorch={len(punc_positions(pt_punc))}")
    print(f"  速度: Sherpa-CPU 总计 {t_total_a:.2f}s vs PyTorch-GPU 总计 {t_total_b:.2f}s")
    if t_total_b > 0:
        print(f"  PyTorch-GPU 相对 Sherpa-CPU 加速比: {t_total_a/t_total_b:.2f}x")

    report["eval"] = {
        "cer_depunctuated": round(c, 4),
        "len_sherpa_dp": len(sherpa_dp),
        "len_pytorch_dp": len(pt_dp),
        "n_punc_sherpa": len(punc_positions(sherpa_punc)),
        "n_punc_pytorch": len(punc_positions(pt_punc)),
        "speedup_pytorch_vs_sherpa": round(t_total_a / t_total_b, 3) if t_total_b > 0 else None,
    }

    out = os.path.join(ROOT, "tests", "bench_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {out}")


if __name__ == "__main__":
    main()
