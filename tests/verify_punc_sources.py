"""验证 Sherpa-ONNX 与原生 PyTorch SenseVoice 的原始输出是否自带标点。
使用 VAD 对 李雪花2.wav 切片，分别解码后检查：
  1) 输出是否含中文标点（，。！？等）
  2) 输出是否含 <|...|> 富文本标签
结论用于判断迁移到 Sherpa 后是否仍需保留 PUNC_MODEL。
"""
import os
import re
import sys
import io

for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[k] = "6"

import torch
torch.set_num_threads(6)

from funasr import AutoModel
import sherpa_onnx
import librosa

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

AUDIO = r"E:\下载\下载\李雪花2.wav"
SHERPA_MODEL = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmallOnnx"
MODEL_INT8 = os.path.join(SHERPA_MODEL, "model.int8.onnx")
SHERPA_TOKENS = os.path.join(SHERPA_MODEL, "tokens.txt")
PT_ASR = r"E:\project\funclip-pro\model\models\iic\SenseVoiceSmall"
VAD = r"E:\project\funclip-pro\model\models\damo\speech_fsmn_vad_zh-cn-16k-common-pytorch"

PUNCS = set("，。！？、；：“”‘’（）《》—…·")
TAG_RE = re.compile(r"<\|.*?\|>")


def has_punc(s):
    return any(c in PUNCS for c in s)


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


def main():
    assert os.path.exists(AUDIO), f"音频不存在: {AUDIO}"

    # ---- VAD 切分 ----
    vad = AutoModel(model=VAD, trust_remote_code=True, device="cpu",
                    disable_update=True, disable_pbar=True)
    audio, sr = librosa.load(AUDIO, sr=16000)
    vad_out = vad.generate(input=AUDIO, batch_size_s=5000, max_single_segment_time=60000)
    raw_segs = vad_out[0]['value'] if vad_out and vad_out[0].get('value') else [[0, len(audio) / 16 * 1000]]
    opt = merge(raw_segs)
    chunks = []
    for s, e in opt:
        si, ei = int(s * 16), int(e * 16)
        c = audio[max(0, si - 800):min(len(audio), ei + 800)]
        if len(c) < 1600:
            continue
        chunks.append(c)
    print(f"[VAD] 原始片段 {len(raw_segs)} -> 有效 chunks {len(chunks)}")

    # ---- Sherpa-ONNX (use_itn=True) ----
    rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=MODEL_INT8, tokens=SHERPA_TOKENS, num_threads=6, use_itn=True)
    ss = []
    for c in chunks:
        st = rec.create_stream()
        st.accept_waveform(16000, c)
        ss.append(st)
    rec.decode_streams(ss)
    sherpa_text = "".join(s.result.text for s in ss)
    print("\n=== SHERPA 原始输出（前 500 字）===")
    print(sherpa_text[:500])
    print(f"[SHERPA] 含标点: {has_punc(sherpa_text)} | 含 <|...|> 标签: {bool(TAG_RE.search(sherpa_text))}")

    # ---- 原生 PyTorch SenseVoice (use_itn=True) ----
    pt = AutoModel(model=PT_ASR, trust_remote_code=True, device="cpu", disable_update=True)
    pt_out = pt.generate(input=chunks, batch_size_s=0, language="auto", use_itn=True)
    pt_text = "".join(o.get('text', '') for o in pt_out)
    print("\n=== PYTORCH 原始输出（前 500 字）===")
    print(pt_text[:500])
    print(f"[PYTORCH] 含标点: {has_punc(pt_text)} | 含 <|...|> 标签: {bool(TAG_RE.search(pt_text))}")

    print("\n=== 结论 ===")
    print(f"Sherpa 原生带标点: {has_punc(sherpa_text)}")
    print(f"PyTorch 原生带标点: {has_punc(pt_text)}")


if __name__ == "__main__":
    main()
