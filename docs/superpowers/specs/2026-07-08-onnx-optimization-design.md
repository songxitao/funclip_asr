# 2026-07-08 SenseVoiceSmall ONNX GPU 微服务大 Batch 推理与 CPU 标点模型集成设计

## 1. 目标 (Goal)
对 `funclip-pro` 的 ONNX GPU 微服务进行重构与性能优化，主要解决两个核心痛点：
1. **CPU 全核疯跑导致卡死/死机问题**：通过硬性绑定 CPU 物理核心与限制线程数量，从底层守住整机功耗防线。
2. **ASR 推理延迟较长**：实现真正的多 Batch 并发 GPU 推理加速（废除原先单句 for 循环），默认启用 VAD 切片。
3. **文本排版差**：在 CPU 上后处理加回标点符号模型，使长音频转写结果自带格式排版。

---

## 2. 详细设计 (Detailed Design)

### 2.1 CPU 物理限流与资源隔离
在 [asr_onnx_service.py](file:///E:/project/funclip-pro/asr_onnx_service.py) 脚本最头部，导入 `psutil`，将微服务进程强制限制在前 4 个逻辑核心（即逻辑 CPU `0, 1, 2, 3`）。
同时结合 `torch.set_num_threads(4)` 以及各类 CPU OMP 环境变量，软性阻止矩阵库在 4 个核心内过度创建开销线程：
```python
import os
import psutil

# 1. Windows CPU 核心硬性亲和性绑定 (只允许在前4个核心上运行，防止全核拉满死机)
psutil.Process().cpu_affinity([0, 1, 2, 3])

# 2. 软性限制内部多线程库的并行线程数
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import torch
torch.set_num_threads(4)
```

### 2.2 重写 SenseVoiceSmall 包装类支持多 Batch 解码
原基类 `SenseVoiceSmallONNX` (在 `model_bin.py` 中定义) 包含硬编码 `# support batch_size=1 only currently`，只解码并返回每个批次中的第 0 个样本结果。
我们将重写 [asr_onnx_service.py](file:///E:/project/funclip-pro/asr_onnx_service.py) 中的 `SenseVoiceSmall` 类的 `__call__` 方法，实现对整个 batch 维度的遍历解码：

```python
class SenseVoiceSmall(SenseVoiceSmallONNX):
    """包装类，重写了初始化和调用，适配用户要求的接口"""
    def __init__(self, model_dir, batch_size=1, quantize=True, device_id="-1", intra_op_num_threads=4, **kwargs):
        super().__init__(model_dir, batch_size=batch_size, device_id=device_id, quantize=quantize, intra_op_num_threads=intra_op_num_threads, **kwargs)
        # 加载 tokens.json 以还原文本
        tokens_path = os.path.join(model_dir, "tokens.json")
        with open(tokens_path, "r", encoding="utf-8") as f:
            self.tokens = json.load(f)

    def __call__(self, wav_content, language=[0], textnorm=[15], tokenizer=None, **kwargs):
        if tokenizer is None:
            # 内部默认 Tokenizer
            class DefaultTokenizer:
                def __init__(self, tokens):
                    self.tokens = tokens
                def tokens2text(self, ids):
                    res = []
                    for i in ids:
                        t = self.tokens[i]
                        # 过滤掉特殊 tag 像 <|zh|> <|happy|> 等
                        if t.startswith("<|") and t.endswith("|>"):
                            continue
                        if t == "<space>":
                            res.append(" ")
                        elif t == "<unk>":
                            continue
                        else:
                            res.append(t)
                    return "".join(res)
            tokenizer = DefaultTokenizer(self.tokens)

        # 核心改进：重写底层以支持 batch_size > 1
        import numpy as np
        waveform_list = self.load_data(wav_content, self.frontend.opts.frame_opts.samp_freq)
        waveform_nums = len(waveform_list)
        asr_res = []
        
        for beg_idx in range(0, waveform_nums, self.batch_size):
            end_idx = min(waveform_nums, beg_idx + self.batch_size)
            feats, feats_len = self.extract_feat(waveform_list[beg_idx:end_idx])
            ctc_logits, encoder_out_lens = self.infer(
                feats, 
                feats_len, 
                np.array(language, dtype=np.int32), 
                np.array(textnorm, dtype=np.int32)
            )
            # 转换为 torch.Tensor 进行便捷解码
            ctc_logits = torch.from_numpy(ctc_logits).float()
            
            # 支持真正的多 batch 索引解析
            for b in range(end_idx - beg_idx):
                x = ctc_logits[b, : encoder_out_lens[b].item(), :]
                yseq = x.argmax(dim=-1)
                yseq = torch.unique_consecutive(yseq, dim=-1)

                mask = yseq != self.blank_id
                token_int = yseq[mask].tolist()
                
                asr_res.append(tokenizer.tokens2text(token_int))
        return asr_res
```

### 2.3 标点模型 (CT-Punc) 集成与推理流水线
在 ASR 服务加载时，在 CPU (限制4线程) 上载入标点模型，并在推理接口内实现后处理标点追加。

1. **模型载入 (`load_models`)**：
   在 CPU 上载入本地标点模型 [punc_ct-transformer_zh-cn-common-vocab272727-pytorch](file:///E:/project/funclip-pro/model/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch)：
   ```python
   global MODEL, VAD_MODEL, PUNC_MODEL
   punc_path = r"E:\project\funclip-pro\model\models\damo\punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
   PUNC_MODEL = AutoModel(
       model=punc_path,
       trust_remote_code=True,
       device="cpu",
       disable_update=True,
       disable_pbar=True
   )
   PUNC_MODEL.model.to("cpu")
   PUNC_MODEL.kwargs["device"] = "cpu"
   ```

2. **ASR 大 Batch 与标点推理流水线 (`_run_inference`)**：
   重构后的同步推理核心函数如下：
   ```python
   def _run_inference(audio_path: str, vad_split: bool = True) -> str:
       """在独立线程中运行的同步推理逻辑，默认支持开启 VAD"""
       if not vad_split:
           res = MODEL(audio_path)
           if res and len(res) > 0:
               raw_text = res[0].strip()
               clean_text = re.sub(r"<\|.*?\|>", "", raw_text).strip()
               return clean_text
           return ""
       else:
           import librosa
           audio, _ = librosa.load(audio_path, sr=16000)
           
           # 1. 运行 VAD 切分
           vad_out = VAD_MODEL.generate(input=audio_path, batch_size_s=5000, max_single_segment_time=60000)
           raw_segs = vad_out[0]['value'] if vad_out and len(vad_out) > 0 and 'value' in vad_out[0] else [[0, len(audio)/16*1000]]
           
           # 2. 合并小静音切片 (保证单句在 8 秒内)
           def _merge_vad_segments(segments, max_gap_ms=300, max_duration_ms=8000):
               if not segments: return []
               merged = []
               curr_start, curr_end = segments[0]
               for next_start, next_end in segments[1:]:
                   gap = next_start - curr_end
                   duration = (curr_end - curr_start) + (next_end - next_start)
                   if gap < max_gap_ms and duration < max_duration_ms:
                       curr_end = next_end 
                   else:
                       merged.append([curr_start, curr_end]) 
                       curr_start, curr_end = next_start, next_end
               merged.append([curr_start, curr_end])
               return merged
               
           opt_segs = _merge_vad_segments(raw_segs)
           
           # 3. 收集所有音频切片，按 batch 发送给 GPU
           chunks = []
           for start_ms, end_ms in opt_segs:
               s_idx = int(start_ms * 16)
               e_idx = int(end_ms * 16)
               chunk = audio[max(0, s_idx-800):min(len(audio), e_idx+800)]
               if len(chunk) < 1600: continue
               chunks.append(chunk)
               
           if not chunks:
               return ""
               
           # 一次性喂给大 batch MODEL，自动在内部按 batch_size=16 切分并跑 GPU
           # 返回为拼接后的 list 文本结果
           texts = MODEL(chunks)
           
           # 4. 去除特殊标签并用空格/换行拼接
           clean_texts = []
           for t in texts:
               clean = re.sub(r"<\|.*?\|>", "", t).strip()
               if clean:
                   clean_texts.append(clean)
                   
           raw_text = "\n".join(clean_texts)
           
           # 5. 后处理：加回标点符号
           if PUNC_MODEL is not None and raw_text.strip():
               try:
                   punc_out = PUNC_MODEL.generate(input=raw_text)
                   if punc_out and len(punc_out) > 0:
                       raw_text = punc_out[0].get('text', raw_text)
               except Exception as punc_err:
                   logger.error(f"标点还原失败: {punc_err}")
                   
           return raw_text
   ```

3. **FastAPI 的 `/transcribe` 接口变动**：
   - 修改 `vad_split` 的默认值为 `True`。

---

## 3. 测试与验证计划 (Verification Plan)
1. **服务启动与稳定性测试**：
   - 运行服务 `python asr_onnx_service.py` 并观察 CPU 亲和性。
   - 在 Windows 任务管理器中确认该 Python 进程仅使用前 4 个 CPU 逻辑处理器，其余 28 个逻辑处理器占用率为 0%。
2. **多 Batch 推理准确性测试**：
   - 使用包含多句长音频的测试音频（如 `E:\下载\下载\李雪花2.wav`）向 `/transcribe` 发送请求。
   - 验证返回文本是否包含整段对话（如未遗漏 batch 首句之外的内容），并且带上了标点符号。
3. **接口耗时与并发吞吐量比对**：
   - 确认大 batch 重构后转写耗时（应与 CPU 单句 17.19 秒相当或更快，且加上标点模型时间）。
