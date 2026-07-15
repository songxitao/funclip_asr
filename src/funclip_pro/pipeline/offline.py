"""pipeline.offline — OfflinePipeline 统一转写流水线（整合收口层）。

等价整合原根文件 asr_onnx_service.py 的 _run_inference 编排逻辑：
  VAD 三态策略 + 引擎自动路由 + 廉价 trim + FSMN VAD 切分
  + seg_clustering / sliding / two_stage 说话人分离分支 + 子句级对齐
  + SRT 组装，对外暴露统一步骤管理器 OfflinePipeline。

设计约束（来自 07-pipeline-offline.md 契约 + P0/P1 spec）：
  - 等价优先：算法逻辑与原 _run_inference 字节级一致，禁止 45454 交替
    / 字级时间戳对齐优化。
  - 时间戳单位统一毫秒(ms)。
  - 复用已落地的 funclip_pro.core / funclip_pro.utils，不重新实现引擎。
  - DLL 补丁保活：apply_dll_patch() 在导入时点亮一次。
  - 路径解耦：全部经 resolve_model_path 解析，零硬编码盘符/绝对路径。
  - 不加载模型权重于 import 期；权重在实例化/惰性 getter 中按原路径解析加载。
"""

from __future__ import annotations

import logging
import threading

# DLL 补丁保活：必须在首次加载 torch / onnxruntime / campplus 等重型库前点亮
# 一次（等价 asr_onnx_service.py 顶部 + core.speaker 的处理）。
from funclip_pro.config.loader import resolve_model_path, apply_dll_patch
apply_dll_patch()

from funclip_pro.core import (
    SegmentationEngine,
    CampPlusSpeaker,
    SenseVoiceSmall,
    PyTorchSenseVoice,
    SherpaSenseVoice,
    load_models,
    _assign_clauses_to_speakers,
    _assign_clauses_to_speakers_seamless,
)
# VAD 模型以模块级全局句柄持有（load_models 填充）；原 _run_inference 直接引用
# 模块全局 VAD_MODEL，这里通过 asr 子模块句柄等价引用，保证调用期读取最新值。
from funclip_pro.core import asr as asr_mod
from funclip_pro.utils import (
    _ms_to_srt,
    _merge_same_speaker_segments,
    _segments_to_srt,
)

logger = logging.getLogger("OfflinePipeline")

# 说话人 / 分割模型目录（路径解耦，由 resolve_model_path 解析，等价原 _get_spk_model/_get_seg_model）
SPK_MODEL_DIR = str(resolve_model_path("models/damo/speech_campplus_sv_zh-cn_16k-common"))
SEG_MODEL_DIR = str(resolve_model_path("models/damo/segmentation-3.0"))

# VAD 生成参数（等价原 _run_inference 的两处 VAD_MODEL.generate 调用）
_VAD_BATCH_SIZE_S = 5000
_VAD_MAX_SINGLE_SEGMENT_TIME = 60000


class OfflinePipeline:
    """统一转写流水线收口层。

    等价原 asr_onnx_service._run_inference 的编排；复用 funclip_pro.core
    各引擎与 funclip_pro.utils 的 SRT 工具，不重新实现推理逻辑。

    主方法 run() 返回四元组 (raw_text, engine_key, segments, diarized_text)，
    与原 _run_inference 字节级一致 —— 任意分支都不得以 list 提前 return。
    """

    def __init__(self, config=None, device: str = None, auto_load: bool = True):
        """初始化流水线。

        Args:
            config: 可选配置对象（config.loader.load_config() 结果），当前仅占位，
                    路径一律由 resolve_model_path 解析。
            device: 偏好设备（"cuda"/"cpu"），默认 None 表示沿用原代码的
                    CUDA 优先 + CPU 回退策略。
            auto_load: 是否在实例化时加载 Sherpa-ASR / VAD / PUNC 模型权重。
                       等价原 @app.on_event("startup") 的 load_models 调用。
        """
        self.config = config
        self.device = device

        # 惰性句柄（等价原模块级 SPK_MODEL / SEG_MODEL 全局）
        self._spk_model = None
        self._seg_model = None
        self._spk_lock = threading.Lock()
        self._seg_lock = threading.Lock()

        if auto_load:
            self.load_models()

    # ------------------------------------------------------------------
    # 模型加载 / 惰性 getter（等价原 asr_onnx_service 的 load_models /
    # _get_spk_model / _get_seg_model）
    # ------------------------------------------------------------------
    def load_models(self):
        """加载 Sherpa-ASR、VAD(优先 GPU)、CPU 标点模型。

        等价原 load_models()：委托 core.asr.load_models 填充 asr_mod.MODEL /
        asr_mod.VAD_MODEL / asr_mod.PUNC_MODEL 三个全局句柄；torch/spk/seg
        引擎保持惰性，首次使用时才构造。
        """
        # 保活 DLL 搜索目录（等价原顶部调用）
        apply_dll_patch()
        load_models()

    def _get_spk_model(self):
        """惰性构建 Cam++ 说话人模型；首次 diarize 才加载，CUDA 优先回退 CPU。"""
        with self._spk_lock:
            if self._spk_model is None:
                try:
                    self._spk_model = CampPlusSpeaker(model_dir=SPK_MODEL_DIR, device="cuda")
                    logger.info("[Speaker] Cam++ 已加载到 CUDA")
                except Exception as e:
                    logger.warning(f"[Speaker] Cam++ CUDA 加载失败，回退 CPU: {e}")
                    self._spk_model = CampPlusSpeaker(model_dir=SPK_MODEL_DIR, device="cpu")
            return self._spk_model

    def _get_seg_model(self):
        """惰性构建 Segmentation 模型；首次 seg_clustering 才加载，CUDA 优先回退 CPU。"""
        with self._seg_lock:
            if self._seg_model is None:
                try:
                    self._seg_model = SegmentationEngine(model_dir=SEG_MODEL_DIR, device="cuda")
                    logger.info("[Segmentation] pyannote/segmentation-3.0 已加载到 CUDA")
                except Exception as e:
                    logger.warning(f"[Segmentation] pyannote/segmentation-3.0 CUDA 加载失败，回退 CPU: {e}")
                    self._seg_model = SegmentationEngine(model_dir=SEG_MODEL_DIR, device="cpu")
            return self._seg_model

    # ------------------------------------------------------------------
    # 主转写方法（等价原 _run_inference）
    # ------------------------------------------------------------------
    def run(self, audio_path: str, vad_strategy: str = "auto", diarize: bool = False,
            engine=None, language: list = None, textnorm: list = None,
            diarize_strategy: str = "two_stage", num_speakers: int = None):
        """同步推理逻辑（等价原 _run_inference，在独立线程运行）。

        - 根据 vad_strategy 决定走廉价 trim 直解还是完整 FSMN VAD 切分
        - 根据 engine 覆盖 / 自动路由选择 Sherpa-CPU 或 PyTorch-GPU
        - diarize=True 时先做 VAD 切分，对片段离线聚类，产出段级 [说话人] 标注
        - diarize_strategy: "single" | "two_stage"(默认) | "spectral" | "sliding"
          | "seg_clustering" — 聚类策略
        返回 (text, engine_key, segments, diarized_text)

        language / textnorm 为接口兼容参数（保持与原签名一致），当前_decode
        路径沿用各引擎默认的 language=[0] / textnorm=[15]，与原实现等价。
        """
        import librosa

        y, sr = librosa.load(audio_path, sr=16000)
        duration_ms = len(y) / sr * 1000

        engine_key = asr_mod._select_engine(engine, duration_ms)

        # 🔥 Qwen3 (Docker) 引擎专用分支：直接调用 Docker API，跳过本地 VAD/解码
        if engine_key == "qwen":
            from funclip_pro.core.asr import QwenEngine, parse_qwen_timestamps

            qwen_engine = QwenEngine()
            result = qwen_engine.transcribe(audio_path, language=language[0] if isinstance(language, list) else (language or "auto"))
            text = result["text"]
            segments = parse_qwen_timestamps(result.get("raw", {}))
            return text, "qwen", segments or [], ""

        if diarize and diarize_strategy == "seg_clustering":
            try:
                # 1. 运行全局说话人聚类与分割（保证说话人3等全局一致性）
                seg_engine = self._get_seg_model()
                spk_model = self._get_spk_model()
                seamless_segs = spk_model.cluster_with_seamless_segmentation(
                    y, segment_engine=seg_engine, sr=16000, n_speakers=num_speakers
                )

                # 无缝时间轴：确定段毫秒，未知段保留 seg_type
                refined_segs = []
                for st_sec, en_sec, val in seamless_segs:
                    if isinstance(val, int):
                        refined_segs.append((st_sec * 1000, en_sec * 1000, val))
                    else:
                        refined_segs.append((st_sec * 1000, en_sec * 1000, val))
                refined_segs = sorted(refined_segs, key=lambda x: x[0])

                # 2. 运行 VAD 分割过滤静音段（粗粒度段）
                vad_model = asr_mod.VAD_MODEL
                vad_out = vad_model.generate(
                    input=audio_path,
                    batch_size_s=_VAD_BATCH_SIZE_S,
                    max_single_segment_time=_VAD_MAX_SINGLE_SEGMENT_TIME,
                )
                raw_segs = vad_out[0]['value'] if vad_out and len(vad_out) > 0 and 'value' in vad_out[0] else [[0, duration_ms]]
                opt_segs = asr_mod._merge_vad_segments(raw_segs)

                # 3. 对 VAD 粗粒度段进行 ASR 波形切分，加左右各 800ms 缓冲，保证识别不丢字
                asr_waveforms = []
                final_opt_segs = []
                for start_ms, end_ms in opt_segs:
                    s_idx = int(start_ms * 16)
                    e_idx = int(end_ms * 16)
                    chunk = y[max(0, s_idx - 800): min(len(y), e_idx + 800)]
                    if len(chunk) < 1600:
                        continue
                    asr_waveforms.append(chunk)
                    final_opt_segs.append((start_ms, end_ms))

                if not asr_waveforms:
                    return ("", engine_key, [], "")

                # 4. 批量解码 ASR chunks 并施加标点
                texts = asr_mod._decode(engine_key, asr_waveforms)
                punc_texts = []
                for t in texts:
                    cleaned = asr_mod._clean(t)
                    punc_text = asr_mod._post_punc(cleaned)
                    punc_texts.append(punc_text)

                # 5. 子句级对齐分配与排重（每个 VAD 段内独立合并同说话人，不跨段）
                segments = []
                for (asr_start, asr_end), punc_text in zip(final_opt_segs, punc_texts):
                    if not punc_text.strip():
                        continue
                    sub_segs = _assign_clauses_to_speakers_seamless(asr_start, asr_end, punc_text, refined_segs)
                    # 只在 VAD 段内部合并相邻同说话人，防止解说与角色跨段混淆
                    sub_segs = _merge_same_speaker_segments(sub_segs)
                    segments.extend(sub_segs)

                # 再次按时间排序
                segments = sorted(segments, key=lambda x: x["start"])

                # 6. 生成 diarized_text 和 raw_text
                diarized_text = "\n".join(
                    f"[说话人{seg['speaker']}] {seg['text']}" for seg in segments if seg["text"].strip()
                )
                raw_text = "\n".join([seg["text"] for seg in segments if seg["text"].strip()])

                return (raw_text, engine_key, segments, diarized_text)
            except Exception as e:
                logger.error(f"VAD-assisted Diarization-driven ASR failed: {e}", exc_info=True)
                # 失败则回退原流程

        use_vad = asr_mod._use_vad(vad_strategy, duration_ms)

        # 说话人分离要求先做语音切分
        if diarize:
            use_vad = True

        if not use_vad:
            # 廉价 trim 直解（防幻觉最低保障，不跑完整 VAD）
            y_trim, _ = asr_mod._cheap_trim(audio_path)
            waveforms = [y_trim]
        else:
            # 完整 FSMN VAD 切分
            vad_model = asr_mod.VAD_MODEL
            vad_out = vad_model.generate(
                input=audio_path,
                batch_size_s=_VAD_BATCH_SIZE_S,
                max_single_segment_time=_VAD_MAX_SINGLE_SEGMENT_TIME,
            )
            raw_segs = vad_out[0]['value'] if vad_out and len(vad_out) > 0 and 'value' in vad_out[0] else [[0, duration_ms]]

            opt_segs = asr_mod._merge_vad_segments(raw_segs)

            chunks = []
            seg_meta = []   # 与 chunks 一一对应的 (start_ms, end_ms)，用于段级标注
            for start_ms, end_ms in opt_segs:
                s_idx = int(start_ms * 16)
                e_idx = int(end_ms * 16)
                chunk = y[max(0, s_idx - 800): min(len(y), e_idx + 800)]
                if len(chunk) < 1600:
                    continue
                chunks.append(chunk)
                seg_meta.append((start_ms, end_ms))
            waveforms = chunks

        if not waveforms:
            return ("", engine_key, [], "")

        texts = asr_mod._decode(engine_key, waveforms)
        clean_texts = [asr_mod._clean(t) for t in texts]   # 与 chunks 一一对齐
        # 全文拼接（向后兼容）：过滤空串后跑一次 PUNC（整句上下文，断句最准）
        joined = "\n".join([t for t in clean_texts if t])
        raw_text = asr_mod._post_punc(joined)

        # 说话人分离：对切分片段做离线聚类，产出段级 [说话人] 标注
        segments = []
        diarized_text = ""
        if diarize and chunks:
            try:
                if diarize_strategy == "sliding":
                    # 滑窗说话人分离：整段音频内部固定窗滑切，逐窗提 Cam++ 向量，
                    # 复用 spectral 聚类并合并相邻同人窗。VAD 段仅服务 ASR，不参与分人。
                    merged = self._get_spk_model().cluster_sliding(
                        y, sr=16000, strategy="spectral",
                        n_speakers=num_speakers, win_sec=1.5, step_sec=0.5,
                    )
                    for st, en, spk in merged:
                        segments.append({
                            "start": int(st * 1000),
                            "end": int(en * 1000),
                            "speaker": str(spk),
                            "text": "",
                        })
                elif diarize_strategy == "seg_clustering":
                    seg_engine = self._get_seg_model()
                    merged = self._get_spk_model().cluster_with_segmentation(
                        y, segment_engine=seg_engine, sr=16000, n_speakers=num_speakers
                    )
                    for st_sec, en_sec, spk in merged:
                        seg_start_ms = st_sec * 1000
                        seg_end_ms = en_sec * 1000

                        # 按时间重叠判断回填重合部分的 ASR text
                        matched_texts = []
                        for i, (asr_start, asr_end) in enumerate(seg_meta):
                            overlap = min(seg_end_ms, asr_end) - max(seg_start_ms, asr_start)
                            if overlap > 0:
                                asr_text = clean_texts[i] if i < len(clean_texts) else ""
                                if asr_text.strip():
                                    matched_texts.append(asr_text.strip())

                        seg_text = "".join(matched_texts)
                        segments.append({
                            "start": int(seg_start_ms),
                            "end": int(seg_end_ms),
                            "speaker": str(spk),
                            "text": seg_text,
                        })
                else:
                    spk_cache = self._get_spk_model().cluster(
                        chunks, strategy=diarize_strategy, seg_times=seg_meta, n_speakers=num_speakers
                    )
                    for i, (start_ms, end_ms) in enumerate(seg_meta):
                        spk = str(spk_cache.get(i, "?"))
                        seg_text = clean_texts[i] if i < len(clean_texts) else ""
                        segments.append({
                            "start": start_ms,
                            "end": end_ms,
                            "speaker": spk,
                            "text": seg_text,
                        })
                # 合并相邻同说话人段（适用于所有 diarize 策略）
                segments = _merge_same_speaker_segments(segments)
                diarized_text = "\n".join(
                    f"[说话人{seg['speaker']}] {seg['text']}" for seg in segments if seg["text"]
                )
            except Exception as spk_err:
                logger.error(f"说话人分离失败，退回无标注: {spk_err}", exc_info=True)

        return (raw_text, engine_key, segments, diarized_text)
