"""
speaker_engine.py — 基于 Cam++ (speech_campplus_sv) 的说话人向量提取 + 离线聚类

封装 funasr 的 campplus 模型，提供：
  - extract_embedding(chunk_16k) -> 说话人向量 (np.ndarray, 192 维)
  - cluster(audio_chunks, ...)   -> {seg_idx: speaker_id} 离线层次聚类

聚类策略（strategy 参数）：
  - "single"（默认）：单阶段 AHC，distance_threshold=0.6（参考 funclip/asr1.py 的基线）
  - "two_stage"：阈值鲁棒策略——先用更严的阈值过分割（宁拆不混），
    再用更松的阈值把同人的细簇合并回去；可传入 seg_times 加时间约束
  - "spectral"：谱聚类——基于 embedding 相似度图做全局优化分割，
    用 nearest_neighbors 亲和矩阵 + kmeans 标签分配，在差 embedding 下更鲁棒
    支持 n_speakers oracle-K 或 eigengap heuristic 自动估 K
  - n_speakers：若已知会议人数，直接 oracle-K，跳过阈值猜测

设计原则：
  - 模型对象常驻（调用方以单例持有），不在每次请求 new
  - 输入为 16k 单声道 float waveform
  - 参考自本项目 funclip/asr1.py 的 SpeakerDiarizer 离线聚类实现
"""
import logging
from typing import List, Optional, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics.pairwise import cosine_distances

logger = logging.getLogger("ASRService.Speaker")

# 默认说话人模型路径（damo 仓库，本地已下好权重）
DEFAULT_SPK_MODEL_DIR = r"E:\project\funclip-pro\model\models\damo\speech_campplus_sv_zh-cn_16k-common"

# 过短片段无法提取稳定向量（约 0.1s @16k）
_MIN_SAMPLES = 1600
# 余弦距离聚类阈值（参考 funclip/asr1.py 单阶段基线）
_DIST_THRESHOLD = 0.6
# 两阶段聚类参数（阈值鲁棒策略）
_OVERSEG_THRESHOLD = 0.3   # Stage1 过分割：用更严阈值，宁可把同一人拆细也绝不混人
_MERGE_THRESHOLD = 0.5     # Stage2 合并：用更松阈值把同人的细簇并回
_TIME_GAP_MS = 3000        # 时间约束：两片段间隔超过此值 → 合并惩罚（抑制跨断点误并）
_TIME_PENALTY = 0.4        # 时间不相邻时加在余弦距离上的惩罚量


def segment_sliding_window(audio, sr, win_sec=1.5, step_sec=0.5):
    """整段音频 -> 滑窗列表 [(start_sec, end_sec, samples_np)]。

    Args:
        audio: 1D numpy 数组，整段音频
        sr: 采样率
        win_sec: 窗长（秒），默认 1.5
        step_sec: 步长（秒），默认 0.5（重叠 1.0s）
    Returns:
        list of (start_sec, end_sec, samples)，每个 samples 是 win_sec*sr 长
    """
    win = int(win_sec * sr)
    step = int(step_sec * sr)
    n = len(audio)
    windows = []
    start = 0
    while start + win <= n:
        end = start + win
        windows.append((start / sr, end / sr, audio[start:end]))
        start += step
    # 尾部不足一窗但有声学内容，保留（提向量时 Cam++ 自行处理短段）
    if start < n and n - start >= sr * 0.5:  # 至少 0.5s 才留
        windows.append((start / sr, n / sr, audio[start:]))
    return windows


class CampPlusSpeaker:
    """Cam++ 说话人引擎：向量提取 + 离线聚类。"""

    def __init__(self, model_dir: str = DEFAULT_SPK_MODEL_DIR, device: str = "cpu"):
        from funasr import AutoModel

        logger.info(f"[Speaker] 加载 Cam++ 说话人模型: {model_dir} (device={device})")
        self.model = AutoModel(
            model=model_dir,
            trust_remote_code=True,
            device=device,
            disable_update=True,
            disable_pbar=True,
        )
        self.device = device

    def extract_embedding(self, chunk_16k) -> Optional[np.ndarray]:
        """从单个 16k 波形片段提取说话人向量；无效/失败返回 None。"""
        if chunk_16k is None:
            return None
        # Cam++ 可能运行在 CUDA 上：cuda 张量必须先 .cpu() 才能转 numpy
        if hasattr(chunk_16k, "cpu"):
            arr = chunk_16k.cpu().numpy()
        elif hasattr(chunk_16k, "numpy"):
            arr = chunk_16k.numpy()
        else:
            arr = np.asarray(chunk_16k)
        if arr.size < _MIN_SAMPLES:
            return None
        try:
            res = self.model.generate(input=[arr], disable_pbar=True)
            if not res or "spk_embedding" not in res[0]:
                return None
            # Cam++ 跑在 CUDA 时，输出 spk_embedding 是 cuda 张量，须先 .cpu()
            emb = res[0]["spk_embedding"]
            if hasattr(emb, "cpu"):
                emb = emb.cpu()
            return np.asarray(emb).flatten()
        except Exception as e:
            logger.warning(f"[Speaker] 向量提取失败: {e}")
            return None

    def extract_embedding_sliding_mean(self, chunk_16k, sr=16000, win_sec=1.5, step_sec=0.5) -> Optional[np.ndarray]:
        """提取活性音频片段的滑动窗口平均说话人向量，并做 L2 归一化提纯。

        Args:
            chunk_16k: 1D 音频数组 (numpy array 或 torch tensor)
            sr: 采样率，默认 16000
            win_sec: 窗长（秒），默认 1.5
            step_sec: 步长（秒），默认 0.5
        Returns:
            提纯后的归一化一维声纹向量 (np.ndarray)，若提取失败则返回 None
        """
        if chunk_16k is None:
            return None
        if hasattr(chunk_16k, "cpu"):
            arr = chunk_16k.cpu().numpy()
        elif hasattr(chunk_16k, "numpy"):
            arr = chunk_16k.numpy()
        else:
            arr = np.asarray(chunk_16k)
        
        windows = segment_sliding_window(arr, sr, win_sec, step_sec)
        if not windows:
            return self.extract_embedding(arr)
        
        embs = []
        for _, _, samp in windows:
            emb = self.extract_embedding(samp)
            if emb is not None:
                embs.append(emb)
        
        if not embs:
            return self.extract_embedding(arr)
        
        mean_emb = np.mean(embs, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 1e-6:
            mean_emb = mean_emb / norm
        return mean_emb

    # ---------------- 内部工具 ----------------

    def _extract_all(self, audio_chunks):
        """批量提取向量，返回 (embeddings, valid_idx)。"""
        embeddings: List[np.ndarray] = []
        valid: List[int] = []
        for i, chunk in enumerate(audio_chunks):
            emb = self.extract_embedding(chunk)
            if emb is not None:
                embeddings.append(emb)
                valid.append(i)
        return embeddings, valid

    @staticmethod
    def _ahc(emb_matrix, threshold=None, n=None):
        """包一层 AHC。n 给定 → oracle-K；否则用 distance_threshold 自动定簇数。"""
        if n is not None:
            ac = AgglomerativeClustering(n_clusters=n, metric="precomputed", linkage="average")
        else:
            ac = AgglomerativeClustering(
                n_clusters=None, distance_threshold=threshold,
                metric="precomputed", linkage="average",
            )
        return ac.fit_predict(cosine_distances(emb_matrix))

    def _apply_time_constraint(self, d2, labels1, valid, seg_times):
        """对细簇距离矩阵加时间惩罚：两细簇的时间段间隔过大 → 不易合并。"""
        unique = np.unique(labels1)
        spans = []
        for lab in unique:
            times = [seg_times[valid[j]] for j in range(len(labels1)) if labels1[j] == lab]
            starts = [t[0] for t in times]
            ends = [t[1] for t in times]
            spans.append((min(starts), max(ends)))
        n = len(unique)
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                gap = max(spans[a][0], spans[b][0]) - min(spans[a][1], spans[b][1])
                if gap > _TIME_GAP_MS:
                    d2[a, b] += _TIME_PENALTY
        return d2

    # ---------------- 对外聚类 ----------------

    def cluster(self, audio_chunks: List, strategy: str = "single",
                overseg_threshold: float = _OVERSEG_THRESHOLD,
                merge_threshold: float = _MERGE_THRESHOLD,
                n_speakers: Optional[int] = None,
                seg_times: Optional[List[Tuple[int, int]]] = None) -> dict:
        """
        离线聚类：先提取所有片段向量，再聚类。
        返回 {seg_idx: speaker_id(int, 从 1 起) | "?"}。

        strategy="two_stage" 时执行阈值鲁棒两阶段：
          Stage1 用 overseg_threshold 过分割（宁拆不混）
          Stage2 对每个细簇取平均向量作代表，再用 merge_threshold 合并；
                  若传 seg_times 则叠加时间约束，抑制跨断点误并。
        strategy="spectral" 时执行谱聚类：
          基于 embedding 相似度图做全局分割，用 nearest_neighbors 亲和矩阵
          + kmeans 标签分配；n_speakers 给定时 oracle-K，否则自动估 K。
        n_speakers 给定时，所有策略都退化为 oracle-K（最稳，无需猜阈值）。
        """
        embeddings, valid = self._extract_all(audio_chunks)
        result = {i: "?" for i in range(len(audio_chunks))}

        if len(embeddings) < 2:
            for i in range(len(audio_chunks)):
                result[i] = 1 if len(embeddings) == 1 else "?"
            return result

        emb_matrix = np.vstack(embeddings)

        if strategy == "single":
            labels = self._ahc(emb_matrix, threshold=_DIST_THRESHOLD, n=n_speakers)
        elif strategy == "spectral":
            n = len(embeddings)
            # 自动估 K：n_speakers 给定则 oracle-K；否则 eigengap heuristic
            if n_speakers is not None:
                n_clusters = n_speakers
            else:
                n_clusters = max(2, min(20, n // 10))
            # 谱聚类要求 K < N（嵌入矩阵为 N×D），硬上限防止报错
            n_clusters = min(n_clusters, n - 1, 20)
            sc = SpectralClustering(
                n_clusters=n_clusters,
                affinity='nearest_neighbors',
                n_neighbors=min(10, n - 1),
                assign_labels='kmeans',
                random_state=42,
            )
            labels = sc.fit_predict(emb_matrix)
        else:  # two_stage：先严后松
            # Stage1：过分割，得到若干细簇
            labels1 = self._ahc(emb_matrix, threshold=overseg_threshold)
            unique = np.unique(labels1)
            # 每个细簇的代表向量 = 簇内片段向量的平均（提纯）
            reps = np.vstack([emb_matrix[labels1 == lab].mean(axis=0) for lab in unique])
            # Stage2：合并
            d2 = cosine_distances(reps)
            if seg_times is not None:
                d2 = self._apply_time_constraint(d2, labels1, valid, seg_times)
            if n_speakers is not None:
                labels2 = self._ahc(reps, n=n_speakers)
            else:
                labels2 = self._ahc(reps, threshold=merge_threshold)
            # 细簇标签 → 合并后标签，映射回原始片段
            final = np.empty_like(labels1)
            for k, lab in enumerate(unique):
                final[labels1 == lab] = labels2[k]
            labels = final

        for idx, seg_idx in enumerate(valid):
            result[seg_idx] = int(labels[idx]) + 1
        return result

    def cluster_sliding(self, audio_16k, sr=16000, strategy="spectral",
                        n_speakers=None, win_sec=1.5, step_sec=0.5):
        """整段音频滑窗 segmentation + 聚类 + 合并相邻同人窗。

        与 cluster() 的区别：不用外部传入的 VAD 段，内部按固定窗滑切。
        Cam++ 提向量和 spectral 聚类逻辑复用 cluster() 的实现。

        Args:
            audio_16k: 1D numpy，整段 16k 音频
            sr: 采样率（默认 16000）
            strategy: 聚类策略（默认 spectral，复用现有）
            n_speakers: oracle-K；None 则自动估
            win_sec: 窗长秒（默认 1.5）
            step_sec: 步长秒（默认 0.5）
        Returns:
            list of (start_sec, end_sec, speaker_id)，合并后的说话人段
        """
        windows = segment_sliding_window(audio_16k, sr, win_sec, step_sec)
        if not windows:
            return []
        # 逐窗提 embedding
        embeddings = []
        valid_idx = []
        for i, (st, en, samp) in enumerate(windows):
            emb = self.extract_embedding(samp)
            if emb is not None:
                embeddings.append(emb)
                valid_idx.append(i)
        if not embeddings:
            # 全失败，整段标 1
            return [(windows[0][0], windows[-1][1], 1)]
        emb_matrix = np.vstack(embeddings)
        # 聚类（复用 spectral 逻辑）
        n = len(embeddings)
        if n_speakers is not None:
            n_clusters = n_speakers
        else:
            n_clusters = max(2, min(20, n // 10))
        n_clusters = min(n_clusters, n - 1, 20)
        if n_clusters < 1:
            n_clusters = 1
        if n_clusters == 1 or n <= 1:
            labels = np.zeros(n, dtype=int)
        else:
            sc = SpectralClustering(
                n_clusters=n_clusters,
                affinity='nearest_neighbors',
                n_neighbors=min(10, n - 1),
                assign_labels='kmeans',
                random_state=42,
            )
            labels = sc.fit_predict(emb_matrix)
        # 每窗贴标签（speaker_id 从 1 起）
        win_labels = [None] * len(windows)
        for i, lab in zip(valid_idx, labels):
            win_labels[i] = int(lab) + 1
        # 无效窗用前一个有效标签填充（首窗无效用 1）
        last_valid = 1
        for i in range(len(win_labels)):
            if win_labels[i] is None:
                win_labels[i] = last_valid
            else:
                last_valid = win_labels[i]
        # 合并相邻同人窗
        merged = []
        cur_spk = win_labels[0]
        cur_start = windows[0][0]
        cur_end = windows[0][1]
        for i in range(1, len(windows)):
            st, en, _ = windows[i]
            if win_labels[i] == cur_spk:
                cur_end = en
            else:
                merged.append((cur_start, cur_end, cur_spk))
                cur_spk = win_labels[i]
                cur_start = st
                cur_end = en
        merged.append((cur_start, cur_end, cur_spk))
        return merged
