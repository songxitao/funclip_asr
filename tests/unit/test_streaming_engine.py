"""P3.2 流式 ASR 引擎测试套件。

TDD 方式：先写测试（红），再实现（绿）。
不依赖实际硬件/模型，全部使用 mock。

⚠️ funclip_pro.core.__init__ 导入了 segmentation → torch → pyannote 等重型依赖，
   为纯单元测试环境采用 importlib 直接加载 streaming_asr.py 源文件的方式绕过。
"""

import importlib.machinery
import os
import sys

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# ========================
# 辅助：绕过 core.__init__ 导入 streaming_asr
# ========================

_STREAMING_ASR = None


def _import_streaming_asr():
    """直接加载 streaming_asr.py，绕过 funclip_pro.core.__init__ 的重型依赖链。

    返回 module 对象，各测试用例通过 module.SileroVAD 等形式访问。
    """
    global _STREAMING_ASR
    if _STREAMING_ASR is not None:
        return _STREAMING_ASR

    # 提前注册 funclip_pro 和子包（但不触发 core.__init__）
    import funclip_pro
    import funclip_pro.config

    # 定位 streaming_asr.py 的物理路径
    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(test_dir)
    module_path = os.path.join(
        project_root, "src", "funclip_pro", "core", "streaming_asr.py"
    )
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"streaming_asr.py not found: {module_path}")

    loader = importlib.machinery.SourceFileLoader(
        "_streaming_asr_test", module_path
    )
    _STREAMING_ASR = loader.load_module("_streaming_asr_test")
    return _STREAMING_ASR


# ========================
# Test 1: SileroVAD 纯逻辑
# ========================


class TestSileroVAD:
    """SileroVAD __call__ 输出范围 [0, 1] 且状态管理正确。"""

    def _setup_mock(self, input_names=None):
        """创建模拟的 ONNX InferenceSession。"""
        mock_sess = MagicMock()
        if input_names is None:
            input_names = ["input", "sr", "state"]

        names = []
        for name in input_names:
            m = MagicMock()
            m.name = name
            names.append(m)

        mock_sess.get_inputs.return_value = names
        return mock_sess

    @patch("os.path.exists", return_value=True)
    @patch("onnxruntime.InferenceSession")
    def test_call_output_range(self, mock_inference_session, mock_path_exists):
        """给定全零和正弦波帧，__call__ 输出应在 [0, 1] 范围内。"""
        streaming = _import_streaming_asr()

        mock_sess = self._setup_mock()
        mock_sess.run.return_value = (
            np.array([[0.3]], dtype=np.float32),
            np.zeros((2, 1, 128), dtype=np.float32),
        )
        mock_inference_session.return_value = mock_sess

        vad = streaming.SileroVAD("dummy/path.onnx")

        # 全零帧
        zero_chunk = np.zeros(512, dtype=np.float32)
        prob_zero = vad(zero_chunk)
        assert 0.0 <= prob_zero <= 1.0, f"概率应在 [0,1]，得到 {prob_zero}"

        # 正弦波帧
        t = np.arange(512, dtype=np.float32)
        sine_chunk = 0.5 * np.sin(2 * np.pi * 440 * t / 16000)
        prob_sine = vad(sine_chunk)
        assert 0.0 <= prob_sine <= 1.0, f"概率应在 [0,1]，得到 {prob_sine}"

    @patch("os.path.exists", return_value=True)
    @patch("onnxruntime.InferenceSession")
    def test_reset_states(self, mock_inference_session, mock_path_exists):
        """reset_states 应重置内部状态为全零。"""
        streaming = _import_streaming_asr()

        mock_sess = self._setup_mock()
        mock_sess.run.return_value = (
            np.array([[0.3]], dtype=np.float32),
            np.ones((2, 1, 128), dtype=np.float32),
        )
        mock_inference_session.return_value = mock_sess

        vad = streaming.SileroVAD("dummy/path.onnx")
        chunk = np.zeros(512, dtype=np.float32)
        vad(chunk)  # 让 state 变为非零（mock 返回 ones）

        # 重置
        vad.reset_states()
        # 再次调用时，state 应传全零
        mock_sess.run.reset_mock()
        vad(chunk)
        # run() 被调用为 run(None, ort_inputs)，ort_inputs 在 pos arg 1
        call_args = mock_sess.run.call_args[0]
        ort_inputs = call_args[1]
        assert np.all(ort_inputs["state"] == 0.0), "reset 后 state 应为零"

    @patch("os.path.exists", return_value=True)
    @patch("onnxruntime.InferenceSession")
    def test_h_c_state_variant(self, mock_inference_session, mock_path_exists):
        """支持 h/c 输入名变种的 SileroVAD 模型也要能正确处理。"""
        streaming = _import_streaming_asr()

        mock_sess = self._setup_mock(["input", "sr", "h", "c"])
        mock_sess.run.return_value = (
            np.array([[0.5]], dtype=np.float32),
            np.zeros((2, 1, 64), dtype=np.float32),
            np.zeros((2, 1, 64), dtype=np.float32),
        )
        mock_inference_session.return_value = mock_sess

        vad = streaming.SileroVAD("dummy/path.onnx")
        chunk = np.zeros(512, dtype=np.float32)
        prob = vad(chunk)
        assert 0.0 <= prob <= 1.0

    @patch("onnxruntime.InferenceSession")
    def test_model_not_found(self, mock_inference_session):
        """模型文件不存在应抛出 FileNotFoundError。"""
        streaming = _import_streaming_asr()
        mock_inference_session.side_effect = FileNotFoundError("模型未找到")
        with pytest.raises(FileNotFoundError):
            streaming.SileroVAD("nonexistent/path.onnx")


# ================================
# Test 2: FsmnVadStreaming 段落检测
# ================================


class TestFsmnVadStreaming:
    """FsmnVadStreaming 在有声音+静音序列中能正确返回段落时间。"""

    @patch("funasr.AutoModel")
    def test_detect_speech_segment(self, mock_auto_model):
        """模拟有声音后静音，应返回一个段落。"""
        streaming = _import_streaming_asr()
        vad = streaming.FsmnVadStreaming(chunk_size_ms=200)

        loud_chunk = np.random.randn(3200).astype(np.float32) * 0.1
        for _ in range(5):
            segs, acc = vad.process_chunk(loud_chunk)
            assert segs == [], "说话中不应返回完成段落"

        silent_chunk = np.zeros(3200, dtype=np.float32)
        segs = []
        for _ in range(15):
            segs, acc = vad.process_chunk(silent_chunk)
            if segs:
                break

        assert len(segs) > 0, "静音足够久后应返回语音段落"
        start_ms, end_ms = segs[0]
        assert end_ms > start_ms, "end_ms 应大于 start_ms"
        assert end_ms >= 1000, "5 个 200ms chunk 至少 1000ms"

    @patch("funasr.AutoModel")
    def test_no_paragraph_on_silence(self, mock_auto_model):
        """持续静音不应返回段落。"""
        streaming = _import_streaming_asr()
        vad = streaming.FsmnVadStreaming(chunk_size_ms=200)
        silent_chunk = np.zeros(3200, dtype=np.float32)

        for _ in range(20):
            segs, acc = vad.process_chunk(silent_chunk)
            assert segs == [], "纯静音不应触发段落完成"

    @patch("funasr.AutoModel")
    def test_short_speech_filtered(self, mock_auto_model):
        """说话不足 300ms 应该被过滤。"""
        streaming = _import_streaming_asr()
        vad = streaming.FsmnVadStreaming(chunk_size_ms=200)
        loud_chunk = np.random.randn(3200).astype(np.float32) * 0.1

        segs, acc = vad.process_chunk(loud_chunk)
        assert segs == []

        silent_chunk = np.zeros(3200, dtype=np.float32)
        segs = []
        for _ in range(12):
            segs, acc = vad.process_chunk(silent_chunk)
            if segs:
                break

        assert segs == [], "短于 300ms 的音频不应返回段落"

    @patch("funasr.AutoModel")
    def test_reset(self, mock_auto_model):
        """reset() 应清空所有内部状态。"""
        streaming = _import_streaming_asr()
        vad = streaming.FsmnVadStreaming(chunk_size_ms=200)
        loud_chunk = np.random.randn(3200).astype(np.float32) * 0.1

        vad.process_chunk(loud_chunk)
        assert vad.is_speaking is True
        assert len(vad.accumulated_audio) > 0

        vad.reset()
        assert vad.is_speaking is False
        assert len(vad.accumulated_audio) == 0
        assert vad.cache == {}


# ===========================================
# Test 3: FunAsrStreamingEngine 会话生命周期
# ===========================================


class TestFunAsrStreamingEngine:
    """引擎会话隔离与基本生命周期。"""

    def _make_engine(self):
        """创建引擎实例，mock 内部懒加载方法使其不加载真实模型。"""
        streaming = _import_streaming_asr()
        engine = streaming.FunAsrStreamingEngine({})
        engine._ensure_vad = MagicMock()
        engine._ensure_asr = MagicMock()
        engine._vad_model = MagicMock()
        engine._vad_model.return_value = 0.1
        engine._asr_model = MagicMock()
        engine._asr_model.generate.return_value = [{"text": "测试文本"}]
        return engine

    def test_create_session_unique(self):
        """每次 create_session 返回唯一 ID。"""
        engine = self._make_engine()
        sid1 = engine.create_session()
        sid2 = engine.create_session()
        assert sid1 != sid2, "每次 create_session 应返回唯一 ID"

    def test_session_isolation(self):
        """两个 session 的缓冲区互不干扰。"""
        engine = self._make_engine()
        sid_a = engine.create_session()
        sid_b = engine.create_session()

        assert sid_a in engine._sessions
        assert sid_b in engine._sessions
        assert engine._sessions[sid_a] is not engine._sessions[sid_b]

    def test_session_lifecycle(self):
        """创建 → feed → 销毁，不泄漏。"""
        engine = self._make_engine()
        sid = engine.create_session()
        assert sid in engine._sessions, "创建后应在 sessions 中"

        chunk = np.zeros(512, dtype=np.float32)
        result = engine.feed_chunk(sid, chunk)
        assert isinstance(result, list), "feed_chunk 应返回 list"

        engine.destroy_session(sid)
        assert sid not in engine._sessions, "销毁后不应在 sessions 中"

    def test_feed_chunk_invalid_session(self):
        """feed_chunk 使用无效 session_id 应抛出 KeyError。"""
        engine = self._make_engine()
        chunk = np.zeros(512, dtype=np.float32)
        with pytest.raises(KeyError):
            engine.feed_chunk("nonexistent-session", chunk)

    def test_destroy_nonexistent_session(self):
        """销毁不存在的 session 应静默忽略。"""
        engine = self._make_engine()
        engine.destroy_session("nonexistent-id")


# ===========================================
# Test 4: 引擎初始化与配置
# ===========================================


class TestFunAsrStreamingEngineInit:
    """引擎初始化中的配置管理和模型加载。"""

    def test_module_importable(self):
        """模块应可被直接加载，不触发重型依赖。"""
        streaming = _import_streaming_asr()
        assert hasattr(streaming, "FunAsrStreamingEngine")
        assert hasattr(streaming, "SileroVAD")
        assert hasattr(streaming, "FsmnVadStreaming")
