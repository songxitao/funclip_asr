import os
import tempfile
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from funclip_pro.core.asr import QwenEngine
from funclip_pro.pipeline.offline import OfflinePipeline

# 1. 模拟 requests.post 以供 QwenEngine 测试
@pytest.fixture
def mock_requests_post():
    with patch("requests.post") as mock_post:
        # 模拟响应数据
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "text": "测试一",
                    "timestamps": [
                        {"text": "测", "start": 100, "end": 200},
                        {"text": "试", "start": 200, "end": 300},
                        {"text": "一", "start": 300, "end": 400}
                    ]
                },
                {
                    "text": "测试二",
                    "timestamps": [
                        {"text": "测", "start": 150, "end": 250},
                        {"text": "试", "start": 250, "end": 350},
                        {"text": "二", "start": 350, "end": 450}
                    ]
                }
            ]
        }
        mock_post.return_value = mock_response
        yield mock_post


def test_qwen_engine_transcribe_batch(mock_requests_post):
    """验证 QwenEngine.transcribe_batch 是否能正常编码并批量发送"""
    engine = QwenEngine()
    
    # 创建一些假文件来当参数
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f1, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f2:
        path1 = f1.name
        path2 = f2.name
        f1.write(b"fake data 1")
        f2.write(b"fake data 2")

    try:
        results = engine.transcribe_batch([path1, path2], language="zh")
        
        # 校验请求载荷
        assert mock_requests_post.called
        called_args, called_kwargs = mock_requests_post.call_args
        payload = called_kwargs["json"]
        
        # 验证批量 Base64 参数和映射过的语言
        assert "audio_batch_base64" in payload
        assert len(payload["audio_batch_base64"]) == 2
        assert payload["language"] == "Chinese"
        
        # 校验解析返回
        assert len(results) == 2
        assert results[0]["text"] == "测试一"
        assert len(results[0]["timestamps"]) == 3
        assert results[1]["text"] == "测试二"
    finally:
        # 清理临时假文件
        if os.path.exists(path1): os.remove(path1)
        if os.path.exists(path2): os.remove(path2)


def test_qwen_engine_transcribe_batch_numpy(mock_requests_post):
    """验证 QwenEngine.transcribe_batch 处理 numpy 数组时走单次 Base64 批量路径"""
    engine = QwenEngine()

    # 创建假 numpy 音频数据（2 个短片段）
    chunk1 = np.zeros(16000, dtype=np.float32)  # 1秒静音 @16kHz
    chunk2 = np.ones(16000, dtype=np.float32)   # 1秒直流 @16kHz
    chunks = [chunk1, chunk2]

    results = engine.transcribe_batch(chunks, language="zh")

    # 校验调用
    assert mock_requests_post.called
    called_args, called_kwargs = mock_requests_post.call_args

    # 验证请求 URL 是 batch_transcriptions
    assert "/v1/audio/batch_transcriptions" in called_args[0]

    # 验证 payload 结构
    payload = called_kwargs["json"]
    assert "audio_batch_base64" in payload
    assert len(payload["audio_batch_base64"]) == 2
    assert payload["return_timestamps"] == True
    assert payload["language"] == "Chinese"

    # 验证 Base64 内容是有效的 WAV
    import base64
    import io
    wav_bytes = base64.b64decode(payload["audio_batch_base64"][0])
    assert wav_bytes.startswith(b"RIFF")  # WAV header

    # 校验解析返回
    assert len(results) == 2
    assert results[0]["text"] == "测试一"


# 2. 模拟 OfflinePipeline 中 Qwen 分割转写逻辑
@patch("funclip_pro.pipeline.offline.resolve_model_path", return_value="fake_model_path")
@patch("funclip_pro.pipeline.offline.load_models")
@patch("librosa.load")
@patch("funclip_pro.core.asr._select_engine", return_value="qwen")
@patch("funclip_pro.pipeline.offline.OfflinePipeline._get_spk_model")
@patch("funclip_pro.pipeline.offline.OfflinePipeline._get_seg_model")
def test_offline_pipeline_qwen_vad_branch(
    mock_seg, mock_spk, mock_select, mock_load_audio, mock_load_models, mock_resolve
):
    """验证 OfflinePipeline 在 Qwen 引擎下执行 VAD 分割和绝对时间戳校准"""
    # 模拟音频时长：10秒 (16000Hz采样)
    mock_load_audio.return_value = (np.zeros(160000), 16000)
    
    pipeline = OfflinePipeline(auto_load=False)
    
    # 模拟 VAD 结果：两段音频 (0-4秒，6-10秒)
    # VAD_MODEL 产生的段是 [[0, 4000], [6000, 10000]]，单位是 ms
    mock_vad = MagicMock()
    mock_vad.generate.return_value = [{"value": [[0, 4000], [6000, 10000]]}]
    
    # 模拟 QwenEngine 及其批量转写结果
    mock_qwen = MagicMock()
    mock_qwen.transcribe_batch.return_value = [
        {
            "text": "测试一",
            "timestamps": [
                {"text": "测", "start": 0.1, "end": 0.2}, # 注意：timestamps 可能是以秒(sec)返回的，或者毫秒。
                {"text": "试", "start": 0.2, "end": 0.3},
                {"text": "一", "start": 0.3, "end": 0.4}
            ]
        },
        {
            "text": "测试二",
            "timestamps": [
                {"text": "测", "start": 0.1, "end": 0.2},
                {"text": "试", "start": 0.2, "end": 0.3},
                {"text": "二", "start": 0.3, "end": 0.4}
            ]
        }
    ]

    # 将 ASR 核心包和 VAD MODEL 进行 patch 绑定
    with patch("funclip_pro.pipeline.offline.asr_mod") as mock_asr_mod, \
         patch("funclip_pro.core.asr.QwenEngine", return_value=mock_qwen):
         
         mock_asr_mod._select_engine.return_value = "qwen"
         mock_asr_mod.VAD_MODEL = mock_vad
         # _use_vad(vad_strategy, duration_ms) -> True
         mock_asr_mod._use_vad.return_value = True
         # _merge_vad_segments 返回合并后的 VAD 段 (ms)
         mock_asr_mod._merge_vad_segments.return_value = [(0, 4000), (6000, 10000)]
         
         # 触发离线转写运行
         raw_text, engine, segments, diarized = pipeline.run(
             "fake_audio.wav", vad_strategy="auto", engine="qwen", language=["zh"]
         )
         # 1. 验证 VAD 是否被正确调用
         assert mock_vad.generate.called
         
         # 2. 验证 transcribe_batch 被调用，且参数为 numpy 数组（纯内存路径）
         assert mock_qwen.transcribe_batch.called
         call_args, call_kwargs = mock_qwen.transcribe_batch.call_args
         assert call_args and len(call_args) > 0
         chunk_arg = call_args[0]
         assert isinstance(chunk_arg, list) and len(chunk_arg) == 2
         assert isinstance(chunk_arg[0], np.ndarray)  # 传的是 numpy 切片，不是文件路径
         assert call_kwargs.get("language") == "zh"  # pipeline 传原始语言参数，映射在 transcribe_batch 内部
         
         # 3. 校验绝对时间轴对齐 (核心重点)
         # 第一段的偏移是 0ms，所以时间戳保持原样 (单位可能是秒，换算为 ms: 0.1 -> 100)
         # 第二段的偏移是 6000ms，所以时间戳应该累加偏移量: 0.1 -> 6100ms
         assert len(segments) == 6
         
         # 校验对齐后的时间戳 (以毫秒 ms 为单位)
         # 对应“测试一”
         assert segments[0]["start"] == 100
         assert segments[0]["end"] == 200
         assert segments[0]["text"] == "测"
         
         # 对应“测试二”
         assert segments[3]["start"] == 6100  # 6000 + 100
         assert segments[3]["end"] == 6200    # 6000 + 200
         assert segments[3]["text"] == "测"
         
         assert segments[5]["start"] == 6300  # 6000 + 300
         assert segments[5]["end"] == 6400    # 6000 + 400
         assert segments[5]["text"] == "二"
