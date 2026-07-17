"""Mock ASR engines for fast pipeline testing.

每个 mock 返回固定结果，不做任何真实推理/网络请求。
"""


class MockQwenEngine:
    """Mock Qwen engine — returns hardcoded results without HTTP calls."""

    def transcribe(self, audio_path, language="auto"):
        """Return a fake transcription result with word timestamps."""
        return {
            "text": "你好世界 今天天气不错 我们去散步吧",
            "srt": (
                "1\n00:00:00,000 --> 00:00:04,200\n"
                "你好世界 今天天气不错 我们去散步吧\n"
            ),
            "raw": {
                "timestamps": [
                    {"text": "你好", "start": 0.0, "end": 0.8},
                    {"text": "世界", "start": 0.8, "end": 1.2},
                    {"text": "今天", "start": 1.2, "end": 1.6},
                    {"text": "天气", "start": 1.6, "end": 2.0},
                    {"text": "不错", "start": 2.0, "end": 2.5},
                    {"text": "我们", "start": 3.0, "end": 3.3},
                    {"text": "去", "start": 3.3, "end": 3.5},
                    {"text": "散步", "start": 3.5, "end": 4.0},
                    {"text": "吧", "start": 4.0, "end": 4.2},
                ],
            },
        }

    def transcribe_batch(self, audio_paths, language="auto"):
        """Mock batch transcription — return one result per input."""
        raw = self.transcribe(None)
        return [
            {"text": raw["text"], "timestamps": raw["raw"]["timestamps"]}
            for _ in audio_paths
        ]


class MockSeACoEngine:
    """Mock SeACo engine — returns hardcoded segments with speaker info."""

    def __call__(self, audio_input, hotwords="", language="auto"):
        return {
            "text": "你好 今天天气不错",
            "segments": [
                {"start": 0, "end": 1500, "speaker": "1", "text": "你好"},
                {"start": 1500, "end": 3000, "speaker": "2", "text": "今天天气不错"},
            ],
        }


class MockSenseVoiceEngine:
    """Mock SenseVoice engine — returns plain text list."""

    def __call__(self, waveforms):
        if isinstance(waveforms, list):
            return ["你好世界"] * len(waveforms)
        return ["你好世界"]


class MockSherpaEngine:
    """Mock Sherpa ONNX engine — returns plain text list."""

    def __call__(self, wav_content):
        return ["你好世界"]


# 路由映射：把 _select_engine 返回的 key 对应到 mock 引擎
MOCK_ENGINES = {
    "qwen": MockQwenEngine(),
    "seaco": MockSeACoEngine(),
    "torch": MockSenseVoiceEngine(),
    "sherpa": MockSherpaEngine(),
}
