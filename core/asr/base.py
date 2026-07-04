from abc import ABC, abstractmethod

class ASREngine(ABC):
    def __init__(self, device="cuda", **kwargs):
        self.device = device
        self.log_callback = print # Default to print

    def set_log_callback(self, callback):
        """Set a callback function for UI logging"""
        self.log_callback = callback

    def log(self, msg):
        self.log_callback(msg)

    @abstractmethod
    def load_model(self):
        """Load the model into memory"""
        pass

    @abstractmethod
    def transcribe(self, audio_path, language="auto", batch_size=None, **kwargs):
        """
        Transcribe audio file.
        Returns:
            {
                "text": "Full text",
                "srt": "SRT content",
                "segments":List[{"start":0.0, "end":1.0, "text":"..."}]
            }
        """
        pass
