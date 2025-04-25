from faster_whisper import WhisperModel

class FasterWhisperSTT:
    def __init__(self):
        self.model = WhisperModel("base.en", compute_type="int8")

    async def transcribe(self, audio_path: str) -> str:
        segments, _ = self.model.transcribe(audio_path)
        return " ".join(segment.text for segment in segments)
