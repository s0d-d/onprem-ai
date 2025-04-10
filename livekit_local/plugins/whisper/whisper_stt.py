import os
import numpy as np
import torch
from typing import Optional, List, Dict, Any, AsyncIterator
from livekit.agents import Plugin
from livekit.agents.stt.stt import (
    STT,
    STTCapabilities,
    AudioBuffer,
    RecognizedText,
    RecognizeStream,
    StreamingRecognizeRequest,
    RecognizedResult
)
from livekit.plugins import NOT_GIVEN, DEFAULT_API_CONNECT_OPTIONS

class WhisperRecognizeStream(RecognizeStream):
    def __init__(self, stt, language, conn_options):
        super().__init__()
        self.stt = stt
        self.language = language
        self.conn_options = conn_options
        self.buffer = AudioBuffer()
        self.is_running = False
        self.current_text = ""

    async def _start(self) -> None:
        self.is_running = True

    async def _stop(self) -> None:
        self.is_running = False
        if len(self.buffer) > 0:
            result = await self.stt._recognize_impl(self.buffer, self.language, self.conn_options)
            if result and result.text:
                self.current_text = result.text
                await self._emit_result(RecognizedResult(text=result.text, is_final=True))

    async def feed_audio(self, chunk: bytes) -> None:
        if not self.is_running:
            return

        self.buffer.append(chunk)

        # Process audio in chunks (e.g., every 1 second of audio)
        if len(self.buffer) >= self.stt.chunk_duration * self.stt.sample_rate * 2:  # 16-bit audio = 2 bytes per sample
            result = await self.stt._recognize_impl(self.buffer, self.language, self.conn_options)
            if result and result.text:
                self.current_text = result.text
                await self._emit_result(RecognizedResult(text=result.text, is_final=False))

            # Clear buffer for next chunk
            self.buffer.clear()


class WhisperSTT(STT):
    def __init__(
        self,
        model_path: str = "base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        chunk_duration: float = 2.0,  # Process audio in 2-second chunks
        sample_rate: int = 16000
    ):
        super().__init__(capabilities=STTCapabilities(streaming=True, interim_results=True))
        self.model_path = model_path
        self.device = device
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate

        # Import whisper here to avoid dependency if not used
        import whisper

        # Load the Whisper model
        self.model = whisper.load_model(model_path, device=device)

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        language: Optional[str] = None,
        conn_options: Dict[str, Any] = DEFAULT_API_CONNECT_OPTIONS
    ) -> Optional[RecognizedText]:
        if len(buffer) == 0:
            return None

        # Convert audio buffer to numpy array
        audio_data = buffer.to_array()

        # Perform speech recognition
        options = {}
        if language is not None and language != NOT_GIVEN:
            options["language"] = language

        try:
            result = self.model.transcribe(
                audio_data,
                **options
            )

            return RecognizedText(text=result["text"].strip())
        except Exception as e:
            print(f"Error in Whisper transcription: {e}")
            return None

    def stream(
        self,
        language=NOT_GIVEN,
        conn_options=DEFAULT_API_CONNECT_OPTIONS
    ) -> RecognizeStream:
        return WhisperRecognizeStream(self, language, conn_options)


# Register the plugin
class WhisperSTTPlugin(Plugin):
    @classmethod
    def load(cls):
        return WhisperSTT

Plugin.register_plugin("whisper", WhisperSTTPlugin)