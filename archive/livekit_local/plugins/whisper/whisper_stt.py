import logging
import numpy as np
from typing import Optional, Dict, List, Any, Union

logger = logging.getLogger(__name__)

try:
    # Try to import faster_whisper if available
    import faster_whisper
    WHISPER_IMPLEMENTATION = "faster_whisper"
except ImportError:
    try:
        # Fall back to whisper if faster_whisper not available
        import whisper
        WHISPER_IMPLEMENTATION = "whisper"
    except ImportError:
        logger.error("Neither faster_whisper nor whisper could be imported. Make sure at least one is installed.")
        WHISPER_IMPLEMENTATION = None

# Import STTCapabilities from livekit
from livekit.agents.stt import STTCapabilities


class WhisperSTT:
    """
    A Speech-to-Text implementation using Whisper or faster_whisper locally.

    This implementation supports both the official OpenAI Whisper and the faster-whisper implementation.
    It will try to use faster-whisper first, and fall back to the original Whisper if not available.
    """

    def __init__(
        self,
        # model: str = "base",
        model: str = "large-v3",
        language: Optional[str] = None,
        translate: bool = False,
        # device: Optional[str] = None,
        device: Optional[str] = "cpu",
        # compute_type: str = "float16",
        compute_type: str = "int8",

    ):
        """
        Initialize the WhisperSTT implementation.

        Args:
            model: Whisper model to use ('tiny', 'base', 'small', 'medium', 'large')
                  or a path to a local model file.
            language: Language code to use for transcription (e.g., 'en', 'fr').
                     If None, Whisper will auto-detect the language.
            translate: Whether to translate non-English speech to English.
            device: Device to use for inference ('cpu', 'cuda', 'auto').
                   If None, will automatically choose the available device.
            compute_type: Compute type for faster_whisper ('float16', 'float32', 'int8').

        Raises:
            ImportError: If neither Whisper nor faster-whisper is installed.
        """
        if WHISPER_IMPLEMENTATION is None:
            raise ImportError(
                "Neither faster_whisper nor whisper is installed. "
                "Install at least one of them: "
                "pip install faster-whisper or pip install openai-whisper"
            )

        self.model_name = model
        self.language = language
        self.translate = translate
        self.device = device or "auto"
        self.compute_type = compute_type
        self.model = None

        # Add capabilities attribute that LiveKit expects
        self.capabilities = STTCapabilities(streaming=False, interim_results=False)

        logger.info(f"Initializing Whisper STT using {WHISPER_IMPLEMENTATION} implementation")
        self._load_model()

    def _load_model(self):
        """Load the Whisper model based on the selected implementation."""
        if WHISPER_IMPLEMENTATION == "faster_whisper":
            logger.info(f"Loading faster-whisper model: {self.model_name}")
            self.model = faster_whisper.WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type
            )
        else:  # Original whisper
            logger.info(f"Loading whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name, device=self.device)

    async def transcribe(self, audio: Union[bytes, np.ndarray], sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Transcribe audio data to text.

        Args:
            audio: Audio data as bytes or numpy array.
            sample_rate: Sample rate of the audio data.

        Returns:
            Dictionary containing:
                - text: The full transcription text
                - segments: List of segment dictionaries with text, start, end, and confidence
                - language: Detected language code
        """
        logger.debug(f"Transcribing audio with {WHISPER_IMPLEMENTATION}")

        # Convert audio to numpy array if needed
        if isinstance(audio, bytes):
            import io
            import soundfile as sf
            with io.BytesIO(audio) as buf:
                audio_array, _ = sf.read(buf)
        else:
            audio_array = audio

        # Ensure audio is float32 with values between -1 and 1
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        if audio_array.max() > 1.0:
            audio_array = audio_array / 32768.0  # Convert from int16 range to float32 range

        transcription_options = {
            "language": self.language,
        }

        if WHISPER_IMPLEMENTATION == "faster_whisper":
            segments, info = self.model.transcribe(
                audio_array,
                beam_size=5,
                task="translate" if self.translate else "transcribe",
                **transcription_options
            )

            # Process segments
            text_segments = []
            complete_text = ""

            for segment in segments:
                segment_dict = {
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "confidence": segment.avg_logprob
                }
                text_segments.append(segment_dict)
                complete_text += segment.text

            detected_language = info.language

        else:  # Original whisper
            # Whisper API is different
            result = self.model.transcribe(
                audio_array,
                task="translate" if self.translate else "transcribe",
                **transcription_options
            )

            complete_text = result["text"]
            detected_language = result.get("language")

            # Process segments
            text_segments = []
            for segment in result["segments"]:
                segment_dict = {
                    "text": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "confidence": segment.get("avg_logprob", 0)
                }
                text_segments.append(segment_dict)

        # Create and return the final result
        stt_result = {
            "text": complete_text.strip(),
            "segments": text_segments,
            "language": detected_language
        }

        logger.debug(f"Transcription completed: {complete_text[:30]}...")
        return stt_result