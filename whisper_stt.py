import asyncio
import numpy as np
from typing import AsyncIterator

# Required imports from livekit
from livekit.rtc import AudioFrame
from livekit.agents.stt import (
    STT,
    STTCapabilities,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
)
# Import AudioBuffer for the recognize method
from livekit.agents.utils import AudioBuffer

# Whisper model import
from faster_whisper import WhisperModel

# Define constants for Whisper's expected audio format
WHISPER_SAMPLE_RATE = 16000
WHISPER_CHANNELS = 1
MIN_PROCESS_SAMPLES = int(WHISPER_SAMPLE_RATE * 0.5)  # Min audio needed (0.5s)

class WhisperSTT(STT):
    """
    Custom STT implementation using faster-whisper for offline transcription.
    Handles audio frames from LiveKit, converts them, and performs transcription
    in a non-blocking way. Includes both streaming (_recognize_impl) and
    non-streaming (recognize) methods.
    """
    def __init__(self, model="base", device="cpu", compute_type="int8", language="en"):
        """
        Initializes the WhisperSTT class.

        Args:
            model (str): The size of the Whisper model to load (e.g., "tiny", "base", "small").
            device (str): The device to run the model on ("cpu" or "cuda").
            compute_type (str): The computation type (e.g., "int8", "float16", "float32").
            language (str): The language code to force (e.g., "en" for English, "ja" for Japanese).
                           If None, Whisper will auto-detect the language.
        """
        self.language = language
        super().__init__(
            capabilities=STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )

        # --- Diagnostics ---
        print(f"--- WhisperSTT DIAGNOSTIC ---")
        print(f"Instance created: {self}")
        is_same_method = type(self)._recognize_impl == STT._recognize_impl
        print(f"Is type(self)._recognize_impl the same as STT._recognize_impl? {is_same_method}")
        print(f"WhisperSTT._recognize_impl: {type(self)._recognize_impl}")
        print(f"STT._recognize_impl: {STT._recognize_impl}")
        # Check if recognize is overridden
        is_same_recognize = type(self).recognize == STT.recognize
        print(f"Is type(self).recognize the same as STT.recognize? {is_same_recognize}")
        print(f"--- End Diagnostic ---")
        # --- End Diagnostics ---

        print(f"Loading faster-whisper model '{model}' on {device} with {compute_type} compute type...")
        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        print("Model loaded.")
        self._loop = asyncio.get_running_loop()

    def _frame_to_np(self, frame: AudioFrame) -> np.ndarray:
        """
        Converts a LiveKit AudioFrame to a NumPy array suitable for Whisper.
        Handles potential format mismatches with warnings and basic conversion.
        IMPORTANT: Assumes PCM16 input. Implement proper resampling if needed.
        """
        target_sr = WHISPER_SAMPLE_RATE
        target_ch = WHISPER_CHANNELS

        if frame.sample_rate != target_sr or frame.num_channels != target_ch:
            # print(f"Warning: Audio frame format ({frame.sample_rate} Hz, {frame.num_channels} ch) "
            #       f"does not match Whisper's expected ({target_sr} Hz, {target_ch} ch). "
            #       "Attempting basic conversion. Implement proper resampling for better results.")
            try:
                audio_data = np.frombuffer(frame.data, dtype=np.int16)
                if frame.num_channels > target_ch:
                    # Basic mixdown: average channels if stereo, else take first
                    if frame.num_channels == 2:
                         audio_data = audio_data.reshape(-1, 2).mean(axis=1).astype(np.int16)
                    else:
                         audio_data = audio_data[::frame.num_channels] # Select first channel

                # Convert to float32 and normalize
                audio_float = audio_data.astype(np.float32) / 32768.0

                # Placeholder for resampling - Libraries like librosa are recommended
                if frame.sample_rate != target_sr:
                    print(f"Warning: Resampling needed but not implemented ({frame.sample_rate} -> {target_sr}).")
                    # import librosa
                    # audio_float = librosa.resample(audio_float, orig_sr=frame.sample_rate, target_sr=target_sr)

                return audio_float
            except Exception as e:
                print(f"Error converting audio frame: {e}")
                return np.array([], dtype=np.float32)

        try:
            audio_data = np.frombuffer(frame.data, dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32768.0
            return audio_float
        except Exception as e:
            print(f"Error converting audio frame data: {e}")
            return np.array([], dtype=np.float32)

    # Streaming implementation (overrides base class)
    async def _recognize_impl(self, audio_stream: AsyncIterator[AudioFrame]) -> AsyncIterator[SpeechEvent]:
        """
        Implementation of the transcription stream.
        Receives AudioFrames, buffers them, and uses run_in_executor
        to call the blocking faster-whisper transcribe method with VAD.
        """
        is_speaking = False
        current_transcription = ""
        audio_buffer = np.array([], dtype=np.float32)
        MAX_BUFFER_SECONDS = 5 # Process in chunks up to this size
        MIN_PROCESS_SAMPLES = int(WHISPER_SAMPLE_RATE * 0.5) # Min audio needed (0.5s)
        MAX_BUFFER_SAMPLES = MAX_BUFFER_SECONDS * WHISPER_SAMPLE_RATE
        VAD_MIN_SILENCE_MS = 500 # VAD parameter

        print("Starting STT recognition stream...")

        async for frame in audio_stream:
            frame_np = self._frame_to_np(frame)
            if frame_np.size == 0: continue
            audio_buffer = np.concatenate((audio_buffer, frame_np))

            while len(audio_buffer) >= MAX_BUFFER_SAMPLES:
                segment_to_process = audio_buffer[:MAX_BUFFER_SAMPLES]
                audio_buffer = audio_buffer[MAX_BUFFER_SAMPLES:]
                try:
                    # Define a wrapper function to include all needed parameters
                    def transcribe_segment(audio_data):
                        return self.model.transcribe(
                            audio_data,
                            language=self.language,  # Force language to be the one specified in init
                            vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=VAD_MIN_SILENCE_MS)
                        )

                    segments, info = await self._loop.run_in_executor(
                        None, transcribe_segment, segment_to_process
                    )
                    segment_text = "".join(seg.text for seg in segments)
                    if segment_text.strip():
                        if not is_speaking:
                            yield SpeechEvent(type=SpeechEventType.START_OF_SPEECH)
                            is_speaking = True
                        current_transcription += segment_text
                        yield SpeechEvent(type=SpeechEventType.INTERIM_TRANSCRIPT, alternatives=[SpeechData(text=current_transcription)])
                        # TODO: Potentially yield FINAL_TRANSCRIPT based on VAD info if needed
                except Exception as e:
                    print(f"Whisper transcription error during stream: {e}")

        if len(audio_buffer) > MIN_PROCESS_SAMPLES:
            try:
                # Use a wrapper function to include all parameters
                def transcribe_final_segment(audio_data):
                    return self.model.transcribe(
                        audio_data,
                        language=self.language,  # Force language to be the one specified in init
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=VAD_MIN_SILENCE_MS)
                    )

                segments, info = await self._loop.run_in_executor(
                    None, transcribe_final_segment, audio_buffer
                )
                final_segment_text = "".join(seg.text for seg in segments)
                if final_segment_text.strip():
                    if not is_speaking:
                        yield SpeechEvent(type=SpeechEventType.START_OF_SPEECH)
                        is_speaking = True
                    current_transcription += final_segment_text
                    yield SpeechEvent(type=SpeechEventType.FINAL_TRANSCRIPT, alternatives=[SpeechData(text=current_transcription)])
            except Exception as e:
                print(f"Whisper final transcription error: {e}")

        if is_speaking:
            yield SpeechEvent(type=SpeechEventType.END_OF_SPEECH)
        # print("STT recognition stream finished.")


    # Non-streaming implementation (overrides base class)
    async def recognize(
        self,
        *,
        buffer: AudioBuffer,
        language: str | None = None, # Added language based on potential base class signature
        **kwargs,
    ) -> SpeechEvent:
        """
        Implements the non-streaming recognition method required by the base STT class.
        Uses the transcribe_once helper.
        Now accepts **kwargs to handle arguments like conn_options passed by StreamAdapter.
        """
        print("--- WhisperSTT DIAGNOSTIC: recognize() called ---")
        if not buffer:
             return SpeechEvent(type=SpeechEventType.FINAL_TRANSCRIPT, alternatives=[SpeechData(text="", language="")])

        # # Extract data and format from AudioBuffer
        # # Note: Need to confirm how AudioBuffer exposes its data (e.g., as bytes)
        # # Assuming buffer.data contains the byte data
        # audio_data = bytes(buffer.data) # This is an assumption! Verify AudioBuffer structure.
        # sample_rate = buffer.sample_rate
        # num_channels = buffer.num_channels

        # if not audio_data:
        #      return SpeechEvent(type=SpeechEventType.FINAL_TRANSCRIPT, data=SpeechData(text="", language=""))

        # try:
        #      # Call the existing helper method (which runs transcribe in executor)
        #      # Pass language if provided
        #      final_text = await self.transcribe_once(
        #          audio_data,
        #          sample_rate,
        #          num_channels,
        #          language=language # Pass language to helper
        #      )

        #      print(f"--- WhisperSTT recognize() result: '{final_text}' ---")
        #      # Return event with detected language if possible, else use provided/default
        #      detected_language = language or "" # Placeholder - transcribe_once might return language info
        #      return SpeechEvent(
        #          type=SpeechEventType.FINAL_TRANSCRIPT,
        #          data=SpeechData(text=final_text, language=detected_language)
        #      )
        # except Exception as e:
        #      print(f"Error during WhisperSTT recognize(): {e}")
        #      # Return an empty event on error
        #      return SpeechEvent(type=SpeechEventType.FINAL_TRANSCRIPT, data=SpeechData(text="", language=language or ""))
        # Extract data and format from AudioBuffer
        # Assuming buffer.data contains the byte data - VERIFY THIS ASSUMPTION
        # Based on typical usage, buffer itself might be iterable or have methods
        # Let's assume buffer directly provides frames or concatenated data compatible
        # with your existing transcribe_once helper which expects bytes, sample_rate, num_channels.

        # You need to get the raw bytes, sample_rate, and num_channels from the AudioBuffer
        # The AudioBuffer likely stores frames. You might need to combine them.
        # AudioBuffer in livekit.agents.utils is different from what we expected
        # It appears to be a single AudioFrame, not a collection of frames
        if not buffer or not buffer.data:
             return SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[SpeechData(text="", language="")]
             )

        # Use the single AudioFrame directly
        combined_data = buffer.data
        sample_rate = buffer.sample_rate
        num_channels = buffer.num_channels

        if not combined_data:
            return SpeechEvent(type=SpeechEventType.FINAL_TRANSCRIPT, alternatives=[SpeechData(text="", language="")])

        try:
            final_text = await self.transcribe_once(
                combined_data,         # Pass the combined bytes
                sample_rate,           # Pass the sample rate from the buffer
                num_channels,          # Pass the number of channels from the buffer
                language=language      # Pass language if provided
            )

            print(f"--- WhisperSTT recognize() result: '{final_text}' ---")
            detected_language = language or "" # Placeholder
            return SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[SpeechData(text=final_text, language=detected_language)]
            )
        except Exception as e:
            print(f"Error during WhisperSTT recognize(): {e}")
            # Log the exception traceback for debugging
            import traceback
            traceback.print_exc()
            return SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[SpeechData(text="", language=language or "")]
            )


    # Helper for single transcription task (used by recognize)
    async def transcribe_once(
        self,
        audio_data: bytes,
        sample_rate: int,
        num_channels: int,
        language: str | None = None # Added language parameter
        ) -> str:
        """
        Helper method for a one-off transcription of raw bytes.
        Requires format information. Handles conversion and runs model in executor.
        """
        audio_np = None
        target_sr = WHISPER_SAMPLE_RATE
        target_ch = WHISPER_CHANNELS

        # Conversion logic (similar to _frame_to_np but for raw bytes)
        try:
            _audio_data = np.frombuffer(audio_data, dtype=np.int16)
            if sample_rate != target_sr or num_channels != target_ch:
                print(f"Warning: transcribe_once input format ({sample_rate} Hz, {num_channels} ch) differs. Attempting conversion.")
                if num_channels > target_ch:
                     if num_channels == 2:
                         _audio_data = _audio_data.reshape(-1, 2).mean(axis=1).astype(np.int16)
                     else:
                         _audio_data = _audio_data[::num_channels]
                _audio_float = _audio_data.astype(np.float32) / 32768.0
                if sample_rate != target_sr:
                     print(f"Resampling audio from {sample_rate}Hz to {target_sr}Hz...")
                     # Basic resampling by linear interpolation
                     orig_len = len(_audio_float)
                     target_len = int(orig_len * target_sr / sample_rate)
                     indices = np.linspace(0, orig_len - 1, target_len)
                     indices = indices.astype(np.int32)
                     _audio_float = _audio_float[indices]
                audio_np = _audio_float
            else:
                audio_np = _audio_data.astype(np.float32) / 32768.0

        except Exception as e:
            print(f"Error converting audio in transcribe_once: {e}")
            return ""

        if audio_np is None or audio_np.size < MIN_PROCESS_SAMPLES: # Check size
             print(f"Error: Audio data too short or could not be processed in transcribe_once (size: {audio_np.size if audio_np is not None else 0}).")
             return ""

        try:
            # Create a wrapper function that encapsulates calling model.transcribe with the language parameter
            def transcribe_with_language(audio_data):
                # Use the provided language parameter, fall back to self.language (from init), or None
                lang_to_use = language if language else self.language
                print(f"Using language: {lang_to_use}")
                return self.model.transcribe(audio_data, language=lang_to_use)

            # Pass the wrapper function to run_in_executor
            segments, info = await self._loop.run_in_executor(
                None, transcribe_with_language, audio_np
            )
            # TODO: Could potentially return info.language here as well
            return "".join(seg.text for seg in segments)
        except Exception as e:
            print(f"Error in transcribe_once during transcription: {e}")
            return ""