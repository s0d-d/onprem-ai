import os
import asyncio
import subprocess
import numpy as np
from typing import Optional, List, Dict, Any, AsyncIterator
from livekit.agents import Plugin
from livekit.agents.tts.tts import (
    TTS,
    TTSCapabilities,
    TTSOptions,
    ChunkedStream
)

class PiperTTSChunkedStream(ChunkedStream):
    def __init__(self, audio_data, chunk_size=1024):
        self.audio_data = audio_data
        self.chunk_size = chunk_size
        self.position = 0

    async def __anext__(self):
        if self.position >= len(self.audio_data):
            raise StopAsyncIteration

        chunk = self.audio_data[self.position:self.position + self.chunk_size]
        self.position += self.chunk_size

        # Small delay to simulate real-time streaming
        await asyncio.sleep(0.01)

        return chunk


class PiperTTS(TTS):
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        sample_rate: int = 22050,
        chunk_size: int = 1024,
        piper_binary_path: Optional[str] = None,
        espeak_data_path: Optional[str] = None
    ):
        super().__init__(capabilities=TTSCapabilities(streaming=True))
        self.model_path = model_path
        self.config_path = config_path
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        # Path to the piper binary and espeak data
        self.piper_binary_path = piper_binary_path or os.environ.get('PIPER_BINARY_PATH', 'piper')
        self.espeak_data_path = espeak_data_path or os.environ.get('ESPEAK_DATA_PATH', 'espeak-ng-data')

        # Check if binary exists
        if not os.path.exists(self.piper_binary_path) and self.piper_binary_path == 'piper':
            # Try to find piper in the same directory as this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            possible_path = os.path.join(current_dir, 'piper')
            if os.path.exists(possible_path):
                self.piper_binary_path = possible_path

    async def synthesize(
        self,
        text: str,
        options: Optional[TTSOptions] = None
    ) -> bytes:
        """Synthesize text to speech and return audio bytes."""
        # Process options
        voice_id = None
        if options and options.voice:
            voice_id = options.voice

        # Set up the command with all necessary flags
        cmd = [
            self.piper_binary_path,
            '--model', self.model_path,
            '--output_raw'  # Output raw audio data
        ]

        # Add config if provided
        if self.config_path:
            cmd.extend(['--config', self.config_path])

        # Set environment variables for library paths
        env = os.environ.copy()
        piper_dir = os.path.dirname(self.piper_binary_path)

        # Add piper directory to LD_LIBRARY_PATH
        if 'LD_LIBRARY_PATH' in env:
            env['LD_LIBRARY_PATH'] = f"{piper_dir}:{env['LD_LIBRARY_PATH']}"
        else:
            env['LD_LIBRARY_PATH'] = piper_dir

        # Set ESPEAK_DATA_PATH
        if os.path.exists(os.path.join(piper_dir, self.espeak_data_path)):
            env['ESPEAK_DATA_PATH'] = os.path.join(piper_dir, self.espeak_data_path)
        elif os.path.exists(self.espeak_data_path):
            env['ESPEAK_DATA_PATH'] = self.espeak_data_path

        # Run piper as a subprocess
        try:
            # Run the command with text as input
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )

            # Send text and get output
            stdout, stderr = await process.communicate(text.encode('utf-8'))

            if process.returncode != 0:
                error_msg = stderr.decode('utf-8')
                raise RuntimeError(f"Piper synthesis failed with code {process.returncode}: {error_msg}")

            # Convert output bytes to numpy array for streaming
            return stdout

        except Exception as e:
            raise RuntimeError(f"Piper synthesis failed: {e}")

    async def stream(
        self,
        text: str,
        options: Optional[TTSOptions] = None
    ) -> ChunkedStream:
        """Stream synthesized audio in chunks."""
        audio_bytes = await self.synthesize(text, options)
        return PiperTTSChunkedStream(audio_bytes, self.chunk_size)


# Register the plugin
class PiperTTSPlugin(Plugin):
    @classmethod
    def load(cls):
        return PiperTTS

Plugin.register_plugin("piper", PiperTTSPlugin)