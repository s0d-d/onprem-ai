# LiveKit Offline Voice Agent

This package provides offline components for creating a fully local voice agent using LiveKit Agents.

## Components

- **Whisper STT**: Speech-to-text using OpenAI's Whisper model
- **Piper TTS**: Text-to-speech using Piper TTS
- **Ollama LLM**: Language model integration using Ollama
- **Silero VAD**: Voice activity detection using Silero (already included in LiveKit)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/livekit-plugins-offline.git
cd livekit-plugins-offline

# Install the package
pip install -e .
```

## Prerequisites

1. **Ollama**: Install and run Ollama from https://ollama.ai
   ```bash
   # Pull a model like llama3
   ollama pull llama3
   ```

2. **Piper TTS**: Download a voice model from https://github.com/rhasspy/piper/releases
   ```bash
   # Example: Download and extract a voice model
   mkdir -p models/piper
   cd models/piper
   wget https://github.com/rhasspy/piper/releases/download/v1.0.0/voice-en-us-ryan-high.tar.gz
   tar -xzf voice-en-us-ryan-high.tar.gz
   ```

3. **Whisper**: No additional download needed (will be downloaded automatically)

## Usage

```bash
# Run the basic offline voice agent
python offline_voice_agent.py --piper-model=/path/to/piper/model.onnx

# Run with fallback options
python fallback_agent.py --primary-piper-model=/path/to/primary/model.onnx --fallback-piper-model=/path/to/fallback/model.onnx
```

## Example with different models

```bash
# Using medium Whisper model with a specific Ollama model
python offline_voice_agent.py --whisper-model=medium --ollama-model=mistral --piper-model=/path/to/piper/model.onnx
```
