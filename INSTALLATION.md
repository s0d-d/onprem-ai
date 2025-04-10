# Detailed Installation Guide

This guide will walk you through the complete setup of the offline voice agent.

## System Requirements

- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.8 or newer
- **RAM**: At least 8GB (16GB+ recommended for larger models)
- **Disk Space**:
  - ~2GB for Whisper models
  - ~50-200MB for Piper voices
  - ~4-8GB for Ollama models
- **GPU**: Optional but recommended for faster inference

## Step 1: Install Dependencies

### 1.1. Install Python and pip

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

**macOS:**
```bash
brew install python
```

**Windows:**
Download and install Python from [python.org](https://www.python.org/downloads/)

### 1.2. Install Audio Dependencies

**Ubuntu/Debian:**
```bash
sudo apt install portaudio19-dev python3-pyaudio
```

**macOS:**
```bash
brew install portaudio
```

**Windows:**
PyAudio will be installed via pip in the next steps.

### 1.3. Install Ollama

Follow the installation instructions at [ollama.ai](https://ollama.ai):

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**macOS:**
Download and install from the [Ollama website](https://ollama.ai/download)

**Windows:**
Download and install from the [Ollama website](https://ollama.ai/download)

### 1.4. Start Ollama and Pull Models

```bash
# Start Ollama service
ollama serve

# In a new terminal, pull models
ollama pull llama3
# Optionally pull a fallback model
ollama pull llama2
```

## Step 2: Download Piper Voice Models

```bash
# Create a directory for voice models
mkdir -p models/piper
cd models/piper

# Download a voice model (example: en-us-ryan)
curl -LO https://github.com/rhasspy/piper/releases/download/v1.0.0/voice-en-us-ryan-high.tar.gz
tar -xzf voice-en-us-ryan-high.tar.gz

# Optional: Download a fallback voice model
curl -LO https://github.com/rhasspy/piper/releases/download/v1.0.0/voice-en-us-amy-low.tar.gz
tar -xzf voice-en-us-amy-low.tar.gz

# Return to main directory
cd ../..
```

## Step 3: Set Up the Project

### 3.1. Clone the Repository

```bash
git clone https://github.com/yourusername/livekit-plugins-offline.git
cd livekit-plugins-offline
```

### 3.2. Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3.3. Install the Package

```bash
# Install in development mode
pip install -e .

# Install additional requirements
pip install pyaudio sounddevice
```

## Step 4: Verify Installation

### 4.1. List Audio Devices

Create a script to list available audio devices:

```bash
cat > list_devices.py << 'EOF'
import sounddevice as sd

def list_devices():
    print("Available audio devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"Device {i}: {device['name']} (Input: {device['max_input_channels']}, Output: {device['max_output_channels']})")

if __name__ == "__main__":
    list_devices()
EOF

python list_devices.py
```

Note the device IDs for your microphone and speakers.

### 4.2. Run a Basic Test

```bash
# Test the basic voice agent
python offline_voice_agent.py \
  --whisper-model=tiny \
  --piper-model=/absolute/path/to/models/piper/en-us-ryan-high.onnx \
  --ollama-model=llama3 \
  --mic-device=YOUR_MIC_DEVICE_ID \
  --speaker-device=YOUR_SPEAKER_DEVICE_ID
```

### 4.3. Run with Fallback Support

```bash
python fallback_agent.py \
  --primary-whisper-model=base \
  --fallback-whisper-model=tiny \
  --primary-piper-model=/absolute/path/to/models/piper/en-us-ryan-high.onnx \
  --fallback-piper-model=/absolute/path/to/models/piper/en-us-amy-low.onnx \
  --primary-ollama-model=llama3 \
  --fallback-ollama-model=llama2 \
  --mic-device=YOUR_MIC_DEVICE_ID \
  --speaker-device=YOUR_SPEAKER_DEVICE_ID
```

### 4.4. Run with Extended Tools

```bash
python extended_agent_example.py \
  --whisper-model=base \
  --piper-model=/absolute/path/to/models/piper/en-us-ryan-high.onnx \
  --ollama-model=llama3 \
  --mic-device=YOUR_MIC_DEVICE_ID \
  --speaker-device=YOUR_SPEAKER_DEVICE_ID
```

## Step 5: Troubleshooting

### Audio Issues

If you experience audio issues:

1. Verify your microphone and speaker are working with other applications
2. Try different device IDs using the `--mic-device` and `--speaker-device` flags
3. Check audio format compatibility with:

```bash
python -c "import sounddevice as sd; print(sd.query_devices(device=YOUR_DEVICE_ID))"
```

### Model Loading Issues

If models fail to load:

1. Verify file paths are absolute (not relative)
2. Check file permissions
3. Verify model files are not corrupted (re-download if necessary)

### Performance Issues

If the agent is slow:

1. Try smaller models (tiny/base Whisper model, smaller Ollama model)
2. Reduce the chunk duration in WhisperSTT
3. Enable GPU acceleration if available

### Ollama Connection Issues

If Ollama fails to connect:

1. Verify Ollama is running with `ps aux | grep ollama`
2. Check Ollama API is accessible with `curl http://localhost:11434/api/tags`
3. Verify the model is downloaded with `ollama list`

## Step 6: Customization

### Adding Custom Tools

Extend the agent by creating custom tools in your script. See `extended_agent_example.py` for examples of how to implement:

- Note-taking
- Reminders
- Calendar integration
- Local file search
- Device control

### Changing Voice Models

Experiment with different Piper voices available at [Piper releases](https://github.com/rhasspy/piper/releases).

### Improving STT Accuracy

Customize Whisper settings by modifying the `WhisperSTT` class:

```python
whisper_stt = WhisperSTT(
    model_path="base",
    chunk_duration=1.5,  # Process audio in smaller chunks
    device="cuda"        # Use GPU if available
)
```