# LiveKit Offline Voice Agent

A fully offline voice agent implementation using LiveKit Agents framework. This project allows you to create a completely local voice assistant without any cloud dependencies.

## Features

- 🎙️ **Offline Speech-to-Text** with Whisper
- 🔊 **Offline Text-to-Speech** with Piper
- 🧠 **Offline Language Model** with Ollama
- 🎯 **Voice Activity Detection** with Silero
- 🔄 **Fallback Support** for robustness
- 🛠️ **Custom Tools Support** for extended functionality

## Prerequisites

1. **Python 3.8+** and pip

2. **Ollama**: Install and run Ollama from https://ollama.ai
   ```bash
   # Pull a model like llama3
   ollama pull llama3
   ```

3. **Piper TTS**: Download a voice model from https://github.com/rhasspy/piper/releases
   ```bash
   # Example: Download and extract a voice model
   mkdir -p models/piper
   cd models/piper
   wget https://github.com/rhasspy/piper/releases/download/v1.0.0/voice-en-us-ryan-high.tar.gz
   tar -xzf voice-en-us-ryan-high.tar.gz
   ```

4. **CUDA (Optional)**: For faster processing with NVIDIA GPUs

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/livekit-plugins-offline.git
cd livekit-plugins-offline

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

## Usage

### Basic Offline Voice Agent

```bash
python offline_voice_agent.py --piper-model=/path/to/piper/model.onnx --ollama-model=llama3
```

### With Fallback Support

```bash
python fallback_agent.py \
  --primary-whisper-model=base \
  --fallback-whisper-model=tiny \
  --primary-piper-model=/path/to/primary/model.onnx \
  --fallback-piper-model=/path/to/fallback/model.onnx \
  --primary-ollama-model=llama3 \
  --fallback-ollama-model=llama2
```

### Advanced Configuration

```bash
python offline_voice_agent.py \
  --whisper-model=medium \
  --piper-model=/path/to/piper/model.onnx \
  --piper-config=/path/to/piper/config.json \
  --ollama-model=mistral \
  --ollama-host=http://localhost:11434 \
  --mic-device=1 \
  --speaker-device=2
```

## Project Structure

```
livekit-plugins-offline/
├── livekit/
│   └── plugins/
│       ├── whisper/         # Whisper STT implementation
│       ├── piper/           # Piper TTS implementation
│       └── ollama/          # Ollama LLM implementation
├── offline_voice_agent.py   # Main agent script
├── fallback_agent.py        # Agent with fallback support
└── setup.py                 # Installation script
```

## Custom Tool Integration

You can extend the voice agent with custom tools by implementing them in your application:

```python
from livekit.agents import ToolDefinition
from livekit.agents.llm.llm import LLMTools

# Define a custom tool
weather_tool = ToolDefinition(
    name="weather",
    description="Get the current weather for a location",
    parameters={
        "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA"
        }
    }
)

# Create tool handler
async def handle_weather_tool(name, parameters):
    location = parameters.get("location", "")
    # Implement local weather lookup here (e.g., from a local database)
    return f"The weather in {location} is sunny and 72°F"

# Register tool handler
tools = LLMTools(
    available_tools=[weather_tool],
    tool_handler=handle_weather_tool
)

# Use in your agent
agent = VoiceAgent(
    stt=whisper_stt,
    tts=piper_tts,
    llm=ollama_llm,
    vad=silero_vad,
    options=VoiceAgentOptions(
        use_tools=True,
        initial_prompt="You are a helpful voice assistant with weather information capabilities."
    ),
    llm_tools=tools  # Pass the tools here
)
```

## Troubleshooting

- **Audio device issues**: Use `--mic-device` and `--speaker-device` to specify correct audio devices
- **Model loading errors**: Ensure models are downloaded correctly and paths are absolute
- **Performance issues**: Try smaller models on less powerful hardware
- **Ollama connection errors**: Ensure Ollama is running and accessible at the specified host

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.