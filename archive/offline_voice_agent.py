import asyncio
import os
import argparse
from livekit.agents import (
    VoiceAgent,
    VoiceAgentOptions,
    VoiceAgentAnalytics,
    FallbackAdapter
)
from livekit.plugins import silero

# Import our custom plugins
from livekit_local.plugins.whisper import WhisperSTT
from livekit_local.plugins.piper import PiperTTS
from livekit_local.plugins.ollama import OllamaLLM

async def main():
    parser = argparse.ArgumentParser(description='Run an offline voice agent')
    parser.add_argument('--whisper-model', type=str, default='base', help='Whisper model size (tiny, base, small, medium, large)')
    parser.add_argument('--piper-model', type=str, required=True, help='Path to Piper TTS model')
    parser.add_argument('--piper-config', type=str, default=None, help='Path to Piper TTS config')
    parser.add_argument('--ollama-model', type=str, default='llama3', help='Ollama model name')
    parser.add_argument('--ollama-host', type=str, default='http://localhost:11434', help='Ollama API host')
    parser.add_argument('--mic-device', type=int, default=None, help='Microphone device ID')
    parser.add_argument('--speaker-device', type=int, default=None, help='Speaker device ID')
    args = parser.parse_args()

    print("Initializing offline voice agent components...")

    # Initialize STT with Whisper
    print(f"Loading Whisper STT ({args.whisper_model})...")
    whisper_stt = WhisperSTT(model_path=args.whisper_model)

    # Initialize TTS with Piper
    print(f"Loading Piper TTS from {args.piper_model}...")
    piper_tts = PiperTTS(
        model_path=args.piper_model,
        config_path=args.piper_config
    )

    # Initialize LLM with Ollama
    print(f"Connecting to Ollama at {args.ollama_host} with model {args.ollama_model}...")
    ollama_llm = OllamaLLM(
        model=args.ollama_model,
        host=args.ollama_host,
        temperature=0.7
    )

    # Load Silero VAD
    print("Loading Silero VAD...")
    silero_vad = silero.VAD.load()

    # Create voice agent options
    options = VoiceAgentOptions(
        initial_prompt=(
            "You are a helpful voice assistant. Keep your responses concise and conversational. "
            "You are running completely offline on the user's device."
        ),
        use_tools=True,  # Enable tool usage if needed
        debug=True  # Enable debugging for development
    )

    # Create analytics object
    analytics = VoiceAgentAnalytics()

    # Create voice agent
    print("Creating voice agent...")
    agent = VoiceAgent(
        stt=whisper_stt,
        tts=piper_tts,
        llm=ollama_llm,
        vad=silero_vad,
        options=options,
        analytics=analytics
    )

    # Start the voice agent
    print("Starting voice agent...")
    await agent.start(
        microphone_device=args.mic_device,
        speaker_device=args.speaker_device
    )

    # Run the agent until interrupted
    try:
        print("Voice agent is running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping voice agent...")
    finally:
        await agent.stop()
        print("Voice agent stopped.")

if __name__ == "__main__":
    asyncio.run(main())