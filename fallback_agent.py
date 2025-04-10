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
from livekit.plugins.whisper import WhisperSTT
from livekit.plugins.piper import PiperTTS
from livekit.plugins.ollama import OllamaLLM

async def create_fallback_agent():
    parser = argparse.ArgumentParser(description='Run an offline voice agent with fallbacks')
    parser.add_argument('--primary-whisper-model', type=str, default='base', help='Primary Whisper model size')
    parser.add_argument('--fallback-whisper-model', type=str, default='tiny', help='Fallback Whisper model size')
    parser.add_argument('--primary-piper-model', type=str, required=True, help='Path to primary Piper TTS model')
    parser.add_argument('--fallback-piper-model', type=str, required=True, help='Path to fallback Piper TTS model')
    parser.add_argument('--primary-ollama-model', type=str, default='llama3', help='Primary Ollama model name')
    parser.add_argument('--fallback-ollama-model', type=str, default='llama2', help='Fallback Ollama model name')
    parser.add_argument('--ollama-host', type=str, default='http://localhost:11434', help='Ollama API host')
    parser.add_argument('--mic-device', type=int, default=None, help='Microphone device ID')
    parser.add_argument('--speaker-device', type=int, default=None, help='Speaker device ID')
    args = parser.parse_args()

    print("Initializing offline voice agent with fallbacks...")

    # Initialize primary and fallback STT
    print(f"Loading Whisper STT models (primary: {args.primary_whisper_model}, fallback: {args.fallback_whisper_model})...")
    primary_stt = WhisperSTT(model_path=args.primary_whisper_model)
    fallback_stt = WhisperSTT(model_path=args.fallback_whisper_model)

    # Create fallback STT adapter
    stt_adapter = FallbackAdapter(
        primary=primary_stt,
        fallbacks=[fallback_stt],
        max_retries=2
    )

    # Initialize primary and fallback TTS
    print(f"Loading Piper TTS models...")
    primary_tts = PiperTTS(model_path=args.primary_piper_model)
    fallback_tts = PiperTTS(model_path=args.fallback_piper_model)

    # Create fallback TTS adapter
    tts_adapter = FallbackAdapter(
        primary=primary_tts,
        fallbacks=[fallback_tts],
        max_retries=2
    )

    # Initialize primary and fallback LLM
    print(f"Connecting to Ollama with models (primary: {args.primary_ollama_model}, fallback: {args.fallback_ollama_model})...")
    primary_llm = OllamaLLM(
        model=args.primary_ollama_model,
        host=args.ollama_host,
        temperature=0.7
    )
    fallback_llm = OllamaLLM(
        model=args.fallback_ollama_model,
        host=args.ollama_host,
        temperature=0.7
    )

    # Create fallback LLM adapter
    llm_adapter = FallbackAdapter(
        primary=primary_llm,
        fallbacks=[fallback_llm],
        max_retries=2
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
        use_tools=True,
        debug=True
    )

    # Create analytics object
    analytics = VoiceAgentAnalytics()

    # Create voice agent with fallback adapters
    print("Creating voice agent with fallback adapters...")
    agent = VoiceAgent(
        stt=stt_adapter,
        tts=tts_adapter,
        llm=llm_adapter,
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

    return agent

async def main():
    agent = await create_fallback_agent()

    # Run the agent until interrupted
    try:
        print("Voice agent with fallbacks is running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping voice agent...")
    finally:
        await agent.stop()
        print("Voice agent stopped.")

if __name__ == "__main__":
    asyncio.run(main())