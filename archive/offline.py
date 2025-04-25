import logging
import asyncio
import argparse
from typing import Dict, Any

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    RunContext,
    function_tool,
    metrics
)
from livekit.plugins import silero
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Import our custom local plugins
from livekit_local.plugins.whisper.whisper_stt import WhisperSTT
from livekit_local.plugins.piper.piper_tts import PiperTTS
from livekit_local.plugins.ollama.ollama_llm import OllamaLLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("offline-agent")


class OfflineAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful offline voice assistant named Kelly. "
                "Since you're communicating via voice, keep your responses concise and conversational. "
                "You are curious, friendly, and have a sense of humor. "
                "You're running completely locally on the user's device with no internet connection."
            ),
        )

    async def on_enter(self):
        # Generate greeting when the agent starts
        self.session.generate_reply(
            instructions="Greet the user warmly, introduce yourself as Kelly, and ask how you can help today."
        )

    @function_tool
    async def current_time(self, context: RunContext):
        """Returns the current system time when the user asks about the time."""
        from datetime import datetime
        current_time = datetime.now().strftime("%I:%M %p")
        return {
            "time": current_time
        }

    @function_tool
    async def simple_calculator(
        self,
        context: RunContext,
        operation: str,
        first_number: float,
        second_number: float
    ):
        """Performs basic mathematical operations.

        Args:
            operation: The operation to perform (add, subtract, multiply, divide)
            first_number: The first number in the operation
            second_number: The second number in the operation
        """
        result = None
        if operation.lower() == "add":
            result = first_number + second_number
        elif operation.lower() == "subtract":
            result = first_number - second_number
        elif operation.lower() == "multiply":
            result = first_number * second_number
        elif operation.lower() == "divide":
            if second_number == 0:
                return {"error": "Cannot divide by zero"}
            result = first_number / second_number
        else:
            return {"error": f"Unknown operation: {operation}"}

        return {
            "result": result,
            "operation": operation,
            "first_number": first_number,
            "second_number": second_number
        }


async def run_offline_agent(args: Dict[str, Any]):
    logger.info("Initializing offline voice agent components...")

    # Load Silero VAD (Voice Activity Detection)
    logger.info("Loading Silero VAD...")
    vad = silero.VAD.load()

    # Initialize STT with Whisper
    logger.info(f"Loading Whisper STT ({args.whisper_model})...")
    stt = WhisperSTT(model=args.whisper_model)

    # Initialize TTS with Piper
    logger.info(f"Loading Piper TTS from {args.piper_model}...")
    tts = PiperTTS(
        model_path=args.piper_model,
        config_path=args.piper_config
    )

    # Initialize LLM with Ollama
    logger.info(f"Connecting to Ollama at {args.ollama_host} with model {args.ollama_model}...")
    llm = OllamaLLM(
        model=args.ollama_model,
        host=args.ollama_host,
        temperature=args.temperature
    )

    # Create agent session
    session = AgentSession(
        vad=vad,
        llm=llm,
        stt=stt,
        tts=tts,
        turn_detection=MultilingualModel(),
    )

    # Set up metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    # Create agent with offline capabilities
    agent = OfflineAgent()

    logger.info("Starting the offline agent session...")

    # Start the session - this will need to be adapted for offline use
    # without requiring a LiveKit room
    try:
        # This is a placeholder for where you would start the offline session
        # The actual implementation will depend on how LiveKit's offline mode works
        await session.start_offline(
            agent=agent,
            mic_device=args.mic_device,
            speaker_device=args.speaker_device
        )

        # Keep the session running until interrupted
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    finally:
        # Clean up resources
        await session.stop()

        # Log usage summary
        summary = usage_collector.get_summary()
        logger.info(f"Usage summary: {summary}")

        logger.info("Offline agent session ended")


def main():
    parser = argparse.ArgumentParser(description='Run an offline voice agent')
    parser.add_argument('--whisper-model', type=str, default='base', help='Whisper model size (tiny, base, small, medium, large)')
    parser.add_argument('--piper-model', type=str, required=True, help='Path to Piper TTS model')
    parser.add_argument('--piper-config', type=str, default=None, help='Path to Piper TTS config')
    parser.add_argument('--ollama-model', type=str, default='llama3', help='Ollama model name')
    parser.add_argument('--ollama-host', type=str, default='http://localhost:11434', help='Ollama API host')
    parser.add_argument('--temperature', type=float, default=0.7, help='LLM temperature (0.0-1.0)')
    parser.add_argument('--mic-device', type=int, default=None, help='Microphone device ID')
    parser.add_argument('--speaker-device', type=int, default=None, help='Speaker device ID')

    args = parser.parse_args()

    try:
        asyncio.run(run_offline_agent(args))
    except Exception as e:
        logger.exception(f"Error running offline agent: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())