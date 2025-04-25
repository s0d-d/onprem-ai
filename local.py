import asyncio
import logging
import aiohttp # Make sure this import is present
import os
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
# from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from livekit.agents.stt import StreamAdapter

from whisper_stt import WhisperSTT

# uncomment to enable Krisp background voice/noise cancellation
# currently supported on Linux and MacOS
# from livekit.plugins import noise_cancellation

logger = logging.getLogger("basic-agent")

load_dotenv()

# Get service URLs from environment variables, with fallbacks to localhost for local development
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "ws://localhost:7880")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL=os.environ.get("OLLAMA_MODEL", "smollm2:135m")
KOKORO_BASE_URL = os.environ.get("KOKORO_BASE_URL", "http://localhost:8880/v1")

class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Your name is Kelly. You would interact with users via voice."
            "with that in mind keep your responses concise and to the point."
            "You are curious and friendly, and have a sense of humor.",
        )

    async def on_enter(self):
        # when the agent is added to the session, it'll generate a reply
        # according to its instructions
        self.session.generate_reply(instructions="greet the user and ask about their day")



def prewarm(proc: JobProcess):
    # Pre-load and cache the VAD model
    logger.info("Prewarming VAD model...")
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("VAD model loaded successfully")


async def entrypoint(ctx: JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "user_id": "user_id",
    }

    # First verify Ollama is running
    try:
        logger.info(f"Checking Ollama server availability at {OLLAMA_BASE_URL}...")

        async with aiohttp.ClientSession() as session:
            # Extract the base URL without the '/v1' part for the tags endpoint
            base_url = OLLAMA_BASE_URL.rsplit('/v1', 1)[0]
            tags_url = f"{base_url}/api/tags"
            logger.info(f"Connecting to Ollama tags API at: {tags_url}")

            async with session.get(tags_url, timeout=5) as response:
                if response.status != 200:
                    logger.error(f"Ollama server not responding properly: {response.status}")
                    return
                logger.info("Ollama server is available")
    except Exception as e:
        logger.error(f"Failed to connect to Ollama server: {e}")
        logger.error(f"Tried connecting to: {OLLAMA_BASE_URL}")
        logger.error("Make sure Ollama is running and accessible")
        return

    # Connect to the room first - this is critical
    try:
        logger.info(f"Connecting to LiveKit room at {LIVEKIT_URL}...")
        await ctx.connect()
        logger.info("Successfully connected to room")
    except Exception as e:
        logger.error(f"Failed to connect to room: {e}")
        return

    # Initialize session with more resilient timeout settings
    try:
        # 1. Instantiate your WhisperSTT with forced English language
        whisper_stt = WhisperSTT(model="base", language="en")

        # 2. Wrap it with StreamAdapter
        stt_adapter = StreamAdapter(
            stt=whisper_stt,
            vad=silero.VAD.load(
                min_silence_duration=0.2,
            )
        )

        logger.info(f"Initializing LLM with Ollama at {OLLAMA_BASE_URL}")
        logger.info(f"Initializing TTS with Kokoro at {KOKORO_BASE_URL}")

        session = AgentSession(
            vad=ctx.proc.userdata["vad"],
            llm=openai.LLM.with_ollama(
                # model="llama3.1:8b",
                model=OLLAMA_MODEL,
                # model="deepseek-r1:1.5b",
                base_url=OLLAMA_BASE_URL,
                # timeout=60,
            ),
            # 3. Pass the adapter to the session
            stt=stt_adapter,
            tts=openai.TTS(
                model="kokoro",
                voice="af_alloy",
                api_key="not-needed",
                base_url=KOKORO_BASE_URL,
                response_format="wav",
            ),
            turn_detection=MultilingualModel(),
        )
    except Exception as e:
        logger.error(f"Failed to initialize agent session: {e}")
        return

    # log metrics as they are emitted, and total usage after session is over
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    # shutdown callbacks are triggered when the session is over
    ctx.add_shutdown_callback(log_usage)

    # wait for a participant to join the room
    await ctx.wait_for_participant()

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # uncomment to enable Krisp BVC noise cancellation
            # noise_cancellation=noise_cancellation.BVC(),
            text_enabled=True,
            audio_enabled=True,
        ),
        room_output_options=RoomOutputOptions(
            transcription_enabled=True,
            audio_enabled=True,

        ),
    )


if __name__ == "__main__":
    # Make sure logging is configured if not done elsewhere
    logging.basicConfig(level=logging.INFO)
    # Set livekit logger level if desired (e.g., WARNING to reduce noise)
    logging.getLogger("livekit").setLevel(logging.WARNING)
    logger.setLevel(logging.INFO) # Keep your agent logger at INFO

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))