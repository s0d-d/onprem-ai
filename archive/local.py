import logging
import os
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    cli,
    metrics,
)
# Removing function_tool import as smaller models don't support it
# from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# This is a simplified version of the agent that should work with local components
logger = logging.getLogger("local-agent")
logger.setLevel(logging.INFO)

load_dotenv()

# Make sure we have an API key for the OpenAI plugin
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "dummy-key"


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

    # all functions annotated with @function_tool will be passed to the LLM when this
    # agent is active
    # Removing function tools as smaller models don't support them


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
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:11434/api/tags", timeout=5) as response:
                if response.status != 200:
                    logger.error(f"Ollama server not responding properly: {response.status}")
                    return
                logger.info("Ollama server is available")
    except Exception as e:
        logger.error(f"Failed to connect to Ollama server: {e}")
        logger.error("Make sure Ollama is running with: ollama serve")
        return
    
    # Connect to the room first - this is critical
    try:
        logger.info("Connecting to LiveKit room...")
        await ctx.connect()
        logger.info("Successfully connected to room")
    except Exception as e:
        logger.error(f"Failed to connect to room: {e}")
        return

    # Initialize session with a text-only approach for simplicity
    try:
        session = AgentSession(
            vad=ctx.proc.userdata["vad"],
            llm=openai.LLM.with_ollama(
                model="llama3", 
                base_url="http://localhost:11434/v1",
                timeout=60,  # Increase timeout to 60 seconds
                tools=None,  # Disable tools support
            ),
            # Use Kokoro TTS through OpenAI's TTS interface
            tts=openai.TTS(
                model="kokoro",
                voice="af_alloy",
                api_key="not-needed",
                base_url="http://localhost:8880/v1",
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
            # Set text input enabled, audio input disabled for reliability
            text_enabled=True,
            audio_enabled=False
        ),
        room_output_options=RoomOutputOptions(
            # Enable transcription (text) but disable audio for troubleshooting
            transcription_enabled=True, 
            audio_enabled=False
        ),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))