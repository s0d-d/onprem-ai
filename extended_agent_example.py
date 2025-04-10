import asyncio
import os
import argparse
import json
import datetime
import sqlite3
from pathlib import Path
from livekit.agents import (
    VoiceAgent,
    VoiceAgentOptions,
    VoiceAgentAnalytics,
    ToolDefinition
)
from livekit.agents.llm.llm import LLMTools
from livekit.plugins import silero

# Import our custom plugins
from livekit.plugins.whisper import WhisperSTT
from livekit.plugins.piper import PiperTTS
from livekit.plugins.ollama import OllamaLLM

# SQLite database for storing notes and reminders
DB_PATH = "offline_agent.db"

def setup_database():
    """Setup SQLite database for offline functionality"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create notes table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        content TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Create reminders table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reminders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        datetime TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()
    print("Database setup complete.")

# Define custom tools

# Note-taking tool
note_tool = ToolDefinition(
    name="take_note",
    description="Save a note to local storage",
    parameters={
        "title": {
            "type": "string",
            "description": "Title of the note"
        },
        "content": {
            "type": "string",
            "description": "Content of the note"
        }
    }
)

# Reminder tool
reminder_tool = ToolDefinition(
    name="set_reminder",
    description="Set a reminder for a specific date and time",
    parameters={
        "title": {
            "type": "string",
            "description": "Title of the reminder"
        },
        "datetime": {
            "type": "string",
            "description": "Date and time for the reminder (format: YYYY-MM-DD HH:MM)"
        }
    }
)

# List notes tool
list_notes_tool = ToolDefinition(
    name="list_notes",
    description="List all saved notes",
    parameters={}
)

# List reminders tool
list_reminders_tool = ToolDefinition(
    name="list_reminders",
    description="List all saved reminders",
    parameters={}
)

# Define tool handlers
async def handle_tools(name, parameters):
    """Handle tool calls from the LLM"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        if name == "take_note":
            title = parameters.get("title", "Untitled Note")
            content = parameters.get("content", "")

            cursor.execute(
                "INSERT INTO notes (title, content) VALUES (?, ?)",
                (title, content)
            )
            conn.commit()
            return f"Note '{title}' saved successfully."

        elif name == "set_reminder":
            title = parameters.get("title", "Untitled Reminder")
            datetime_str = parameters.get("datetime", "")

            cursor.execute(
                "INSERT INTO reminders (title, datetime) VALUES (?, ?)",
                (title, datetime_str)
            )
            conn.commit()
            return f"Reminder '{title}' set for {datetime_str}."

        elif name == "list_notes":
            cursor.execute("SELECT title, created_at FROM notes ORDER BY created_at DESC LIMIT 10")
            notes = cursor.fetchall()

            if not notes:
                return "You don't have any notes saved."

            result = "Here are your most recent notes:\n"
            for i, (title, created_at) in enumerate(notes, 1):
                result += f"{i}. {title} (created on {created_at})\n"

            return result

        elif name == "list_reminders":
            cursor.execute(
                "SELECT title, datetime FROM reminders WHERE datetime >= date('now') ORDER BY datetime ASC LIMIT 10"
            )
            reminders = cursor.fetchall()

            if not reminders:
                return "You don't have any upcoming reminders."

            result = "Here are your upcoming reminders:\n"
            for i, (title, reminder_time) in enumerate(reminders, 1):
                result += f"{i}. {title} (scheduled for {reminder_time})\n"

            return result

        else:
            return f"Unknown tool: {name}"

    finally:
        conn.close()

async def create_extended_agent():
    parser = argparse.ArgumentParser(description='Run an extended offline voice agent')
    parser.add_argument('--whisper-model', type=str, default='base', help='Whisper model size')
    parser.add_argument('--piper-model', type=str, required=True, help='Path to Piper TTS model')
    parser.add_argument('--piper-config', type=str, default=None, help='Path to Piper TTS config')
    parser.add_argument('--ollama-model', type=str, default='llama3', help='Ollama model name')
    parser.add_argument('--ollama-host', type=str, default='http://localhost:11434', help='Ollama API host')
    parser.add_argument('--mic-device', type=int, default=None, help='Microphone device ID')
    parser.add_argument('--speaker-device', type=int, default=None, help='Speaker device ID')
    args = parser.parse_args()

    # Setup database for offline tools
    setup_database()

    print("Initializing extended offline voice agent...")

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

    # Create tools
    tools = LLMTools(
        available_tools=[
            note_tool,
            reminder_tool,
            list_notes_tool,
            list_reminders_tool
        ],
        tool_handler=handle_tools
    )

    # Create voice agent options
    options = VoiceAgentOptions(
        initial_prompt=(
            "You are a helpful voice assistant running completely offline on the user's device. "
            "You can take notes, set reminders, and retrieve information that has been saved locally. "
            "Keep your responses concise and conversational. "
            "When appropriate, use your tools to help the user manage their information locally."
        ),
        use_tools=True,
        debug=True
    )

    # Create analytics object
    analytics = VoiceAgentAnalytics()

    # Create voice agent with tools
    print("Creating extended voice agent with offline tools...")
    agent = VoiceAgent(
        stt=whisper_stt,
        tts=piper_tts,
        llm=ollama_llm,
        vad=silero_vad,
        options=options,
        analytics=analytics,
        llm_tools=tools  # Pass the tools here
    )

    # Start the voice agent
    print("Starting extended voice agent...")
    await agent.start(
        microphone_device=args.mic_device,
        speaker_device=args.speaker_device
    )

    return agent

async def main():
    agent = await create_extended_agent()

    # Run the agent until interrupted
    try:
        print("Extended voice agent is running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping voice agent...")
    finally:
        await agent.stop()
        print("Voice agent stopped.")

if __name__ == "__main__":
    asyncio.run(main())