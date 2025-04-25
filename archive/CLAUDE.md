# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands
- Install: `pip install -e .`
- Run voice agent: `python offline_voice_agent.py --whisper-model base --piper-model /path/to/model.onnx --ollama-model llama3`
- Run ChatGPT example: `python chatgpt.py dev`
- Run single test: `python -m unittest path/to/test_file.py`

## Code Style
- PEP 8 for Python code style
- Type hints required for function parameters and return values
- Use async/await for asynchronous code
- Error handling with try/except blocks with specific exceptions
- Use logging instead of print statements
- Class naming: PascalCase (e.g., WhisperSTT)
- Function naming: snake_case (e.g., transcribe_audio)
- Variable naming: snake_case (e.g., audio_data)
- Document classes and functions with docstrings using Google style
- Import order: stdlib, third-party, local modules

## Common Issues
- STTCapabilities requires both 'streaming' and 'interim_results' parameters
- When using livekit-agents functions, check for API changes in the latest version