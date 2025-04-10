from setuptools import setup, find_namespace_packages

setup(
    name="livekit-plugins-offline",
    version="0.1.0",
    description="Offline plugins for LiveKit Agents",
    author="Sod-Erdene Dalaikhuu",
    author_email="dadk62@inf.elte.hu",
    packages=find_namespace_packages(include=["livekit.*"]),
    install_requires=[
        "livekit-agents>=0.12.20",
        "livekit-plugins-silero>=0.1.0",
        "openai-whisper>=20231117",
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "httpx>=0.24.0",
        # "piper-tts>=1.0.0",  # Specify exact version to avoid conflicts
        # "piper-phonemize>=1.1.0"  # Match with piper-tts 1.2.0 requirement
        # "piper-phonemize>=1.1.0"
    ],
    python_requires=">=3.8.10",
)