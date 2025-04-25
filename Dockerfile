FROM python:3.12-slim

WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY local.py .
COPY whisper_stt.py .
COPY .env .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "local.py", "dev"]