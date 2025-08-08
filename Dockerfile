# UPDATED: Use a newer CUDA base image to match PyTorch version requirements
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
# Set a specific location for model files so they are cached
ENV XDG_CACHE_HOME=/root/.cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Use python3.10 explicitly to match the system install
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt
RUN python3.10 -m pip install --upgrade yt-dlp

# Pre-download the Whisper model during the build
RUN python3.10 -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cuda', compute_type='float16', download_root='/root/.cache')"

COPY worker.py .

CMD ["python3.10", "-u", "worker.py"]