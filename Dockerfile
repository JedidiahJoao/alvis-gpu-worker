# FINAL: Use a stable, pre-configured base image from RunPod
# This image includes Python, PyTorch, and all the correct NVIDIA drivers
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV XDG_CACHE_HOME=/root/.cache

# Install system dependencies that are not in the base image
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install our smaller requirements list
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade yt-dlp

# Pre-download the Whisper model during the build
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cuda', compute_type='float16', download_root='/root/.cache')"

# Copy our worker code
COPY worker.py .

# Command to start the worker
CMD ["python3", "-u", "worker.py"]
