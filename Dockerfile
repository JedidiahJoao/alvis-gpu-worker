# gpu-worker/Dockerfile

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
# --- NEW: Set a specific location for model files ---
ENV XDG_CACHE_HOME=/root/.cache

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN python3 -m pip install --upgrade yt-dlp

# --- NEW: Pre-download the Whisper model during the build ---
# This moves the slow download from runtime to build time.
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cuda', compute_type='float16', download_root='/root/.cache')"

COPY worker.py .

CMD ["python3", "-u", "worker.py"]