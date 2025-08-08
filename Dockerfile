# gpu-worker/Dockerfile

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# --- NEW: Force update yt-dlp to the latest version ---
RUN python3 -m pip install --upgrade yt-dlp
# --- END of NEW section ---

COPY worker.py .

CMD ["python3", "-u", "worker.py"]