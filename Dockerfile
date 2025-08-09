# FINAL: Use a stable, pre-configured base image from RunPod
# This image includes Python, PyTorch, and all the correct NVIDIA drivers
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-runtime-ubuntu22.04

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
# Note: yt-dlp is already in requirements, this second install is redundant but harmless.
# We can clean it up later.

# --- THE FIX ---
# Pre-download the model files from Hugging Face without loading them into a GPU.
# This is a simple file download operation that works on any machine.
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Systran/faster-whisper-small', cache_dir='/root/.cache')"

# Copy our worker code
COPY worker.py .

# Command to start the worker
CMD ["python3", "-u", "worker.py"]