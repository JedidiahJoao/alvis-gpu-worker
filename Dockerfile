# gpu-worker/Dockerfile

# Start from an official NVIDIA CUDA base image
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies: Python, pip, and ffmpeg
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the worker code into the container
COPY worker.py .

# This is the command that RunPod will use to start our worker
CMD ["python3", "-u", "worker.py"]