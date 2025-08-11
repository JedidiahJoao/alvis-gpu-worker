import runpod
import tempfile
import subprocess
from pathlib import Path
import time
import torch
from transformers import pipeline
from datetime import timedelta
import uuid
import os

# No more imports from insanely_fast_whisper are needed directly

model = None

def get_audio_duration_seconds(file_path: str) -> float:
    """Uses ffprobe to get the duration of an audio file in seconds."""
    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"ERROR: Could not get duration for {file_path}. Error: {e}")
        return 0.0

def download_audio(url):
    temp_dir = Path(tempfile.gettempdir())
    # Use a unique name for each download to prevent conflicts
    temp_path = temp_dir / f"{uuid.uuid4()}.mp3"
    
    print(f"[DEBUG] Starting audio download for URL: {url}")
    start_time = time.time()
    
    command = [
        "yt-dlp", "-f", "bestaudio/best", "--extract-audio",
        "--audio-format", "mp3", "-o", str(temp_path), url
    ]
    
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        error_message = f"yt-dlp failed: {e.stderr.strip()} | stdout: {e.stdout.strip()}"
        raise RuntimeError(error_message)

    end_time = time.time()
    print(f"[DEBUG] Audio download FINISHED. Took {end_time - start_time:.2f} seconds.")
    return str(temp_path)

def format_transcript(chunks):
    """Formats the output from the model into a string with timestamps."""
    full_transcript = ""
    for chunk in chunks:
        start_seconds = int(chunk['timestamp'][0])
        timestamp = str(timedelta(seconds=start_seconds))
        text = chunk['text'].strip()
        full_transcript += f"[{timestamp}] {text}\n"
    return full_transcript

async def handler(job):
    global model

    print("[DEBUG] Handler started for a new job.")

    if model is None:
        print("[DEBUG] Model not loaded. Initializing transcription pipeline...")
        model_load_start = time.time()
        
        model = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            chunk_length_s=30,
            device="cuda:0",
            torch_dtype=torch.float16,
        )
        
        model_load_end = time.time()
        print(f"[DEBUG] Transcription pipeline loaded successfully. Took {model_load_end - model_load_start:.2f} seconds.")

    job_input = job.get('input', {})
    audio_url = job_input.get('audio_url')

    if not audio_url:
        return {"error": "Missing 'audio_url' in input"}

    audio_path = None
    try:
        audio_path = download_audio(audio_url)
        
        duration_seconds = get_audio_duration_seconds(audio_path)
        if duration_seconds == 0:
            return {"error": "Could not determine audio duration after download."}

        print(f"[DEBUG] Starting transcription for: {audio_path}")
        transcribe_start = time.time()
        
        # Ask the model to return timestamps
        outputs = model(audio_path, batch_size=8, return_timestamps=True)
        
        transcribe_end = time.time()
        print(f"[DEBUG] Transcription FINISHED. Took {transcribe_end - transcribe_start:.2f} seconds.")
        
        # Format the transcript with timestamps
        full_transcript = format_transcript(outputs["chunks"])
        
        print("[DEBUG] Job processing complete.")
        return {
            "transcript": full_transcript.strip(),
            "duration_seconds": duration_seconds
        }

    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        return {"error": str(e)}
    finally:
        # Clean up the downloaded audio file from the worker's temporary disk
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"[DEBUG] Cleaned up temporary audio file: {audio_path}")

runpod.serverless.start({
    "handler": handler
})
