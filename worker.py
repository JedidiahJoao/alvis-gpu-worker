# gpu-worker/worker.py

import runpod
from faster_whisper import WhisperModel
import tempfile
import subprocess
from pathlib import Path

# --- Model Loading (Happens only on cold start) ---
print("Loading Whisper model...")
# This is the key change: device="cuda"
# We also use float16 for better performance on modern GPUs
model = WhisperModel("small", device="cuda", compute_type="float16")
print("Whisper model loaded successfully.")
# ----------------------------------------------------

def download_audio(url):
    """Downloads audio from a URL (e.g., YouTube) into a temporary file."""
    temp_dir = Path(tempfile.gettempdir())
    temp_path = temp_dir / f"{Path(url).stem}.mp3"
    
    print(f"Downloading audio from: {url}")
    # Using yt-dlp to download and convert to mp3
    command = [
        "yt-dlp",
        "-f", "bestaudio",
        "--extract-audio",
        "--audio-format", "mp3",
        "-o", str(temp_path),
        url
    ]
    subprocess.run(command, check=True)
    print(f"Audio downloaded to: {temp_path}")
    return str(temp_path)

async def handler(job):
    """
    This is the main function that RunPod will call.
    The 'job' object contains the input from our API request.
    """
    job_input = job.get('input', {})
    audio_url = job_input.get('audio_url')

    if not audio_url:
        return {"error": "Missing 'audio_url' in input"}

    try:
        # For now, we'll assume the input is always a URL.
        # We can add direct file uploads later.
        audio_path = download_audio(audio_url)
        
        print(f"Starting transcription for: {audio_path}")
        segments, info = model.transcribe(audio_path, beam_size=5, language="en")

        # Concatenate the transcribed segments into a single string
        full_transcript = "".join(segment.text for segment in segments)
        
        print("Transcription complete.")
        return {"transcript": full_transcript.strip()}

    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": str(e)}

# Start the RunPod serverless worker
runpod.serverless.start({
    "handler": handler
})