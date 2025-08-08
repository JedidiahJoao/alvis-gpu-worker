# gpu-worker/worker.py

import runpod
from faster_whisper import WhisperModel
import tempfile
import subprocess
from pathlib import Path

# --- Model Loading (Now uses the pre-downloaded model) ---
print("Loading Whisper model from cache...")
# This tells faster-whisper to look for the model in the cache location
# we saved it to in the Dockerfile.
model = WhisperModel("small", device="cuda", compute_type="float16", download_root="/root/.cache")
print("Whisper model loaded successfully.")
# -----------------------------------------------------------

def download_audio(url):
    temp_dir = Path(tempfile.gettempdir())
    temp_path = temp_dir / "downloaded_audio.mp3"
    
    print(f"Downloading audio from: {url}")
    command = [
        "yt-dlp", "-f", "bestaudio", "--extract-audio",
        "--audio-format", "mp3", "-o", str(temp_path), url
    ]
    
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"yt-dlp failed: {e.stderr.strip()}")

    print(f"Audio downloaded to: {temp_path}")
    return str(temp_path)

async def handler(job):
    job_input = job.get('input', {})
    audio_url = job_input.get('audio_url')

    if not audio_url:
        return {"error": "Missing 'audio_url' in input"}

    try:
        audio_path = download_audio(audio_url)
        
        print(f"Starting transcription for: {audio_path}")
        segments, info = model.transcribe(audio_path, beam_size=5, language="en")

        full_transcript = "".join(segment.text for segment in segments)
        
        print("Transcription complete.")
        return {"transcript": full_transcript.strip()}

    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": str(e)}

runpod.serverless.start({
    "handler": handler
})