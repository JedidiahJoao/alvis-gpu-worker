import runpod
from faster_whisper import WhisperModel
import tempfile
import subprocess
from pathlib import Path

# --- Model Holder ---
# We define a global variable to hold our model, but we don't load it yet.
model = None
# --------------------

def download_audio(url):
    """Downloads audio from a URL into a temporary file."""
    temp_dir = Path(tempfile.gettempdir())
    temp_path = temp_dir / "downloaded_audio.mp3"
    
    print(f"Downloading audio from: {url}")
    command = [
        "yt-dlp", "-f", "bestaudio", "--extract-audio",
        "--audio-format", "mp3", "-o", str(temp_path), url
    ]
    
    try:
        # We add capture_output=True to get the error message if something goes wrong
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        # If the command fails, raise an exception with the clean error message
        raise RuntimeError(f"yt-dlp failed: {e.stderr.strip()}")

    print(f"Audio downloaded to: {temp_path}")
    return str(temp_path)

async def handler(job):
    """
    This is the main function that RunPod will call.
    """
    global model

    # --- THE FIX: Lazy Model Loading ---
    # If the model isn't loaded yet, load it. This will only happen on the first job.
    if model is None:
        print("Model not loaded. Loading Whisper model from cache...")
        model = WhisperModel("small", device="cuda", compute_type="float16", download_root="/root/.cache")
        print("Whisper model loaded successfully.")
    # ------------------------------------

    job_input = job.get('input', {})
    audio_url = job_input.get('audio_url')

    if not audio_url:
        return {"error": "Missing 'audio_url' in input"}

    try:
        audio_path = download_audio(audio_url)
        
        print(f"Starting transcription for: {audio_path}")
        segments, _ = model.transcribe(audio_path, beam_size=5, language="en")

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