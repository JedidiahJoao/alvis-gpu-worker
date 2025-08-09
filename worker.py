import runpod
import tempfile
import subprocess
from pathlib import Path
import time
import torch
from transformers import pipeline # CORRECT IMPORT
from insanely_fast_whisper import InsanelyFastWhisperPipeline # CORRECT IMPORT

model = None

def download_audio(url):
    temp_dir = Path(tempfile.gettempdir())
    temp_path = temp_dir / "downloaded_audio.mp3"
    
    print(f"[DEBUG] Starting audio download for URL: {url}")
    start_time = time.time()
    
    command = [
        "yt-dlp", "-f", "bestaudio", "--extract-audio",
        "--audio-format", "mp3", "-o", str(temp_path), url
    ]
    
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"yt-dlp failed: {e.stderr.strip()}")

    end_time = time.time()
    print(f"[DEBUG] Audio download FINISHED. Took {end_time - start_time:.2f} seconds.")
    return str(temp_path)

async def handler(job):
    global model

    print("[DEBUG] Handler started for a new job.")

    if model is None:
        print("[DEBUG] Model not loaded. Initializing transcription pipeline...")
        model_load_start = time.time()
        
        # CORRECT INITIALIZATION
        model = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            chunk_length_s=30,
            device="cuda:0",
            torch_dtype=torch.float16,
            pipeline_class=InsanelyFastWhisperPipeline,
        )
        
        model_load_end = time.time()
        print(f"[DEBUG] Transcription pipeline loaded successfully. Took {model_load_end - model_load_start:.2f} seconds.")

    job_input = job.get('input', {})
    audio_url = job_input.get('audio_url')

    if not audio_url:
        return {"error": "Missing 'audio_url' in input"}

    try:
        audio_path = download_audio(audio_url)
        
        print(f"[DEBUG] Starting transcription for: {audio_path}")
        transcribe_start = time.time()
        
        outputs = model(audio_path, batch_size=8)
        full_transcript = outputs["text"]
        
        transcribe_end = time.time()
        print(f"[DEBUG] Transcription FINISHED. Took {transcribe_end - transcribe_end:.2f} seconds.")
        
        print("[DEBUG] Job processing complete.")
        return {"transcript": full_transcript.strip()}

    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        return {"error": str(e)}

runpod.serverless.start({
    "handler": handler
})