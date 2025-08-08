# In gpu-worker/worker.py, replace the download_audio function

def download_audio(url):
    """Downloads audio from a URL (e.g., YouTube) into a temporary file."""
    temp_dir = Path(tempfile.gettempdir())
    # Use a generic name to avoid issues with special characters in video titles
    temp_path = temp_dir / "downloaded_audio.mp3"
    
    print(f"Downloading audio from: {url}")
    command = [
        "yt-dlp", "-f", "bestaudio", "--extract-audio",
        "--audio-format", "mp3", "-o", str(temp_path), url
    ]
    
    # NEW: Better error capture
    try:
        # We add capture_output=True to get the error message if something goes wrong
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        # If the command fails, raise an exception with the clean error message
        raise RuntimeError(f"yt-dlp failed: {e.stderr.strip()}")

    print(f"Audio downloaded to: {temp_path}")
    return str(temp_path)