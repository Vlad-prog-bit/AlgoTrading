"""Batch Russian speech transcription with MLX Whisper on Apple Silicon."""

import gc
import json
import platform
import subprocess
from pathlib import Path
import imageio_ffmpeg
import mlx.core as mx
import mlx_whisper
import numpy as np
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

def download_model_whisper(root, model_repo):
    """Return cached MLX weights; download them on first use.

    mlx_whisper keeps the loaded model in memory across transcribe calls
    for the same path. PyTorch Whisper .pt weights cannot be used here.
    """
    root = Path(root).expanduser()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("MLX-транскрибация требует macOS на Apple Silicon")
    if not mx.metal.is_available():
        raise RuntimeError("MLX Metal GPU недоступен")
    print(f"MLX Metal: {model_repo}", flush=True)
    return snapshot_download(
        repo_id=model_repo,
        cache_dir=str(root / "Models/whisper-mlx"),
        allow_patterns=["*.json", "*.safetensors", "*.npz"],
    )


def _audio_path(audio_path, root):
    path = Path(audio_path).expanduser()
    if not path.is_absolute():
        path = root / "data/audio" / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def preprocessing_of_transcript(audio_path, model, root=None):
    """Decode audio with bundled FFmpeg, then transcribe on MLX GPU."""
    root = (
        Path(root).expanduser()
        if root is not None
        else Path.home() / "Desktop/AlgoTrading"
    )
    audio_path = _audio_path(audio_path, root)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-nostdin", "-v", "error", "-threads", "0", "-i", str(audio_path),
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", "16000", "-",
    ]
    try:
        decoded = subprocess.run(command, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stderr.decode(errors="replace")) from error
    audio = np.frombuffer(decoded, dtype=np.int16).astype(np.float32) / 32768.0
    del decoded
    if not audio.size:
        raise ValueError(f"Аудиофайл пуст: {audio_path}")

    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=str(model),
        language="ru",
        task="transcribe",
        fp16=True,
        verbose=False,
    )
    save_dir = root / "data/transcripts"
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir, result


def save_trancript(audio_path, save_dir, result, root=None):
    video_id = Path(audio_path).stem
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    text_path = save_dir / f"{video_id}.txt"
    text_path.write_text(result["text"], encoding="utf-8")

    metadata_path = save_dir / "metadata.json"
    records = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else []
    )

    by_id = {record["video_id"]: record for record in records}
    by_id[video_id] = {
        "video_id": video_id,
        "source_url": f"https://www.youtube.com/watch?v={video_id}",
        "text_file": text_path.name,
    }

    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(list(by_id.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(metadata_path)

def transcript_mac(root=None, model_repo=None):
    """Process metadata audio sequentially using one cached MLX model."""
    if model_repo is None:
        model_repo = "mlx-community/whisper-medium-mlx"
    root = (
        Path(root).expanduser()
        if root is not None
        else Path.home() / "Desktop/AlgoTrading"
    )
    metadata_path = root / "data/audio/metadata.json"
    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = [records]
    video_ids = list(dict.fromkeys(str(record["id"]) for record in records))
    if not video_ids:
        print("Нет аудио для транскрибации")
        return
    model = download_model_whisper(root=root, model_repo=model_repo)
    for video_id in tqdm(video_ids, desc="Транскрибация MLX", unit="файл"):
        try:
            audio_path = _audio_path(f"{video_id}.mp3", root)
            save_dir, result = preprocessing_of_transcript(audio_path, model, root=root)
            save_trancript(audio_path, save_dir, result, root=root)
            del result
        finally:
            gc.collect()
            # Free unused buffers; mlx_whisper retains the model for the batch.
            mx.clear_cache()

