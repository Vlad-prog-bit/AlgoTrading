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



def _audio_path(audio_path, root):
    return audio_path


def preprocessing_of_transcript(audio_path,
                                model_path, 
                                root=None, 
                                save_path = None):
    """Decode audio with bundled FFmpeg, then transcribe on MLX GPU."""

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-nostdin", "-v", "error",
        "-threads", "0",
        "-i", str(audio_path),
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-",
    ]

    try:
        decoded = subprocess.run(
            command,
            capture_output=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            error.stderr.decode(errors="replace")
        ) from error

    audio = np.frombuffer(decoded, dtype=np.int16).astype(np.float32) / 32768.0
    del decoded
    if not audio.size:
        raise ValueError(f"Аудиофайл пуст: {audio_path}")

    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=str(model_path),
        language="ru",
        task="transcribe",
        fp16=True,
        verbose=False,
    )
    save_path.mkdir(parents=True, exist_ok=True)
    return save_path, result


def save_trancript(audio_path, 
                   save_path,
                   result, 
                   root=None, 
                   topic=None):
    video_id = Path(audio_path).stem
    save_path.mkdir(parents=True, exist_ok=True)

    text_path = save_path / f"{video_id}.txt"
    text_path.write_text(result["text"], encoding="utf-8")

    metadata_path = save_path / "metadata.json"
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
        "topic": topic,
    }

    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(list(by_id.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(metadata_path)

def transcript_mac(root=None, 
                   model_path=None,
                   save_path = None,
                   audio_path = None,
                   metadata_path = None):
    """Process metadata audio sequentially using one cached MLX model."""
    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = [records]
    source_by_id = {str(record["id"]): record for record in records}
    video_ids = list(source_by_id)
    if not video_ids:
        print("Нет аудио для транскрибации")
        return
    for video_id in tqdm(video_ids, desc="Транскрибация MLX", unit="файл"):
        try:
            audio_file = Path(audio_path) / f"{video_id}.mp3"
            save_path, result = preprocessing_of_transcript(
                audio_path=audio_file,
                model_path=model_path,
                root=root,
                save_path=save_path,
            )
            save_trancript(
                audio_file,
                save_path,
                result,
                root=root,
                topic=source_by_id[video_id].get("topic"),
            )
            del result
        finally:
            gc.collect()
            # Free unused buffers; mlx_whisper retains the model for the batch.
            mx.clear_cache()

