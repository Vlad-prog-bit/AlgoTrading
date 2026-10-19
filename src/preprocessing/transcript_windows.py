import json
import os
from pathlib import Path
import imageio_ffmpeg
import whisper
import pandas as pd 
from tqdm.auto import tqdm
import torch
def download_model_whisper():
    root = Path.home() / "Desktop/AlgoTrading"
    model = whisper.load_model(
        "medium",
        download_root=str(root / "Models/whisper")
    )
    if torch.backends.cuda.is_available():
        model = model.to("cuda")
        print("CUDA доступен")
    else:
        print("CUDA недоступен")
    return model
    
def preprocessing_of_transcript(audio_path,model):
    root = Path.home() / "Desktop/AlgoTrading"
    json_path = root / "data/audio/metadata.json"
    matadataes_df = pd.read_json(json_path)
    # Whisper вызывает FFmpeg по имени.
    ffmpeg_dir = root / ".venv/bin"
    ffmpeg_path = ffmpeg_dir / "ffmpeg"
    audio_path = root/f'data/audio/{audio_path}'
    if not ffmpeg_path.exists():
        ffmpeg_path.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
    result = model.transcribe(
        str(audio_path),
        language="ru",
        task="transcribe",
        fp16=False
    )

    save_dir = root / "data/transcripts"
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir,result
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
def transcript_windows():
    root = Path.home() / "Desktop/AlgoTrading"
    json_path = root / "data/audio/metadataes.json"
    audio_path = root / "data/audio/KCNKpyP-DNQ.mp3"
    matadatas_df = pd.read_json(json_path)
    metadata_df = pd.read_json(json_path)
    metadata_df.head()
    list_of_id = list(metadata_df['id'])
    model = download_model_whisper()
    for i in tqdm(list_of_id, desc="Транскрибация", unit="файл"):
        save_dir, result = preprocessing_of_transcript(audio_path=i+'.mp3',model=model)
        save_trancript(audio_path=i+'.mp3',save_dir=save_dir, result= result,root=root)

