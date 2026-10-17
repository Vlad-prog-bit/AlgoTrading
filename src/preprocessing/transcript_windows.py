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
    json_path = root / "data/audio/metadataes.json"
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
def save_trancript(audio_path,save_dir,result,root):
    audio_path = root / f"data/audio/{audio_path}.mp3"
    (save_dir / f"{audio_path.stem}.txt").write_text(
        result["text"],
        encoding="utf-8",
    )  
    # Текст с таймкодами для будущего RAG.
    transcript = {
        "video_id": audio_path.stem,
        "source_url": f"https://www.youtube.com/watch?v={audio_path.stem}",
        "language": result["language"],
        "segments": [
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
            }
            for segment in result["segments"]
        ],
    }

    (save_dir / f"{audio_path.stem}.json").write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
def transcript_windows():
    root = Path.home() / "Desktop/AlgoTrading"
    json_path = root / "data/audio/metadataes.json"
    audio_path = root / "data/audio/KCNKpyP-DNQ.mp3"
    matadataes_df = pd.read_json(json_path)
    metadataes_df = pd.read_json(json_path)
    metadataes_df.head()
    list_of_id = list(metadataes_df['id'])
    model = download_model_whisper()
    for i in tqdm(list_of_id, desc="Транскрибация", unit="файл"):
        save_dir, result = preprocessing_of_transcript(audio_path=i+'.mp3',model=model)
        save_trancript(audio_path=i+'.mp3',save_dir=save_dir, result= result,root=root)

