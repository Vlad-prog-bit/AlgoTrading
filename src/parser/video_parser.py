from pathlib import Path

import imageio_ffmpeg
from tqdm.auto import tqdm
from yt_dlp import YoutubeDL
import json


def search_videos(topic, count=2):
    options = {
        "extract_flat": True,
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }


    with YoutubeDL(options) as downloader:
        result = downloader.extract_info(
            f"ytsearch{count}:{topic}",
            download=False,
        )

    videos = []

    for entry in (result or {}).get("entries", []):
        if not entry:
            continue

        videos.append({
            "id": entry["id"],
            "title": entry.get("title"),
            "url": f"https://www.youtube.com/watch?v={entry['id']}",
            "channel": entry.get("channel"),
            "duration": entry.get("duration"),
        })

    return videos

def video_parser(topic = 'финансовая отчетность МСФО анализ компаний',count = 2):
    save_dir = Path.home() / "Desktop/AlgoTrading/data/audio"
    save_dir.mkdir(parents=True, exist_ok=True)
    videos = search_videos(
        topic=topic,
        count=count,
    )


    options = {
    "outtmpl": str(save_dir / "%(id)s.%(ext)s"),
    "format": "bestaudio/best",
    "postprocessors": [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": "mp3",
        "preferredquality": "128",
    }],
    "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
    "noplaylist": True,
    "writeinfojson": False,
    "download_archive": str(save_dir / "downloaded_audio.txt"),
    "ignoreerrors": False,
    "quiet": True,
    "noprogress": True,
    "js_runtimes": {
        "deno": {"path": "/opt/homebrew/bin/deno"},
    },
}

    with YoutubeDL(options) as downloader:
        for video in tqdm(videos, desc="Обработка видео", unit="видео"):
            info = downloader.extract_info(video["url"], download=True)

            if not info:
                continue
            audio_path = save_dir / f"{info['id']}.mp3"

            if not audio_path.is_file() or audio_path.stat().st_size == 0:
                print(f"Аудиофайл не найден: {info['id']}")
                continue

            metadata = {
                "id": info.get("id"),
                "url": info.get("webpage_url"),
                "title": info.get("title"),
                "channel": info.get("channel"),
                "description": info.get("description"),
                "upload_date": info.get("upload_date"),
                "duration": info.get("duration"),
            }

            metadata_path = save_dir / "metadataes.json"

            if metadata_path.exists():
                records = json.loads(metadata_path.read_text(encoding="utf-8"))

                if isinstance(records, dict):
                    records = [records]
            else:
                records = []

            # Добавляем запись, если этого видео ещё нет.
            if not any(record.get("id") == metadata["id"] for record in records):
                records.append(metadata)

            metadata_path.write_text(
                json.dumps(records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
