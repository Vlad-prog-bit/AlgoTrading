from yt_dlp import YoutubeDL

url = input("Ссылка на видео: ").strip()

options = {
    "outtmpl": "data/videos/%(id)s.%(ext)s",
    "format": (
        "bv[vcodec^=avc1]+ba[acodec^=mp4a]"
        "/b[vcodec^=avc1][acodec^=mp4a]"
    ),
    "merge_output_format": "mp4",
    "noplaylist": True,
}

with YoutubeDL(options) as downloader:
    downloader.download([url])