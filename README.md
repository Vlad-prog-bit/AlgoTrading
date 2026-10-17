# AlgoTrading

A Python research project for collecting Russian company financial reports and financial video content, preparing transcripts, and building a foundation for financial RAG and trading hypothesis research.

The project currently focuses on **data collection and speech transcription**. Semantic retrieval, Qwen-based generation, CatBoost training, and strategy backtesting are planned, not implemented.

## Current features

- **Company reports:** download documents from Interfax e-disclosure using Playwright and a browser session. Supported sections include annual reports, financial statements, and consolidated reports.
- **Report filtering and tracking:** select companies, reporting years, and document sections; record source URLs, reporting periods, file hashes, and download status in JSON manifests.
- **Archive extraction:** validate ZIP archives and destination paths before extraction. Successfully extracted ZIP files are deleted; archives that fail validation are retained.
- **YouTube search and audio downloads:** search by topic, download audio, and convert it to MP3 at a target bitrate of 128 kbps using yt-dlp and FFmpeg.
- **Audio metadata:** maintain `metadataes.json` with video IDs, source URLs, titles, channels, descriptions, upload dates, and durations. A download archive tracks previously downloaded audio.
- **Speech transcription:** load Whisper `medium` once per batch, use MPS when available, and save Russian transcripts as TXT and JSON with segment timestamps.

## Workflow

```text
Interfax e-disclosure → reports + manifests → archive extraction
YouTube search → MP3 + metadata → Whisper → text + timestamped segments
```

These outputs will later support a searchable financial knowledge base. Downloading a document or finding a video does not establish its accuracy or research relevance.

## Project structure

```text
AlgoTrading/
├── README.md
├── pyproject.toml
├── requirements-lock.txt
├── .gitignore
├── src/
│   ├── algotrading/
│   │   └── __init__.py       # Current pipeline entry point
│   ├── parser/
│   │   ├── report_parser.py # Interfax document collection
│   │   ├── video_parser.py  # YouTube search and MP3 downloads
│   │   └── work.ipynb       # Development experiments
│   └── preprocessing/
│       ├── transcript.py    # Whisper transcription and output saving
│       └── unzip.py         # ZIP validation and extraction
├── configs/
├── notebooks/
├── tests/
├── data/                    # Local artifacts; excluded from Git
│   ├── reports/
│   │   ├── annual_reports/<company>/
│   │   ├── financial_statements/<company>/
│   │   └── consolidated_reports/<company>/
│   ├── audio/
│   │   ├── <video_id>.mp3
│   │   ├── metadataes.json
│   │   └── downloaded_audio.txt
│   └── transcripts/
└── Models/whisper/          # Downloaded model weights; excluded from Git
```

## Setup

The current implementation targets **macOS**, with Python **3.11**, Yandex Browser, and an optional Apple Silicon MPS backend. Several paths are currently configured for `~/Desktop/AlgoTrading`.

Create and activate an environment from the project directory:

```bash
conda create --prefix ./.venv python=3.11 -y
conda activate ./.venv
python -m pip install -e .
```

Some runtime dependencies are not yet declared in `pyproject.toml`. Install them explicitly, including the YouTube JavaScript solver components:

```bash
python -m pip install playwright tqdm "yt-dlp[default]"
```

For notebook progress widgets, optionally install `ipywidgets` in the notebook environment.

### Browser and audio requirements

- The default report collector expects Yandex Browser at `/Applications/Yandex.app/Contents/MacOS/Yandex`.
- Chromium is also supported by the report collector. Install it with `python -m playwright install chromium` and pass `browser_name="chromium"` to `parse_companies`.
- YouTube extraction uses Deno. With Homebrew, install it using `brew install deno`. The audio downloader currently expects `/opt/homebrew/bin/deno`; adjust the code if your installation differs.
- `imageio-ffmpeg` supplies the FFmpeg executable used for MP3 conversion. Whisper calls `ffmpeg` by name, so it must also be available on `PATH`, normally through `.venv/bin/ffmpeg` after environment activation.
- The browser setup optionally reads `Russian_Trusted_CA.pem` from the project root through `NODE_EXTRA_CA_CERTS`. Use only certificates from a verified source. The file is not required for every source or browser configuration.

Whisper downloads its model weights on first use to `Models/whisper`. Installing the Python dependencies does not download those weights.

## Running the pipeline

From the project root with the environment activated:

```bash
PYTHONPATH=src python src/algotrading/__init__.py
```

The current entry point runs these steps sequentially:

1. Collect 2024 documents for Sber and TBank from sections 2, 3, and 4, then extract report ZIP archives.
2. Search for two YouTube videos using the configured financial reporting query and download their audio.
3. Transcribe audio referenced in `data/audio/metadataes.json`.

Change the companies, years, report types, search query, and result count in `src/algotrading/__init__.py` before a batch run. The configured company map also includes VK, Gazprom, and MegaFon; TBank's entry refers to the bank legal entity rather than the holding company.

The entry point initiates network downloads and model inference. A failure in an earlier step can prevent later steps from running.

## Current limitations

- Local paths, browser location, Deno location, and pipeline settings are configured in code rather than a shared configuration file.
- Website access and available YouTube formats depend on network conditions and upstream service changes. Browser automation does not guarantee access.
- `downloaded_audio.txt` tracks completed downloads, not whether files still exist. A manually deleted MP3 may still be skipped on a later run.
- Transcription currently processes the metadata list without a completed-transcript checkpoint. Re-running may transcribe the same audio and overwrite outputs.
- The current transcript saver appends `.mp3` to a name that already includes it. Outputs can therefore be named `<video_id>.mp3.txt` and `<video_id>.mp3.json`, and the generated source URL can contain an extra `.mp3`. This naming issue needs correction before indexing transcripts.
- `parse_companies` currently returns the result of archive extraction rather than its collected results dictionary. Download details remain in the saved manifests.
- Whisper can misrecognize figures, percentages, company names, and financial terminology. Validate samples against the original audio before using transcripts for research.

## Roadmap

- Consolidate configuration and make paths portable.
- Improve resumable processing, error reporting, and transcript naming.
- Extract text and tables from financial documents while preserving page references.
- Build semantic retrieval with BGE-M3 and FAISS.
- Add Qwen-based financial RAG with citations to documents and video timestamps.
- Formulate measurable trading hypotheses and develop CatBoost baselines.
- Evaluate strategies with time-based validation, transaction costs, and controls against look-ahead bias.

## Data and research practices

Keep source URLs, reporting periods, and actual publication dates. A reporting period is not the date when information became available to the market. Distinguish consolidated group statements from individual legal-entity statements.

Audio, reports, transcripts, model weights, browser profiles, and the local environment are excluded from Git by the current ignore rules. Keep backups separately; Git only protects tracked and committed files.

This project is under development and intended for research and learning.
