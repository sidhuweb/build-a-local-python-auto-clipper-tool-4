# ⚡ AutoClipper

> **AI-Powered Local Stream & Video Auto-Clipper**  
> Automatically detects the **top 3 viral moments** from YouTube streams/videos using **audio spikes** and **chat/replay peaks**, crops them into **9:16 vertical split-screen** (streamer facecam on top, gameplay on bottom), burns **bold animated captions in the middle**, and generates a publishing kit with **viral hook titles, SEO descriptions, and hashtags**.

---

## 🚀 Key Features

- 🔍 **Multi-Signal Viral Moment Detection**:
  - **Audio Energy & Decibel Spikes**: Detects loud laughter, screams, hype clutches, and reaction peaks using rolling RMS and $\Delta dB$ analysis.
  - **YouTube "Most Replayed" Heatmap**: Natively extracts viewer rewind and replay peaks from YouTube's engagement curve.
  - **Live Chat Velocity & Hype Spikes**: Analyzes chat messages per second, caps-lock intensity, and hype keywords (`POG`, `KEKW`, `OMG`, `CLIP THAT`, `W`).
  - **Peak Clustering**: Uses non-maximum suppression to guarantee 3 non-overlapping, high-retention clips.
- 📱 **9:16 Vertical Split-Screen Crop (1080x1920)**:
  - **Top Half**: Streamer Facecam auto-tracked via OpenCV Haar Cascades (or preset corner webcam crop: top-right, top-left, bottom-right, bottom-left).
  - **Bottom Half**: Gameplay action center crop.
  - **Middle Divider**: Sleek neon cyber accent divider line.
- 💬 **Bold Animated Captions (TikTok / Shorts / Reels Style)**:
  - Word-level timestamps extracted using **`faster-whisper`**.
  - Positioned right in the middle divider zone for maximum viewer focus.
  - Bold typography (Montserrat ExtraBold / Anton) with heavy dark outline and shadow.
  - Dynamic active word pop animation and bright yellow/cyan highlight.
- 📝 **Viral Publishing Kit**:
  - Automatically generates `clip_1_metadata.txt`, `clip_2_metadata.txt`, and `clip_3_metadata.txt`.
  - High-CTR curiosity-gap viral hook titles + alternative titles for A/B testing.
  - SEO-optimized video description with timecodes, quotes, and engagement CTAs.
  - Copy-paste ready trending hashtags.
- 🌐 **Interactive Web Studio & Python CLI**:
  - Fast, responsive dark-mode Web GUI with live SVG waveform charts and 9:16 video players.
  - Full-featured command-line tool with progress bars and rich options.

---

## 🛠️ Architecture

```
YouTube URL / Stream / Local Video
               │
               ▼
       [VideoDownloader]
  ├── yt-dlp metadata & replay heatmap
  ├── Live chat replay JSON extraction
  └── 16kHz audio track extraction
               │
               ▼
     [ViralMomentAnalyzer]
  ├── Audio RMS & Decibel Spike Analysis
  ├── YouTube Replay Heatmap Interpolation
  ├── Chat Velocity & Hype Scoring
  └── Peak Clustering -> Top 3 Moments
               │
               ▼
      [FaceCamDetector]
  └── OpenCV Haar Cascade Streamer Tracking
               │
       ┌───────┴───────┐
       ▼               ▼
[CaptionGenerator]   [VerticalSplitScreenCropper]
  ├── faster-whisper   ├── Top: FaceCam (1080x960)
  ├── Word Timestamps  ├── Bottom: Gameplay (1080x960)
  └── Bold ASS Caps    └── Middle Neon Divider
       └───────┬───────┘
               ▼
    [ViralMetadataGenerator]
  ├── High-CTR Hook Titles
  ├── SEO Descriptions
  └── Trending Hashtags
               │
               ▼
  Top 3 9:16 Viral Videos (.mp4) + Metadata Kits (.txt)
```

---

## 📦 Installation

Ensure **Python 3.10+** and **FFmpeg** are installed on your system.

```bash
# Clone or navigate to the directory
cd workspace

# Install dependencies using uv or pip
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

## 💻 CLI Usage

### 1. Clip a YouTube Video / Stream
```bash
python autoclipper.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
```

### 2. Custom Clip Duration and FaceCam Placement
```bash
python autoclipper.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" \
  --clip-duration 45 \
  --facecam top-right \
  --output-dir my_viral_clips
```

### 3. Using Browser Cookies (for Private / Age-Gated / Protected Streams)
```bash
python autoclipper.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" \
  --cookies-from-browser chrome
```

### 4. Process a Local Video File (e.g. OBS Studio Recording)
```bash
python autoclipper.py "/path/to/stream_recording.mp4" --clip-duration 30
```

### 5. Run the Built-In Offline Demo
```bash
python autoclipper.py --demo
```

---

## 🌐 Web Studio GUI

Start the web studio locally:

```bash
./run_clipper.sh --web
# Or:
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser to `http://localhost:8000`.

- Paste any YouTube stream URL or pick the sample video.
- Watch real-time progress as each stage executes.
- View the interactive SVG timeline with audio spikes and heatmap peaks.
- Play the 9:16 vertical split-screen clips directly in the browser.
- Click "Copy" on descriptions or hashtags for one-click publishing to YouTube Shorts, TikTok, or Instagram Reels.

---

## 📄 Output Files Structure

Each clipping run outputs the following structured files into your output directory:

```
output/
  ├── clip_1_9x16.mp4          # 1080x1920 vertical split-screen with burned captions
  ├── clip_1_metadata.txt      # Viral hook title, description, and hashtags
  ├── clip_2_9x16.mp4          # Moment #2
  ├── clip_2_metadata.txt      # Moment #2 metadata
  ├── clip_3_9x16.mp4          # Moment #3
  ├── clip_3_metadata.txt      # Moment #3 metadata
  └── all_clips_summary.json   # Machine-readable metrics & timeline data
```

---

## ⚙️ Configuration & Options

| Option | Flag | Default | Description |
|---|---|---|---|
| Target | positional / `-u` | required | YouTube URL or local MP4 path |
| Output Directory | `-o`, `--output-dir` | `output` | Folder to save clips and metadata |
| Clip Duration | `-d`, `--clip-duration` | `45.0` | Target duration of clips (seconds) |
| Whisper Model | `-m`, `--whisper-model` | `tiny` | `tiny`, `base`, `small`, `medium` |
| FaceCam Preset | `--facecam` | `top-right` | `top-right`, `top-left`, `bottom-right`, `bottom-left`, `auto` |
| Cookies Browser | `--cookies-from-browser` | None | `chrome`, `firefox`, `brave`, `edge` |
| Cookies File | `--cookies` | None | Path to Netscape `cookies.txt` |
| Demo Mode | `--demo` | False | Run test on realistic built-in stream |

---

## 🛡️ License

MIT License. Designed for content creators, stream editors, and viral social media managers.
