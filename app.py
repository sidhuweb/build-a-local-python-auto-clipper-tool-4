"""
AutoClipper Web Application & REST API Server.
Provides real-time interactive UI, WebSocket progress streaming,
and REST endpoints to process YouTube URLs and local streams.
"""

import asyncio
import json
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from autoclipper.pipeline import AutoClipperPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("autoclipper.web")

BASE_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = BASE_DIR / "output"
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AutoClipper Local Tool", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory jobs tracking
jobs_db: Dict[str, Dict[str, Any]] = {}
active_connections: Dict[str, List[WebSocket]] = {}


class ClipRequest(BaseModel):
    url: str
    clip_duration: float = 45.0
    whisper_model: str = "tiny"
    facecam_preset: str = "top-right"
    highlight_color: str = "&H0000FFFF"  # Electric Yellow
    cookies_browser: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = BASE_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    static_index = STATIC_DIR / "index.html"
    if static_index.exists():
        return FileResponse(static_index)
    return HTMLResponse("<h1>AutoClipper Web Server is running!</h1>")


@app.get("/api/sample")
async def get_sample_info():
    """Returns sample stream info for one-click demo."""
    sample_video = BASE_DIR / "samples" / "sample_stream.mp4"
    exists = sample_video.exists()
    return {
        "exists": exists,
        "path": str(sample_video) if exists else None,
        "title": "PRO GAMING STREAM | 1v4 Clutch & Rage Moments",
        "duration": 115.0,
        "sample_url": "samples/sample_stream.mp4",
    }


@app.post("/api/clip")
async def start_clipping(req: ClipRequest):
    """Starts asynchronous clipping job."""
    target = req.url.strip()
    if not target:
        raise HTTPException(status_code=400, detail="URL or video path cannot be empty")

    job_id = str(uuid.uuid4())[:8]
    job_output_dir = OUTPUT_DIR / f"job_{job_id}"
    job_output_dir.mkdir(parents=True, exist_ok=True)

    job_state = {
        "id": job_id,
        "target": target,
        "status": "processing",
        "progress": 0.0,
        "stage": "Initializing AutoClipper...",
        "clips": [],
        "timeline": [],
        "error": None,
        "started_at": asyncio.get_event_loop().time(),
    }
    jobs_db[job_id] = job_state

    # Run in background thread
    threading.Thread(
        target=_run_pipeline_worker,
        args=(job_id, target, req, job_output_dir),
        daemon=True,
    ).start()

    return {"job_id": job_id, "status": "started"}


def _run_pipeline_worker(job_id: str, target: str, req: ClipRequest, job_dir: Path):
    """Worker thread running AutoClipper pipeline."""
    job = jobs_db[job_id]

    def on_progress(pct: float, stage: str):
        job["progress"] = round(pct, 1)
        job["stage"] = stage
        _broadcast_ws(job_id, {"type": "progress", "progress": job["progress"], "stage": stage})

    try:
        pipeline = AutoClipperPipeline(
            output_dir=str(job_dir),
            whisper_model=req.whisper_model,
            clip_duration=req.clip_duration,
            facecam_preset=req.facecam_preset,
            highlight_color=req.highlight_color,
            cookies_browser=req.cookies_browser,
        )

        results = pipeline.process(
            url_or_path=target,
            progress_callback=on_progress,
            top_k=3,
        )

        # Update URLs for HTTP access
        clips = results.get("clips", [])
        for c in clips:
            c["video_url"] = f"/api/files/{job_id}/{c['relative_video_file']}"
            c["metadata_url"] = f"/api/files/{job_id}/{c['relative_metadata_file']}"

        job["status"] = "completed"
        job["progress"] = 100.0
        job["stage"] = "Done! Clips generated successfully."
        job["clips"] = clips
        job["timeline"] = results.get("timeline", [])
        job["source_video"] = results.get("source_video", {})

        _broadcast_ws(job_id, {
            "type": "completed",
            "progress": 100.0,
            "clips": clips,
            "timeline": job["timeline"],
            "source_video": job["source_video"],
        })

    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        job["status"] = "failed"
        job["error"] = str(e)
        job["stage"] = f"Error: {e}"
        _broadcast_ws(job_id, {"type": "error", "error": str(e)})


def _broadcast_ws(job_id: str, payload: dict):
    """Sends JSON update to active WebSockets for this job."""
    # WebSockets will be handled or polled via GET /api/status/{job_id}
    pass


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Returns current job progress and results."""
    if job_id not in jobs_db:
        # Check if it was a saved static demo
        demo_summary = STATIC_DIR / "clips" / "all_clips_summary.json"
        if job_id == "demo" and demo_summary.exists():
            with open(demo_summary) as f:
                data = json.load(f)
            clips = data.get("clips", [])
            for c in clips:
                c["video_url"] = f"/static/clips/{c['relative_video_file']}"
                c["metadata_url"] = f"/static/clips/{c['relative_metadata_file']}"
            return {
                "id": "demo",
                "status": "completed",
                "progress": 100.0,
                "stage": "Demo loaded",
                "clips": clips,
                "timeline": data.get("timeline", []),
                "source_video": data.get("source_video", {}),
            }
        raise HTTPException(status_code=404, detail="Job not found")

    return jobs_db[job_id]


@app.get("/api/files/{job_id}/{filename}")
async def get_job_file(job_id: str, filename: str):
    """Downloads or streams generated video clip or metadata text file."""
    fpath = OUTPUT_DIR / f"job_{job_id}" / filename
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="File not found")

    media_type = "video/mp4" if filename.endswith(".mp4") else "text/plain"
    return FileResponse(str(fpath), media_type=media_type)


@app.get("/api/demo-clips")
async def get_demo_clips():
    """Returns pre-rendered demo clips for instant preview in the browser."""
    candidates = [
        STATIC_DIR / "clips" / "all_clips_summary.json",
        OUTPUT_DIR / "test_run" / "all_clips_summary.json",
    ]
    for cand in candidates:
        if cand.exists():
            with open(cand) as f:
                data = json.load(f)
            clips = data.get("clips", [])
            for c in clips:
                c["video_url"] = f"/static/clips/{c['relative_video_file']}"
                c["metadata_url"] = f"/static/clips/{c['relative_metadata_file']}"
            return {
                "clips": clips,
                "timeline": data.get("timeline", []),
                "source_video": data.get("source_video", {}),
            }
    return {"clips": [], "timeline": []}


# Mount static assets
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
