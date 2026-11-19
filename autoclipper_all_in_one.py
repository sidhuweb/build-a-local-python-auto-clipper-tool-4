#!/usr/bin/env python3
"""
================================================================================
AutoClipper - Standalone All-In-One AI Stream & Video Auto-Clipper
================================================================================
Finds the top 3 viral moments using audio spikes and chat/replay peaks,
crops into 9:16 vertical split-screen (streamer facecam on top, gameplay on bottom),
burns bold animated captions in the middle, and generates viral hook metadata text kits.

Requirements:
    pip install yt-dlp faster-whisper numpy scipy opencv-python-headless
    FFmpeg installed and available in PATH.

Usage:
    python autoclipper_all_in_one.py "https://www.youtube.com/watch?v=VIDEO_ID"
    python autoclipper_all_in_one.py "recording.mp4" --clip-duration 45 --facecam top-right
    python autoclipper_all_in_one.py --demo
================================================================================
"""

import argparse
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("autoclipper")

# Terminal ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


# ==============================================================================
# 1. VIDEO DOWNLOADER & METADATA INGESTION
# ==============================================================================
class VideoDownloader:
    """Handles downloading YouTube videos, replay chat, heatmaps, and audio via yt-dlp."""

    def __init__(
        self,
        output_dir: str = "downloads",
        cookies_file: Optional[str] = None,
        cookies_browser: Optional[str] = None,
        player_clients: Optional[List[str]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cookies_file = cookies_file
        self.cookies_browser = cookies_browser
        self.player_clients = player_clients or ["android", "ios", "web"]

    def is_local_file(self, target: str) -> bool:
        return os.path.exists(target) or not re.match(r"^(https?://|www\.)", target.strip())

    def get_video_info(self, url_or_path: str) -> Dict[str, Any]:
        """Extracts video metadata without downloading full video file."""
        if self.is_local_file(url_or_path):
            return self._get_local_video_info(url_or_path)

        import yt_dlp

        ydl_opts = {
            "extract_flat": False,
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
        }
        if self.cookies_file and os.path.exists(self.cookies_file):
            ydl_opts["cookiefile"] = self.cookies_file
        elif self.cookies_browser:
            ydl_opts["cookiesfrombrowser"] = (self.cookies_browser,)

        if self.player_clients:
            ydl_opts["extractor_args"] = {"youtube": {"player_client": self.player_clients}}

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url_or_path, download=False)
                return self._format_yt_info(info)
        except Exception as e:
            logger.warning(f"yt-dlp extract_info warning: {e}. Retrying minimal options...")
            with yt_dlp.YoutubeDL({"skip_download": True, "quiet": True}) as ydl:
                info = ydl.extract_info(url_or_path, download=False)
                return self._format_yt_info(info)

    def _format_yt_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        title = info.get("title", "Untitled Stream")
        duration = float(info.get("duration", 0.0) or 0.0)
        channel = info.get("uploader") or info.get("channel") or "Streamer"

        # YouTube replay heatmap
        heatmap = info.get("heatmap") or info.get("heatmaps") or []
        formatted_heatmap = []
        if isinstance(heatmap, list):
            for pt in heatmap:
                if isinstance(pt, dict):
                    formatted_heatmap.append({
                        "start_time": float(pt.get("start_time", 0.0)),
                        "end_time": float(pt.get("end_time", 0.0)),
                        "value": float(pt.get("value", 0.0)),
                    })

        has_live_chat = "live_chat" in (info.get("subtitles") or {})

        return {
            "id": info.get("id", "stream"),
            "title": title,
            "channel": channel,
            "duration": duration,
            "description": info.get("description", ""),
            "thumbnail": info.get("thumbnail", ""),
            "heatmap": formatted_heatmap,
            "has_live_chat": has_live_chat,
            "url": info.get("webpage_url", ""),
            "is_local": False,
        }

    def _get_local_video_info(self, filepath: str) -> Dict[str, Any]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Local file not found: {filepath}")

        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration,size,bit_rate:format_tags=title",
            "-show_streams", "-of", "json", filepath,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)

        duration = float(data.get("format", {}).get("duration", 0.0))
        title = Path(filepath).stem

        # Check for accompanying heatmap or chat JSON
        base = Path(filepath).with_suffix("")
        heatmap_file = base.with_name(f"{base.name}_heatmap.json")
        chat_file = base.with_name(f"{base.name}_chat.json")

        heatmap = []
        if heatmap_file.exists():
            try:
                with open(heatmap_file, "r") as f:
                    heatmap = json.load(f)
            except Exception:
                pass

        return {
            "id": Path(filepath).stem,
            "title": title,
            "channel": "Local Streamer",
            "duration": duration,
            "description": "Local video file",
            "heatmap": heatmap,
            "has_live_chat": chat_file.exists(),
            "filepath": str(Path(filepath).resolve()),
            "is_local": True,
        }

    def download_audio_track(self, url_or_path: str, output_path: Optional[str] = None) -> str:
        """Extracts 16kHz mono WAV audio track for fast analysis & transcription."""
        if output_path is None:
            output_path = str(self.output_dir / "full_audio.wav")

        if self.is_local_file(url_or_path):
            cmd = [
                "ffmpeg", "-y", "-i", url_or_path,
                "-vn", "-ar", "16000", "-ac", "1", "-f", "wav", output_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            return output_path

        import yt_dlp

        temp_audio = str(self.output_dir / "temp_audio.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": temp_audio,
            "quiet": True,
            "no_warnings": True,
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        }
        if self.cookies_file and os.path.exists(self.cookies_file):
            ydl_opts["cookiefile"] = self.cookies_file
        elif self.cookies_browser:
            ydl_opts["cookiesfrombrowser"] = (self.cookies_browser,)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url_or_path])

        wav_cand = str(self.output_dir / "temp_audio.wav")
        if os.path.exists(wav_cand):
            cmd = ["ffmpeg", "-y", "-i", wav_cand, "-ar", "16000", "-ac", "1", output_path]
            subprocess.run(cmd, capture_output=True, check=True)
            if wav_cand != output_path and os.path.exists(wav_cand):
                os.remove(wav_cand)
            return output_path

        raise RuntimeError("Failed to extract audio track")

    def download_full_video(self, url_or_path: str, output_path: Optional[str] = None) -> str:
        """Downloads complete video or returns local path."""
        if self.is_local_file(url_or_path):
            return str(Path(url_or_path).resolve())

        if output_path is None:
            output_path = str(self.output_dir / "source_video.mp4")

        import yt_dlp

        ydl_opts = {
            "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]/best",
            "outtmpl": output_path,
            "quiet": False,
            "merge_output_format": "mp4",
        }
        if self.cookies_file and os.path.exists(self.cookies_file):
            ydl_opts["cookiefile"] = self.cookies_file
        elif self.cookies_browser:
            ydl_opts["cookiesfrombrowser"] = (self.cookies_browser,)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url_or_path])

        return output_path

    def download_video_segment(self, url_or_path: str, start_time: float, end_time: float, output_path: str) -> str:
        """Extracts a sub-clip segment."""
        duration = end_time - start_time
        if self.is_local_file(url_or_path):
            cmd = [
                "ffmpeg", "-y", "-ss", f"{start_time:.3f}", "-i", url_or_path,
                "-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", "veryfast",
                "-c:a", "aac", output_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            return output_path

        # If remote YouTube, download full or segment
        full_vid = self.download_full_video(url_or_path)
        cmd = [
            "ffmpeg", "-y", "-ss", f"{start_time:.3f}", "-i", full_vid,
            "-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", "veryfast",
            "-c:a", "aac", output_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path

    def download_live_chat(self, url: str) -> Optional[List[Dict[str, Any]]]:
        """Downloads YouTube live chat replay messages."""
        if self.is_local_file(url):
            base = Path(url).with_suffix("")
            chat_file = base.with_name(f"{base.name}_chat.json")
            if chat_file.exists():
                try:
                    with open(chat_file, "r") as f:
                        return json.load(f)
                except Exception:
                    return None
            return None

        import yt_dlp

        chat_out = str(self.output_dir / "chat_replay")
        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "subtitleslangs": ["live_chat"],
            "outtmpl": chat_out,
            "quiet": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            live_chat_json = f"{chat_out}.live_chat.json"
            if os.path.exists(live_chat_json):
                messages = []
                with open(live_chat_json, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            replay_action = item.get("replayChatItemAction", {})
                            for action in replay_action.get("actions", []):
                                renderer = action.get("addChatItemAction", {}).get("item", {}).get("liveChatTextMessageRenderer", {})
                                if renderer:
                                    usec = int(renderer.get("timestampUsec", 0))
                                    runs = renderer.get("message", {}).get("runs", [])
                                    text = "".join(r.get("text", "") for r in runs)
                                    author = renderer.get("authorName", {}).get("simpleText", "")
                                    messages.append({
                                        "time_offset": usec / 1_000_000.0,
                                        "text": text,
                                        "author": author,
                                    })
                        except Exception:
                            continue
                return messages
        except Exception:
            pass
        return None


# ==============================================================================
# 2. VIRAL MOMENT ANALYZER (AUDIO SPIKES + REPLAY HEATMAP + CHAT DENSITY)
# ==============================================================================
class ViralMomentAnalyzer:
    """Fuses audio spikes, replay heatmap, and live chat into top 3 viral moments."""

    def __init__(
        self,
        clip_duration: float = 45.0,
        pre_peak_lead: float = 15.0,
        min_clip_distance: float = 50.0,
    ):
        self.clip_duration = clip_duration
        self.pre_peak_lead = pre_peak_lead
        self.min_clip_distance = min_clip_distance

    def analyze(
        self,
        audio_wav_path: str,
        video_duration: float,
        heatmap_data: Optional[List[Dict[str, Any]]] = None,
        chat_messages: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        time_grid, audio_scores = self._analyze_audio(audio_wav_path, video_duration)
        replay_scores = self._analyze_heatmap(heatmap_data, time_grid)
        chat_scores = self._analyze_chat(chat_messages, time_grid, video_duration)

        has_replay = np.max(replay_scores) > 0.05
        has_chat = np.max(chat_scores) > 0.05

        if has_replay and has_chat:
            w_audio, w_replay, w_chat = 0.45, 0.40, 0.15
        elif has_replay:
            w_audio, w_replay, w_chat = 0.50, 0.50, 0.0
        elif has_chat:
            w_audio, w_replay, w_chat = 0.70, 0.0, 0.30
        else:
            w_audio, w_replay, w_chat = 1.0, 0.0, 0.0

        composite = (w_audio * audio_scores + w_replay * replay_scores + w_chat * chat_scores)
        smoothed = gaussian_filter1d(composite, sigma=2.0)
        viral_curve = (smoothed / np.max(smoothed)) * 100.0 if np.max(smoothed) > 0 else np.zeros_like(smoothed)

        step_sec = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 1.0
        adapted_min_dist = min(self.min_clip_distance, max(12.0, video_duration / float(top_k + 1.2)))
        min_distance_steps = max(1, int(adapted_min_dist / step_sec))

        peak_indices, _ = find_peaks(viral_curve, distance=min_distance_steps, prominence=4.0, height=12.0)

        if len(peak_indices) < top_k:
            peak_indices, _ = find_peaks(viral_curve, distance=max(1, min_distance_steps // 2), height=8.0)

        if len(peak_indices) < top_k:
            sorted_indices = np.argsort(viral_curve)[::-1]
            chosen = []
            for idx in sorted_indices:
                if all(abs(idx - c) >= max(1, min_distance_steps // 2) for c in chosen):
                    chosen.append(idx)
                    if len(chosen) >= top_k:
                        break
            peak_indices = np.array(chosen)

        if len(peak_indices) > 0:
            top_order = np.argsort(viral_curve[peak_indices])[::-1][:top_k]
            selected_peaks = peak_indices[top_order]
        else:
            selected_peaks = np.array([int((video_duration * r) / step_sec) for r in [0.25, 0.5, 0.75][:top_k]])

        clips = []
        for rank, peak_idx in enumerate(selected_peaks, start=1):
            peak_time = float(time_grid[peak_idx])
            start_time = max(0.0, peak_time - self.pre_peak_lead)
            end_time = min(video_duration, start_time + self.clip_duration)
            if end_time - start_time < self.clip_duration and start_time > 0:
                start_time = max(0.0, end_time - self.clip_duration)

            a_val = float(audio_scores[peak_idx])
            r_val = float(replay_scores[peak_idx])
            c_val = float(chat_scores[peak_idx])

            reasons = []
            if a_val > 0.5:
                reasons.append("Audio Spike (Laughter/Scream/Hype)")
            if r_val > 0.4:
                reasons.append("YouTube Most Replayed Peak")
            if c_val > 0.4:
                reasons.append("Chat Velocity & Hype Surge")
            if not reasons:
                reasons.append("High Engagement Window")

            clips.append({
                "rank": rank,
                "peak_time": round(peak_time, 2),
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
                "duration": round(end_time - start_time, 2),
                "viral_score": round(float(viral_curve[peak_idx]), 1),
                "primary_reason": " + ".join(reasons),
            })

        stride = max(1, len(time_grid) // 300)
        timeline = [
            {
                "time": round(float(time_grid[i]), 1),
                "viral_score": round(float(viral_curve[i]), 1),
                "audio": round(float(audio_scores[i] * 100), 1),
                "replay": round(float(replay_scores[i] * 100), 1),
            }
            for i in range(0, len(time_grid), stride)
        ]

        return {"top_clips": clips, "timeline": timeline}

    def _analyze_audio(self, audio_wav_path: str, duration: float) -> Tuple[np.ndarray, np.ndarray]:
        sr, data = wavfile.read(audio_wav_path)
        if data.ndim > 1:
            data = data.mean(axis=1)

        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        else:
            data = data.astype(np.float32)

        window_size = int(sr * 1.0)
        step_size = int(sr * 0.5)
        num_steps = max(1, int((len(data) - window_size) / step_size))
        times = np.zeros(num_steps, dtype=np.float32)
        rms_vals = np.zeros(num_steps, dtype=np.float32)

        for i in range(num_steps):
            start = i * step_size
            chunk = data[start:start + window_size]
            rms_vals[i] = np.sqrt(np.mean(chunk ** 2) + 1e-9)
            times[i] = (start + window_size / 2) / float(sr)

        db_vals = 20.0 * np.log10(rms_vals + 1e-6)
        baseline_steps = max(1, int(30.0 / 0.5))
        rolling_baseline = np.zeros_like(db_vals)
        for i in range(len(db_vals)):
            w_start = max(0, i - baseline_steps // 2)
            w_end = min(len(db_vals), i + baseline_steps // 2)
            rolling_baseline[i] = np.median(db_vals[w_start:w_end])

        delta_db = np.maximum(0.0, db_vals - rolling_baseline)
        norm_rms = (rms_vals - np.min(rms_vals)) / (np.ptp(rms_vals) + 1e-6)
        norm_delta = (delta_db - np.min(delta_db)) / (np.ptp(delta_db) + 1e-6)

        audio_score = np.clip(0.60 * norm_delta + 0.40 * norm_rms, 0.0, 1.0)
        return times, audio_score

    def _analyze_heatmap(self, heatmap: Optional[List[Dict[str, Any]]], time_grid: np.ndarray) -> np.ndarray:
        scores = np.zeros_like(time_grid)
        if not heatmap:
            return scores
        try:
            hp_times = [(pt["start_time"] + pt["end_time"]) / 2.0 for pt in heatmap]
            hp_vals = [float(pt["value"]) for pt in heatmap]
            if len(hp_times) > 1:
                scores = np.interp(time_grid, hp_times, hp_vals, left=0.0, right=0.0)
                if np.max(scores) > 0:
                    scores = scores / np.max(scores)
        except Exception:
            pass
        return np.clip(scores, 0.0, 1.0)

    def _analyze_chat(self, chat_messages: Optional[List[Dict[str, Any]]], time_grid: np.ndarray, duration: float) -> np.ndarray:
        scores = np.zeros_like(time_grid)
        if not chat_messages:
            return scores

        hype_terms = {"w", "l", "omg", "pog", "poggers", "kekw", "lol", "lmao", "wtf", "clutch", "clip"}
        bucket_size = 10.0
        num_buckets = max(1, int(duration / bucket_size) + 1)
        velocity = np.zeros(num_buckets, dtype=np.float32)

        for msg in chat_messages:
            t = float(msg.get("time_offset", 0.0))
            if 0 <= t < duration:
                b = int(t / bucket_size)
                if b < num_buckets:
                    words = set(str(msg.get("text", "")).lower().split())
                    matched = len(words.intersection(hype_terms))
                    velocity[b] += 1.0 + matched * 0.8

        bucket_times = np.arange(num_buckets) * bucket_size + bucket_size / 2.0
        if len(bucket_times) > 1 and np.max(velocity) > 0:
            scores = np.interp(time_grid, bucket_times, velocity, left=0.0, right=0.0)
            scores = scores / np.max(scores)

        return np.clip(scores, 0.0, 1.0)


# ==============================================================================
# 3. FACECAM TRACKER & WEBCAM CROPPING
# ==============================================================================
class FaceCamDetector:
    """Auto-detects streamer webcam coordinates or applies corner presets."""

    PRESETS = {
        "top-right": (0.65, 0.02, 0.33, 0.40),
        "top-left": (0.02, 0.02, 0.33, 0.40),
        "bottom-right": (0.65, 0.58, 0.33, 0.40),
        "bottom-left": (0.02, 0.58, 0.33, 0.40),
        "top-center": (0.25, 0.0, 0.50, 0.48),
    }

    def __init__(self, cascade_path: Optional[str] = None):
        if cascade_path and os.path.exists(cascade_path):
            self.cascade_path = cascade_path
        else:
            default_xml = Path(__file__).parent / "assets" / "models" / "haarcascade_frontalface_default.xml"
            self.cascade_path = str(default_xml) if default_xml.exists() else cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        self.classifier = None
        if os.path.exists(self.cascade_path):
            try:
                self.classifier = cv2.CascadeClassifier(self.cascade_path)
            except Exception:
                pass

    def detect_facecam(self, video_path: str, preset_fallback: str = "top-right") -> Dict[str, int]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return self.get_preset_crop(1920, 1080, preset_fallback)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        faces = []
        if self.classifier is not None and not self.classifier.empty():
            for i in range(12):
                idx = int((i * total_frames) / 12)
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                detected = self.classifier.detectMultiScale(gray, 1.15, 4, minSize=(int(h * 0.08), int(h * 0.08)))
                for box in detected:
                    faces.append(box)

        cap.release()

        if faces:
            arr = np.array(faces)
            cx = np.median(arr[:, 0] + arr[:, 2] / 2.0)
            cy = np.median(arr[:, 1] + arr[:, 3] / 2.0)
            box_h = np.median(arr[:, 3]) * 2.3
            box_w = box_h * 1.125  # 9:8 ratio
            x1 = max(0, int(cx - box_w / 2.0))
            y1 = max(0, int(cy - box_h / 2.0))
            crop_w = min(w - x1, int(box_w))
            crop_h = min(h - y1, int(box_h))
            return {"x": x1 - (x1 % 2), "y": y1 - (y1 % 2), "w": crop_w - (crop_w % 2), "h": crop_h - (crop_h % 2)}

        return self.get_preset_crop(w, h, preset_fallback)

    def get_preset_crop(self, frame_w: int, frame_h: int, preset: str = "top-right") -> Dict[str, int]:
        fx, fy, fw, fh = self.PRESETS.get(preset, self.PRESETS["top-right"])
        x = int(frame_w * fx)
        y = int(frame_h * fy)
        w = int(frame_w * fw)
        h = int(frame_h * fh)
        return {"x": x - (x % 2), "y": y - (y % 2), "w": w - (w % 2), "h": h - (h % 2)}


# ==============================================================================
# 4. 9:16 VERTICAL SPLIT-SCREEN CROPPER (FFMPEG)
# ==============================================================================
class VerticalSplitScreenCropper:
    """Renders 1080x1920 9:16 split-screen with middle border and burned captions."""

    def __init__(self, output_width: int = 1080, output_height: int = 1920, fonts_dir: Optional[str] = None):
        self.output_width = output_width
        self.output_height = output_height
        self.split_height = output_height // 2
        default_font_dir = Path(__file__).parent / "assets" / "fonts"
        self.fonts_dir = str(default_font_dir.resolve()) if default_font_dir.exists() else fonts_dir

    def render_split_screen(
        self,
        input_video_path: str,
        output_video_path: str,
        facecam_crop: Dict[str, int],
        ass_subtitles_path: Optional[str] = None,
    ) -> str:
        fc_x, fc_y = facecam_crop["x"], facecam_crop["y"]
        fc_w, fc_h = facecam_crop["w"], facecam_crop["h"]

        top_filter = f"[0:v]crop={fc_w}:{fc_h}:{fc_x}:{fc_y},scale={self.output_width}:{self.split_height}:force_original_aspect_ratio=increase,crop={self.output_width}:{self.split_height}[top]"
        bot_filter = f"[0:v]crop=iw*0.75:ih*0.75:(iw-out_w)/2:(ih-out_h)/2,scale={self.output_width}:{self.split_height}:force_original_aspect_ratio=increase,crop={self.output_width}:{self.split_height}[bot]"
        stack_filter = "[top][bot]vstack[vsplit]"
        effects_filter = f"[vsplit]drawbox=y={self.split_height - 4}:h=8:color=0x00F0FF@0.9:t=fill"

        if ass_subtitles_path and os.path.exists(ass_subtitles_path):
            ass_clean = str(Path(ass_subtitles_path).resolve()).replace("\\", "/").replace(":", "\\:")
            if self.fonts_dir and os.path.exists(self.fonts_dir):
                f_clean = self.fonts_dir.replace("\\", "/").replace(":", "\\:")
                effects_filter += f",ass='{ass_clean}':fontsdir='{f_clean}'"
            else:
                effects_filter += f",ass='{ass_clean}'"

        effects_filter += "[outv]"
        filter_complex = f"{top_filter};{bot_filter};{stack_filter};{effects_filter}"

        cmd = [
            "ffmpeg", "-y", "-i", input_video_path,
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p",
            output_video_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return output_video_path


# ==============================================================================
# 5. FASTER-WHISPER TRANSCRIBER & ANIMATED CAPTION GENERATOR
# ==============================================================================
class CaptionGenerator:
    """Generates word-level timestamps and bold animated ASS subtitles in the center."""

    def __init__(
        self,
        model_size: str = "tiny",
        font_name: str = "Montserrat ExtraBold",
        highlight_color: str = "&H0000FFFF",  # Bright Yellow (BGR format in ASS)
    ):
        self.model_size = model_size
        self.font_name = font_name
        self.highlight_color = highlight_color
        self._model = None

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        return self._model

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        model = self._get_model()
        segments, info = model.transcribe(audio_path, word_timestamps=True)
        words_list = []
        transcript_parts = []

        for seg in segments:
            transcript_parts.append(seg.text.strip())
            for w in (seg.words or []):
                cleaned = w.word.strip()
                if cleaned:
                    words_list.append({
                        "word": cleaned.upper(),
                        "start": float(w.start),
                        "end": float(w.end),
                    })

        return {
            "full_transcript": " ".join(transcript_parts),
            "words": words_list,
        }

    def generate_animated_ass(self, words: List[Dict[str, Any]], out_path: str, y_pos: int = 960) -> str:
        header = f"""[Script Info]
Title: AutoClipper Captions
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ViralCap,{self.font_name},70,&H00FFFFFF,&H000000FF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,7,4,5,60,60,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        events = []
        # Group into 3 words per line
        chunks = [words[i:i + 3] for i in range(0, len(words), 3)]

        for chunk in chunks:
            c_start = chunk[0]["start"]
            c_end = chunk[-1]["end"] + 0.2

            for active_idx, target_w in enumerate(chunk):
                w_start = target_w["start"]
                w_end = chunk[active_idx + 1]["start"] if active_idx + 1 < len(chunk) else c_end

                start_str = self._fmt_time(w_start)
                end_str = self._fmt_time(w_end)

                line_parts = []
                for idx, w in enumerate(chunk):
                    txt = w["word"].replace("{", "(").replace("}", ")")
                    if idx == active_idx:
                        line_parts.append(f"{{\\c{self.highlight_color}\\fscx112\\fscy112}}{txt}{{\\r}}")
                    else:
                        line_parts.append(f"{{\\c&H00FFFFFF}}{txt}{{\\r}}")

                pos_tag = f"{{\\pos(540,{y_pos})}}"
                events.append(f"Dialogue: 0,{start_str},{end_str},ViralCap,,0,0,0,,{pos_tag}{' '.join(line_parts)}")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(header + "\n".join(events) + "\n")

        return out_path

    def _fmt_time(self, seconds: float) -> str:
        cs = int(round(seconds * 100)) % 100
        s = int(seconds) % 60
        m = (int(seconds) // 60) % 60
        h = int(seconds) // 3600
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# ==============================================================================
# 6. VIRAL HOOK METADATA GENERATOR (TITLES, DESCRIPTION, HASHTAGS)
# ==============================================================================
class ViralMetadataGenerator:
    """Generates viral titles, SEO description, and hashtags for each clip."""

    def generate_clip_metadata(
        self,
        clip_info: Dict[str, Any],
        transcript: str,
        channel: str,
        output_txt_path: str,
    ) -> Dict[str, Any]:
        rank = clip_info["rank"]
        clean_tx = transcript.strip().replace('"', "")
        words = clean_tx.split()
        short_quote = " ".join(words[:min(6, len(words))]).upper() if words else "INSANE MOMENT"
        short_quote = re.sub(r"[.,!?;:]+$", "", short_quote)

        title = f'"{short_quote}..." 💀 (HE LOST IT)'
        alt_titles = [
            "HE REALLY THOUGHT HE WAS GOING TO SURVIVE THIS... 💀",
            "THE MOMENT EVERYTHING WENT COMPLETELY WRONG 🤯 #Shorts",
            "THE CLEANEST 1-IN-A-MILLION CLUTCH YOU WILL SEE TODAY 🔥",
        ]

        hashtags = [
            "#shorts", "#fyp", "#viral", "#trending", "#gaming",
            "#streamer", "#twitch", "#twitchclips", "#insane", "#clutch"
        ]
        hashtags_str = " ".join(hashtags)

        start_fmt = f"{int(clip_info['start_time'] // 60):02d}:{int(clip_info['start_time'] % 60):02d}"
        end_fmt = f"{int(clip_info['end_time'] // 60):02d}:{int(clip_info['end_time'] % 60):02d}"

        description = (
            f"{title}\n\n"
            f"You won't believe how this ended! Watch till the very last second! 😱\n\n"
            f"🎯 Clip Timestamp: {start_fmt} - {end_fmt}\n"
            f"🎙️ Streamer: {channel}\n"
            f"💬 Highlight Quote: \"{transcript[:100]}...\"\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"👉 Drop a LIKE if you would have reacted the same way!\n"
            f"🔔 SUBSCRIBE for daily stream highlights & insane clutches!\n"
            f"💬 Comment your reaction below! 👇\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        content = f"""================================================================================
VIRAL CLIP #{rank} - PUBLISHING METADATA KIT
================================================================================

[RECOMMENDED VIRAL HOOK TITLE]
{title}

[ALTERNATIVE TITLES FOR A/B TESTING]
1. {alt_titles[0]}
2. {alt_titles[1]}
3. {alt_titles[2]}

--------------------------------------------------------------------------------
[VIDEO METRICS & SOURCE INFO]
Duration     : {clip_info['duration']}s ({clip_info['start_time']}s - {clip_info['end_time']}s)
Viral Score  : {clip_info['viral_score']} / 100
Primary Reason: {clip_info['primary_reason']}
Format       : 9:16 Vertical (1080x1920 Split-Screen with Animated Captions)

--------------------------------------------------------------------------------
[SEO OPTIMIZED DESCRIPTION]
{description}

--------------------------------------------------------------------------------
[TRENDING HASHTAGS]
{hashtags_str}

--------------------------------------------------------------------------------
[TRANSCRIPT EXCERPT]
"{transcript}"
================================================================================
"""
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(content)

        return {
            "rank": rank,
            "title": title,
            "alternative_titles": alt_titles,
            "description": description,
            "hashtags_string": hashtags_str,
            "file_path": str(Path(output_txt_path).resolve()),
        }


# ==============================================================================
# 7. END-TO-END PIPELINE ORCHESTRATOR
# ==============================================================================
class AutoClipperPipeline:
    def __init__(
        self,
        output_dir: str = "output",
        whisper_model: str = "tiny",
        clip_duration: float = 45.0,
        facecam_preset: str = "top-right",
        highlight_color: str = "&H0000FFFF",
        cookies_file: Optional[str] = None,
        cookies_browser: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.clip_duration = clip_duration
        self.facecam_preset = facecam_preset

        self.downloader = VideoDownloader(str(self.output_dir / "temp"), cookies_file, cookies_browser)
        self.analyzer = ViralMomentAnalyzer(clip_duration=clip_duration)
        self.face_tracker = FaceCamDetector()
        self.cropper = VerticalSplitScreenCropper()
        self.transcriber = CaptionGenerator(model_size=whisper_model, highlight_color=highlight_color)
        self.meta_gen = ViralMetadataGenerator()

    def process(self, url_or_path: str, progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
        def report(pct: float, msg: str):
            logger.info(f"[{pct:.0f}%] {msg}")
            if progress_callback:
                progress_callback(pct, msg)

        report(5, "Extracting video metadata & YouTube replay heatmap...")
        info = self.downloader.get_video_info(url_or_path)

        report(15, "Extracting audio track for spike analysis...")
        audio_path = self.downloader.download_audio_track(url_or_path)

        report(25, "Extracting stream live chat replay...")
        chat = self.downloader.download_live_chat(url_or_path)

        report(35, "Analyzing audio spikes, heatmap peaks & chat density...")
        analysis = self.analyzer.analyze(
            audio_path, info["duration"], info.get("heatmap"), chat, top_k=3
        )
        top_clips = analysis["top_clips"]

        report(45, "Downloading source video...")
        source_vid = self.downloader.download_full_video(url_or_path)

        report(55, f"Detecting streamer facecam (preset: {self.facecam_preset})...")
        facecam_crop = self.face_tracker.detect_facecam(source_vid, self.facecam_preset)

        clips_res = []
        base_progress = 60.0
        step_progress = 35.0 / max(1, len(top_clips))

        for idx, clip in enumerate(top_clips, start=1):
            cur_pct = base_progress + (idx - 1) * step_progress
            report(cur_pct, f"Processing Clip #{idx} ({clip['start_time']}s - {clip['end_time']}s)...")

            raw_clip = str(self.output_dir / "temp" / f"clip_{idx}_raw.mp4")
            clip_audio = str(self.output_dir / "temp" / f"clip_{idx}_audio.wav")
            clip_ass = str(self.output_dir / "temp" / f"clip_{idx}_captions.ass")
            final_mp4 = str(self.output_dir / f"clip_{idx}_9x16.mp4")
            meta_txt = str(self.output_dir / f"clip_{idx}_metadata.txt")

            # Cut segment
            self.downloader.download_video_segment(source_vid, clip["start_time"], clip["end_time"], raw_clip)
            self.downloader.download_audio_track(raw_clip, clip_audio)

            # Transcribe & captions
            report(cur_pct + step_progress * 0.3, f"Clip #{idx}: Transcribing word-level captions...")
            tx_data = self.transcriber.transcribe(clip_audio)
            self.transcriber.generate_animated_ass(tx_data["words"], clip_ass, y_pos=960)

            # Crop 9:16 split-screen
            report(cur_pct + step_progress * 0.6, f"Clip #{idx}: Rendering 9:16 split-screen MP4...")
            self.cropper.render_split_screen(raw_clip, final_mp4, facecam_crop, clip_ass)

            # Generate metadata kit
            report(cur_pct + step_progress * 0.85, f"Clip #{idx}: Generating viral hook metadata...")
            meta = self.meta_gen.generate_clip_metadata(clip, tx_data["full_transcript"], info["channel"], meta_txt)

            meta.update({
                "start_time": clip["start_time"],
                "end_time": clip["end_time"],
                "duration": clip["duration"],
                "viral_score": clip["viral_score"],
                "primary_reason": clip["primary_reason"],
                "video_file": final_mp4,
                "relative_video_file": os.path.basename(final_mp4),
                "relative_metadata_file": os.path.basename(meta_txt),
                "transcript": tx_data["full_transcript"],
            })
            clips_res.append(meta)

        summary_file = self.output_dir / "all_clips_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump({
                "source": info["title"],
                "clips": clips_res,
                "timeline": analysis["timeline"]
            }, f, indent=2)

        report(100, "Done! All 3 viral clips generated successfully.")
        return {"clips": clips_res, "summary_file": str(summary_file)}


# ==============================================================================
# 8. COMMAND LINE INTERFACE (CLI)
# ==============================================================================
def main():
    print(f"\n{CYAN}{BOLD}⚡ AutoClipper: AI Stream & Video Auto-Clipper (9:16 Vertical Shorts){RESET}\n")

    parser = argparse.ArgumentParser(description="AutoClipper: Finds viral moments, crops 9:16, burns animated captions.")
    parser.add_argument("target", nargs="?", help="YouTube video/stream URL or local MP4 path")
    parser.add_argument("--url", "-u", dest="url_flag", help="YouTube URL (alternative to positional argument)")
    parser.add_argument("--output-dir", "-o", default="output", help="Directory to save clips and metadata")
    parser.add_argument("--clip-duration", "-d", type=float, default=45.0, help="Clip duration in seconds (default: 45)")
    parser.add_argument("--whisper-model", "-m", default="tiny", choices=["tiny", "base", "small"], help="faster-whisper model")
    parser.add_argument("--facecam", default="top-right", choices=["auto", "top-right", "top-left", "bottom-right", "bottom-left", "top-center"])
    parser.add_argument("--cookies", help="Path to cookies.txt")
    parser.add_argument("--cookies-from-browser", choices=["chrome", "firefox", "brave", "edge"])
    parser.add_argument("--demo", action="store_true", help="Run on built-in realistic mock stream sample")

    args = parser.parse_args()
    target = args.target or args.url_flag

    if args.demo:
        print(f"{YELLOW}[DEMO MODE]{RESET} Running test on sample stream...")
        sample_path = Path("samples/sample_stream.mp4").resolve()
        if not sample_path.exists():
            from samples.generate_sample import generate_sample_stream
            generate_sample_stream(str(sample_path))
        target = str(sample_path)

    if not target:
        parser.print_help()
        print(f"\n{RED}Error: Please provide a YouTube URL or video file, or run with --demo.{RESET}")
        sys.exit(1)

    print(f"{BOLD}Target:{RESET} {target}")
    print(f"{BOLD}Output Directory:{RESET} {args.output_dir}")
    print(f"{BOLD}Clip Duration:{RESET} {args.clip_duration}s")
    print(f"{BOLD}FaceCam Mode:{RESET} {args.facecam}\n")

    pipeline = AutoClipperPipeline(
        output_dir=args.output_dir,
        whisper_model=args.whisper_model,
        clip_duration=args.clip_duration,
        facecam_preset=args.facecam,
        cookies_file=args.cookies,
        cookies_browser=args.cookies_from_browser,
    )

    def on_progress(pct: float, stage: str):
        bar = "█" * int(25 * (pct / 100.0)) + "░" * (25 - int(25 * (pct / 100.0)))
        sys.stdout.write(f"\r{CYAN}[{bar}] {pct:3.0f}%{RESET} | {stage:<50}")
        sys.stdout.flush()
        if pct >= 100:
            sys.stdout.write("\n")

    results = pipeline.process(target, on_progress)

    print(f"\n{GREEN}{BOLD}🎉 SUCCESS! Generated {len(results['clips'])} Viral Clips:{RESET}\n")
    for c in results["clips"]:
        print(f"{YELLOW}▶ CLIP #{c['rank']}:{RESET} {BOLD}{c['title']}{RESET}")
        print(f"  • Video:    {CYAN}file://{c['video_file']}{RESET}")
        print(f"  • Metadata: {CYAN}file://{c['file_path']}{RESET}")
        print(f"  • Tags:     {c['hashtags_string']}\n")


if __name__ == "__main__":
    main()
