"""
Downloader module for AutoClipper.
Handles fetching video metadata, live chat replays, YouTube "most replayed" heatmap,
and video/audio stream downloading via yt-dlp.
"""

import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("autoclipper.downloader")


class VideoDownloader:
    """Handles downloading YouTube videos, streams, replay chat, and metadata."""

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
        """Checks if target is a local file path or a URL."""
        return os.path.exists(target) or not re.match(r"^(https?://|www\.)", target.strip())

    def get_video_info(self, url_or_path: str) -> Dict[str, Any]:
        """
        Extracts video metadata without necessarily downloading the full media file.
        Returns dict with title, duration, heatmap, chat replay info, etc.
        """
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
            logger.warning(f"yt-dlp extract_info failed: {e}. Trying fallback options...")
            # Retry with minimal options
            try:
                minimal_opts = {"skip_download": True, "quiet": True}
                with yt_dlp.YoutubeDL(minimal_opts) as ydl:
                    info = ydl.extract_info(url_or_path, download=False)
                    return self._format_yt_info(info)
            except Exception as e2:
                logger.error(f"Failed to fetch YouTube metadata: {e2}")
                raise RuntimeError(
                    f"Unable to extract video information from '{url_or_path}'. "
                    f"If YouTube is blocking requests with a bot-check, try passing "
                    f"--cookies-from-browser chrome or provide a local MP4 file. Error: {e2}"
                )

    def _format_yt_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Formats raw yt-dlp info dictionary into standardized AutoClipper structure."""
        title = info.get("title", "Untitled Stream")
        duration = float(info.get("duration", 0.0) or 0.0)
        channel = info.get("uploader") or info.get("channel") or "Streamer"
        video_id = info.get("id", "stream")

        # YouTube 'most replayed' heatmap data
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

        # Chapters if marked by creator
        chapters = []
        for chap in info.get("chapters") or []:
            chapters.append({
                "title": chap.get("title", ""),
                "start_time": float(chap.get("start_time", 0.0)),
                "end_time": float(chap.get("end_time", 0.0)),
            })

        has_live_chat = "live_chat" in (info.get("subtitles") or {})

        return {
            "id": video_id,
            "title": title,
            "channel": channel,
            "duration": duration,
            "description": info.get("description", ""),
            "thumbnail": info.get("thumbnail", ""),
            "heatmap": formatted_heatmap,
            "chapters": chapters,
            "has_live_chat": has_live_chat,
            "url": info.get("webpage_url", ""),
            "is_local": False,
        }

    def _get_local_video_info(self, filepath: str) -> Dict[str, Any]:
        """Extracts metadata from a local video file using ffprobe."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Local file not found: {filepath}")

        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration,size,bit_rate:format_tags=title",
            "-show_streams",
            "-of", "json",
            filepath,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)

        duration = float(data.get("format", {}).get("duration", 0.0))
        tags = data.get("format", {}).get("tags", {})
        title = tags.get("title") or Path(filepath).stem

        # Check if there is an accompanying heatmap or chat json in same folder
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

        has_chat = chat_file.exists()

        return {
            "id": Path(filepath).stem,
            "title": title,
            "channel": "Local Recording",
            "duration": duration,
            "description": "Local video recording",
            "thumbnail": "",
            "heatmap": heatmap,
            "chapters": [],
            "has_live_chat": has_chat,
            "filepath": str(Path(filepath).resolve()),
            "is_local": True,
        }

    def download_audio_track(self, url_or_path: str, output_path: Optional[str] = None) -> str:
        """
        Extracts or downloads the audio track as a 16kHz mono WAV file.
        This enables fast audio energy / spike analysis and Whisper transcription.
        """
        if output_path is None:
            output_path = str(self.output_dir / "full_audio.wav")

        if self.is_local_file(url_or_path):
            cmd = [
                "ffmpeg", "-y",
                "-i", url_or_path,
                "-vn",
                "-ar", "16000",
                "-ac", "1",
                "-f", "wav",
                output_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            return output_path

        import yt_dlp

        # Download best audio directly into wav
        temp_audio = str(self.output_dir / "temp_audio.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": temp_audio,
            "quiet": True,
            "no_warnings": True,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }],
        }
        if self.cookies_file and os.path.exists(self.cookies_file):
            ydl_opts["cookiefile"] = self.cookies_file
        elif self.cookies_browser:
            ydl_opts["cookiesfrombrowser"] = (self.cookies_browser,)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url_or_path])

        # Convert to 16kHz mono wav
        wav_cand = str(self.output_dir / "temp_audio.wav")
        if os.path.exists(wav_cand):
            cmd = [
                "ffmpeg", "-y",
                "-i", wav_cand,
                "-ar", "16000",
                "-ac", "1",
                output_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            if os.path.exists(wav_cand) and wav_cand != output_path:
                os.remove(wav_cand)
            return output_path

        raise RuntimeError("Failed to extract audio track")

    def download_full_video(
        self,
        url_or_path: str,
        output_path: Optional[str] = None,
        max_height: int = 1080,
    ) -> str:
        """Downloads full video or returns existing local file path."""
        if self.is_local_file(url_or_path):
            return str(Path(url_or_path).resolve())

        if output_path is None:
            output_path = str(self.output_dir / "source_video.mp4")

        import yt_dlp

        ydl_opts = {
            "format": f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}]/best",
            "outtmpl": output_path,
            "quiet": False,
            "no_warnings": True,
            "merge_output_format": "mp4",
        }
        if self.cookies_file and os.path.exists(self.cookies_file):
            ydl_opts["cookiefile"] = self.cookies_file
        elif self.cookies_browser:
            ydl_opts["cookiesfrombrowser"] = (self.cookies_browser,)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url_or_path])

        return output_path

    def download_video_segment(
        self,
        url_or_path: str,
        start_time: float,
        end_time: float,
        output_path: str,
    ) -> str:
        """
        Extracts or downloads a specific time segment [start_time, end_time].
        If local file, uses ffmpeg fast seek.
        If YouTube, can use yt-dlp download_ranges or fast stream download.
        """
        duration = end_time - start_time
        if self.is_local_file(url_or_path):
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start_time:.3f}",
                "-i", url_or_path,
                "-t", f"{duration:.3f}",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-c:a", "aac",
                output_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            return output_path

        # For remote YouTube URL, try yt-dlp download-sections
        import yt_dlp

        section_str = f"*{start_time:.1f}-{end_time:.1f}"
        ydl_opts = {
            "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]/best",
            "outtmpl": output_path,
            "quiet": True,
            "download_ranges": yt_dlp.utils.download_range_func(None, [(start_time, end_time)]),
            "force_keyframes_at_cuts": True,
        }
        if self.cookies_file and os.path.exists(self.cookies_file):
            ydl_opts["cookiefile"] = self.cookies_file
        elif self.cookies_browser:
            ydl_opts["cookiesfrombrowser"] = (self.cookies_browser,)

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url_or_path])
            if os.path.exists(output_path):
                return output_path
        except Exception as e:
            logger.warning(f"Direct range download failed ({e}), falling back to full download then cut...")

        # Fallback: full download or ffmpeg direct stream cut
        full_vid = self.download_full_video(url_or_path)
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_time:.3f}",
            "-i", full_vid,
            "-t", f"{duration:.3f}",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-c:a", "aac",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path

    def download_live_chat(self, url: str) -> Optional[List[Dict[str, Any]]]:
        """Downloads and parses YouTube live chat replay if available."""
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
                            actions = replay_action.get("actions", [])
                            for action in actions:
                                add_item = action.get("addChatItemAction", {}).get("item", {})
                                renderer = add_item.get("liveChatTextMessageRenderer", {})
                                if renderer:
                                    offset_usec = int(renderer.get("timestampUsec", 0))
                                    runs = renderer.get("message", {}).get("runs", [])
                                    text = "".join(r.get("text", "") for r in runs)
                                    author = renderer.get("authorName", {}).get("simpleText", "")
                                    messages.append({
                                        "time_offset": offset_usec / 1_000_000.0,
                                        "text": text,
                                        "author": author,
                                    })
                        except Exception:
                            continue
                return messages
        except Exception as e:
            logger.debug(f"Chat replay download not available: {e}")
        return None
