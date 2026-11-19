"""
Pipeline module for AutoClipper.
Orchestrates the complete auto-clipping workflow from YouTube URL / local video
to the top 3 9:16 vertical split-screen clips with bold animated captions and metadata kits.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from autoclipper.analyzer import ViralMomentAnalyzer
from autoclipper.cropper import VerticalSplitScreenCropper
from autoclipper.downloader import VideoDownloader
from autoclipper.face_tracker import FaceCamDetector
from autoclipper.metadata_generator import ViralMetadataGenerator
from autoclipper.transcriber import CaptionGenerator

logger = logging.getLogger("autoclipper.pipeline")


class AutoClipperPipeline:
    """End-to-end AutoClipper orchestrator."""

    def __init__(
        self,
        output_dir: str = "output",
        whisper_model: str = "tiny",
        clip_duration: float = 45.0,
        facecam_preset: str = "top-right",
        font_name: str = "Montserrat ExtraBold",
        highlight_color: str = "&H0000FFFF",  # Bright Yellow in ASS
        cookies_file: Optional[str] = None,
        cookies_browser: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.clip_duration = clip_duration
        self.facecam_preset = facecam_preset

        # Initialize sub-modules
        self.downloader = VideoDownloader(
            output_dir=str(self.output_dir / "temp"),
            cookies_file=cookies_file,
            cookies_browser=cookies_browser,
        )
        self.analyzer = ViralMomentAnalyzer(clip_duration=clip_duration)
        self.face_tracker = FaceCamDetector()
        self.cropper = VerticalSplitScreenCropper()
        self.transcriber = CaptionGenerator(
            model_size=whisper_model,
            font_name=font_name,
            highlight_color=highlight_color,
        )
        self.meta_gen = ViralMetadataGenerator()

    def process(
        self,
        url_or_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Executes full auto-clipper pipeline.
        progress_callback(percent, stage_description) can be supplied for UI feedback.
        """
        start_time_bench = time.time()

        def report(pct: float, msg: str):
            logger.info(f"[{pct:.0f}%] {msg}")
            if progress_callback:
                progress_callback(pct, msg)

        report(5, "Extracting video metadata & YouTube replay data...")
        info = self.downloader.get_video_info(url_or_path)
        video_duration = info.get("duration", 0.0)
        video_title = info.get("title", "Video")
        channel = info.get("channel", "Streamer")
        heatmap = info.get("heatmap", [])

        report(15, "Extracting audio track for spike analysis...")
        audio_path = self.downloader.download_audio_track(url_or_path)

        # Check if chat replay is available
        report(25, "Checking for stream live chat replay...")
        chat_messages = self.downloader.download_live_chat(url_or_path)

        report(35, "Analyzing audio spikes, replay heatmap, and chat density...")
        analysis_result = self.analyzer.analyze(
            audio_wav_path=audio_path,
            video_duration=video_duration,
            heatmap_data=heatmap,
            chat_messages=chat_messages,
            top_k=top_k,
        )
        top_clips = analysis_result["top_clips"]
        logger.info(f"Identified {len(top_clips)} viral moments!")

        # Prepare source video
        report(45, "Downloading / preparing source video...")
        source_video = self.downloader.download_full_video(url_or_path)

        # Auto-detect facecam
        report(55, f"Detecting streamer facecam (preset: {self.facecam_preset})...")
        facecam_crop = self.face_tracker.detect_facecam(
            video_path=source_video,
            preset_fallback=self.facecam_preset,
        )

        clips_output = []
        base_progress = 60.0
        progress_per_clip = 35.0 / max(1, len(top_clips))

        for idx, clip in enumerate(top_clips, start=1):
            cur_pct = base_progress + (idx - 1) * progress_per_clip
            c_start = clip["start_time"]
            c_end = clip["end_time"]
            c_dur = clip["duration"]

            report(cur_pct, f"Processing Clip #{idx} ({c_start:.1f}s - {c_end:.1f}s)...")

            clip_stem = f"clip_{idx}"
            raw_clip_path = str(self.output_dir / "temp" / f"{clip_stem}_raw.mp4")
            clip_audio_path = str(self.output_dir / "temp" / f"{clip_stem}_audio.wav")
            clip_ass_path = str(self.output_dir / "temp" / f"{clip_stem}_captions.ass")
            final_mp4_path = str(self.output_dir / f"{clip_stem}_9x16.mp4")
            metadata_txt_path = str(self.output_dir / f"{clip_stem}_metadata.txt")

            # 1. Cut clip segment
            self.downloader.download_video_segment(
                url_or_path=source_video,
                start_time=c_start,
                end_time=c_end,
                output_path=raw_clip_path,
            )

            # 2. Extract clip audio for faster-whisper
            self.downloader.download_audio_track(raw_clip_path, clip_audio_path)

            # 3. Transcribe audio with word timestamps
            report(cur_pct + progress_per_clip * 0.3, f"Clip #{idx}: Generating AI word captions...")
            transcription = self.transcriber.transcribe(clip_audio_path)

            # 4. Generate bold animated ASS subtitles centered in middle (y=960)
            self.transcriber.generate_animated_ass(
                words_data=transcription["words"],
                output_ass_path=clip_ass_path,
                relative_start_time=0.0,
                y_position=960,
            )

            # 5. Crop 9:16 vertical split-screen with middle border and burn subtitles
            report(cur_pct + progress_per_clip * 0.6, f"Clip #{idx}: Rendering 9:16 vertical split-screen...")
            self.cropper.render_split_screen(
                input_video_path=raw_clip_path,
                output_video_path=final_mp4_path,
                facecam_crop=facecam_crop,
                ass_subtitles_path=clip_ass_path,
            )

            # 6. Generate viral hook titles, description, hashtags, and text file
            report(cur_pct + progress_per_clip * 0.85, f"Clip #{idx}: Generating viral hook metadata...")
            clip_meta = self.meta_gen.generate_clip_metadata(
                clip_info=clip,
                transcript=transcription["full_transcript"],
                channel=channel,
                stream_title=video_title,
                output_txt_path=metadata_txt_path,
            )

            clip_meta["video_file"] = final_mp4_path
            clip_meta["relative_video_file"] = os.path.basename(final_mp4_path)
            clip_meta["relative_metadata_file"] = os.path.basename(metadata_txt_path)
            clip_meta["words_count"] = len(transcription["words"])
            clips_output.append(clip_meta)

        # Write overall summary JSON
        summary_payload = {
            "source_video": {
                "title": video_title,
                "channel": channel,
                "duration": video_duration,
                "heatmap_available": len(heatmap) > 0,
                "chat_available": chat_messages is not None,
            },
            "facecam_crop": facecam_crop,
            "clips": clips_output,
            "timeline": analysis_result["timeline"],
            "processing_time_sec": round(time.time() - start_time_bench, 2),
        }

        summary_file = self.output_dir / "all_clips_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        report(100, f"Auto-clipping complete! All 3 clips generated in {summary_payload['processing_time_sec']}s.")
        return summary_payload
