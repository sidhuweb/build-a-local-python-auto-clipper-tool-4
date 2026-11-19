"""
Cropper module for AutoClipper.
Assembles 9:16 vertical split-screen videos (1080x1920) with streamer facecam on top,
gameplay on bottom, animated captions in the middle, and sleek accent divider line.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("autoclipper.cropper")


class VerticalSplitScreenCropper:
    """Builds and executes FFmpeg 9:16 vertical split-screen rendering."""

    def __init__(
        self,
        output_width: int = 1080,
        output_height: int = 1920,
        divider_height: int = 8,
        divider_color: str = "0x00F0FF@0.9",  # Neon Cyan
        fonts_dir: Optional[str] = None,
    ):
        self.output_width = output_width
        self.output_height = output_height
        self.split_height = output_height // 2  # 960 each
        self.divider_height = divider_height
        self.divider_color = divider_color

        if fonts_dir and os.path.exists(fonts_dir):
            self.fonts_dir = str(Path(fonts_dir).resolve())
        else:
            default_fdir = Path(__file__).parent.parent / "assets" / "fonts"
            self.fonts_dir = str(default_fdir.resolve()) if default_fdir.exists() else None

    def render_split_screen(
        self,
        input_video_path: str,
        output_video_path: str,
        facecam_crop: Dict[str, int],
        gameplay_crop: Optional[Dict[str, int]] = None,
        ass_subtitles_path: Optional[str] = None,
        start_time: Optional[float] = None,
        duration: Optional[float] = None,
        preset: str = "veryfast",
        crf: int = 20,
    ) -> str:
        """
        Renders the 9:16 vertical split-screen video with optional subtitle burn.
        """
        output_path = Path(output_video_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fc_x = facecam_crop.get("x", 0)
        fc_y = facecam_crop.get("y", 0)
        fc_w = facecam_crop.get("w", 640)
        fc_h = facecam_crop.get("h", 480)

        # Top half: Facecam
        # Scale & center-crop to 1080x960
        top_filter = (
            f"[0:v]crop={fc_w}:{fc_h}:{fc_x}:{fc_y},"
            f"scale={self.output_width}:{self.split_height}:force_original_aspect_ratio=increase,"
            f"crop={self.output_width}:{self.split_height}[top]"
        )

        # Bottom half: Gameplay
        # Default gameplay is center-weighted 16:9 crop
        if gameplay_crop:
            gp_x = gameplay_crop.get("x", 0)
            gp_y = gameplay_crop.get("y", 0)
            gp_w = gameplay_crop.get("w", 1280)
            gp_h = gameplay_crop.get("h", 720)
            bot_filter = (
                f"[0:v]crop={gp_w}:{gp_h}:{gp_x}:{gp_y},"
                f"scale={self.output_width}:{self.split_height}:force_original_aspect_ratio=increase,"
                f"crop={self.output_width}:{self.split_height}[bot]"
            )
        else:
            # Smart default: center 75% crop of main action
            bot_filter = (
                f"[0:v]crop=iw*0.75:ih*0.75:(iw-out_w)/2:(ih-out_h)/2,"
                f"scale={self.output_width}:{self.split_height}:force_original_aspect_ratio=increase,"
                f"crop={self.output_width}:{self.split_height}[bot]"
            )

        # Stack top and bottom vertically -> total 1080x1920
        stack_filter = "[top][bot]vstack[vsplit]"

        # Divider bar in the middle
        div_y = self.split_height - (self.divider_height // 2)
        effects_filter = f"[vsplit]drawbox=y={div_y}:h={self.divider_height}:color={self.divider_color}:t=fill"

        # Burn ASS subtitles if provided
        if ass_subtitles_path and os.path.exists(ass_subtitles_path):
            ass_clean_path = str(Path(ass_subtitles_path).resolve()).replace("\\", "/").replace(":", "\\:")
            if self.fonts_dir and os.path.exists(self.fonts_dir):
                fonts_clean = self.fonts_dir.replace("\\", "/").replace(":", "\\:")
                effects_filter += f",ass='{ass_clean_path}':fontsdir='{fonts_clean}'"
            else:
                effects_filter += f",ass='{ass_clean_path}'"

        effects_filter += "[outv]"

        filter_complex = f"{top_filter};{bot_filter};{stack_filter};{effects_filter}"

        cmd = ["ffmpeg", "-y"]

        if start_time is not None:
            cmd.extend(["-ss", f"{start_time:.3f}"])

        cmd.extend(["-i", input_video_path])

        if duration is not None:
            cmd.extend(["-t", f"{duration:.3f}"])

        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-c:a", "aac",
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ])

        logger.info(f"Running FFmpeg 9:16 split-screen rendering to {output_path}...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            logger.error(f"FFmpeg error: {res.stderr}")
            raise RuntimeError(f"FFmpeg rendering failed: {res.stderr[-400:]}")

        logger.info(f"Successfully rendered 9:16 vertical split-screen: {output_path}")
        return str(output_path)
