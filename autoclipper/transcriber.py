"""
Transcriber and Animated Caption Generator module for AutoClipper.
Uses faster-whisper to extract word-level timestamps, and produces bold,
TikTok/Reels/Shorts-style animated ASS subtitles centered in the middle of the vertical frame.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("autoclipper.transcriber")


class CaptionGenerator:
    """Transcribes audio using faster-whisper and generates bold animated ASS captions."""

    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        font_name: str = "Montserrat ExtraBold",
        font_size: int = 70,
        highlight_color: str = "&H0000FFFF",  # Bright Yellow (BGR format in ASS)
        primary_color: str = "&H00FFFFFF",    # Crisp White
        outline_color: str = "&H00000000",    # Solid Black
        outline_width: int = 7,
        words_per_line: int = 3,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.font_name = font_name
        self.font_size = font_size
        self.highlight_color = highlight_color
        self.primary_color = primary_color
        self.outline_color = outline_color
        self.outline_width = outline_width
        self.words_per_line = words_per_line

        self._model = None

    def _get_model(self):
        """Lazy load faster-whisper model."""
        if self._model is None:
            from faster_whisper import WhisperModel
            logger.info(f"Loading faster-whisper model '{self.model_size}' on {self.device} ({self.compute_type})...")
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribes audio file and returns list of word dictionaries with start and end times.
        """
        model = self._get_model()
        logger.info(f"Transcribing {audio_path} with word-level timestamps...")

        segments, info = model.transcribe(
            audio_path,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
        )

        words_list = []
        full_transcript_parts = []

        for seg in segments:
            full_transcript_parts.append(seg.text.strip())
            for w in (seg.words or []):
                cleaned_word = w.word.strip()
                if cleaned_word:
                    words_list.append({
                        "word": cleaned_word.upper(),
                        "start": float(w.start),
                        "end": float(w.end),
                        "probability": float(w.probability),
                    })

        full_transcript = " ".join(full_transcript_parts)
        logger.info(f"Transcribed {len(words_list)} words (language: {info.language}).")

        return {
            "language": info.language,
            "full_transcript": full_transcript,
            "words": words_list,
        }

    def generate_animated_ass(
        self,
        words_data: List[Dict[str, Any]],
        output_ass_path: str,
        relative_start_time: float = 0.0,
        y_position: int = 960,  # Exact middle of 1080x1920 canvas
    ) -> str:
        """
        Generates an Advanced SubStation Alpha (.ass) file with dynamic word-by-word highlight.
        """
        out_path = Path(output_ass_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        header = f"""[Script Info]
Title: AutoClipper Bold Viral Captions
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ViralCaption,{self.font_name},{self.font_size},{self.primary_color},&H000000FF,{self.outline_color},&H90000000,-1,0,0,0,100,100,0,0,1,{self.outline_width},4,5,60,60,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        events = []

        if not words_data:
            # Empty subtitle fallback
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(header)
            return str(out_path)

        # Group words into chunks of `self.words_per_line` (e.g. 2 to 3 words)
        chunks = []
        for i in range(0, len(words_data), self.words_per_line):
            chunk = words_data[i:i + self.words_per_line]
            if chunk:
                chunks.append(chunk)

        for chunk in chunks:
            chunk_start = max(0.0, chunk[0]["start"] - relative_start_time)
            chunk_end = max(chunk_start + 0.3, chunk[-1]["end"] - relative_start_time)

            # Within each chunk, create an animated event for each active word
            for active_idx, target_word in enumerate(chunk):
                w_start = max(chunk_start, target_word["start"] - relative_start_time)
                # Word end is next word's start or chunk end
                if active_idx + 1 < len(chunk):
                    w_end = max(w_start + 0.15, chunk[active_idx + 1]["start"] - relative_start_time)
                else:
                    w_end = chunk_end

                if w_end <= w_start:
                    w_end = w_start + 0.2

                # Format start and end timestamps (H:MM:SS.cs)
                start_str = self._format_ass_time(w_start)
                end_str = self._format_ass_time(w_end)

                # Build line text where active word has highlight color + scale pop
                line_parts = []
                for idx, w in enumerate(chunk):
                    clean_text = self._escape_ass(w["word"])
                    if idx == active_idx:
                        # Highlighted word with scale pop (112%) and glowing highlight color
                        line_parts.append(
                            f"{{\\c{self.highlight_color}\\fscx112\\fscy112}}{clean_text}{{\\r}}"
                        )
                    else:
                        line_parts.append(f"{{\\c{self.primary_color}}}{clean_text}{{\\r}}")

                formatted_line = " ".join(line_parts)
                # Position right in the center (pos(540, y_position))
                pos_tag = f"{{\\pos(540,{y_position})}}"
                event_line = f"Dialogue: 0,{start_str},{end_str},ViralCaption,,0,0,0,,{pos_tag}{formatted_line}"
                events.append(event_line)

        ass_content = header + "\n".join(events) + "\n"

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(ass_content)

        logger.info(f"Generated animated ASS subtitles with {len(events)} events: {out_path}")
        return str(out_path)

    def _format_ass_time(self, seconds: float) -> str:
        """Converts float seconds to ASS timestamp format: H:MM:SS.cc"""
        total_cs = int(round(seconds * 100))
        cs = total_cs % 100
        total_s = total_cs // 100
        s = total_s % 60
        total_m = total_s // 60
        m = total_m % 60
        h = total_m // 60
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    def _escape_ass(self, text: str) -> str:
        """Sanitizes text for ASS format."""
        return text.replace("{", "(").replace("}", ")").replace("\\", "/")
