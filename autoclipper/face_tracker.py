"""
FaceCam detector module for AutoClipper.
Uses OpenCV Haar Cascades to automatically detect and track the streamer's webcam box,
with intelligent corner presets and boundary normalization for 9:16 split-screen framing.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger("autoclipper.face_tracker")


class FaceCamDetector:
    """Detects streamer facecam location in video or applies corner presets."""

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
            # Check default asset paths
            default_asset = Path(__file__).parent.parent / "assets" / "models" / "haarcascade_frontalface_default.xml"
            if default_asset.exists():
                self.cascade_path = str(default_asset)
            else:
                self.cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        self.classifier = None
        if os.path.exists(self.cascade_path):
            try:
                self.classifier = cv2.CascadeClassifier(self.cascade_path)
            except Exception as e:
                logger.warning(f"Could not load Haar cascade: {e}")

    def detect_facecam(
        self,
        video_path: str,
        start_time: float = 0.0,
        sample_duration: float = 30.0,
        num_samples: int = 15,
        preset_fallback: str = "top-right",
    ) -> Dict[str, int]:
        """
        Samples frames from video to detect persistent streamer face coordinates.
        Returns dict with crop parameters: {'x': x, 'y': y, 'w': w, 'h': h}.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Failed to open video {video_path}, using preset fallback.")
            return self.get_preset_crop(1920, 1080, preset_fallback)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        detected_faces = []

        if self.classifier is not None and not self.classifier.empty():
            start_frame = int(start_time * fps)
            step_frame = max(1, int((sample_duration * fps) / num_samples))

            for i in range(num_samples):
                frame_idx = min(start_frame + i * step_frame, total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Equalize histogram for contrast
                gray = cv2.equalizeHist(gray)

                # Prioritize upper corners and edges where streamer facecam is located
                faces = self.classifier.detectMultiScale(
                    gray,
                    scaleFactor=1.15,
                    minNeighbors=4,
                    minSize=(int(height * 0.08), int(height * 0.08)),
                    maxSize=(int(height * 0.50), int(height * 0.50)),
                )

                for (x, y, w, h) in faces:
                    detected_faces.append((x, y, w, h))

        cap.release()

        # If faces detected, cluster and find most frequent webcam region
        if detected_faces:
            crop = self._cluster_and_bound_faces(detected_faces, width, height)
            logger.info(f"Auto-detected streamer facecam: {crop}")
            return crop

        # Fallback to smart corner preset
        logger.info(f"No clear face detected, applying {preset_fallback} preset crop.")
        return self.get_preset_crop(width, height, preset_fallback)

    def _cluster_and_bound_faces(
        self,
        faces: List[Tuple[int, int, int, int]],
        frame_w: int,
        frame_h: int,
    ) -> Dict[str, int]:
        """Expands detected face bounds to capture the full streamer webcam frame."""
        # Convert to numpy array
        arr = np.array(faces)
        # Average face box
        med_x = np.median(arr[:, 0])
        med_y = np.median(arr[:, 1])
        med_w = np.median(arr[:, 2])
        med_h = np.median(arr[:, 3])

        # Center of face
        center_x = med_x + med_w / 2.0
        center_y = med_y + med_h / 2.0

        # We want top half aspect ratio in split screen:
        # Split screen top half is 1080x960 (ratio 9:8 = 1.125) or 16:9 webcam scaled
        target_aspect = 1.125

        # Expand box by factor of 2.2 to include hair, headset, shoulders
        box_h = med_h * 2.3
        box_w = box_h * target_aspect

        # Minimum webcam size
        min_w = frame_w * 0.25
        min_h = frame_h * 0.25
        box_w = max(box_w, min_w)
        box_h = max(box_h, min_h)

        # Calculate top-left corner
        x1 = int(center_x - box_w / 2.0)
        y1 = int(center_y - box_h / 2.0)
        x2 = int(x1 + box_w)
        y2 = int(y1 + box_h)

        # Clamp inside frame boundaries
        if x1 < 0:
            x2 += abs(x1)
            x1 = 0
        if y1 < 0:
            y2 += abs(y1)
            y1 = 0
        if x2 > frame_w:
            shift = x2 - frame_w
            x1 = max(0, x1 - shift)
            x2 = frame_w
        if y2 > frame_h:
            shift = y2 - frame_h
            y1 = max(0, y1 - shift)
            y2 = frame_h

        crop_w = int(x2 - x1)
        crop_h = int(y2 - y1)

        # Ensure even numbers for ffmpeg
        crop_w = crop_w - (crop_w % 2)
        crop_h = crop_h - (crop_h % 2)
        x1 = x1 - (x1 % 2)
        y1 = y1 - (y1 % 2)

        return {"x": x1, "y": y1, "w": crop_w, "h": crop_h}

    def get_preset_crop(self, frame_w: int, frame_h: int, preset: str = "top-right") -> Dict[str, int]:
        """Generates crop coordinates from standard streamer corner presets."""
        fx, fy, fw, fh = self.PRESETS.get(preset, self.PRESETS["top-right"])
        x = int(frame_w * fx)
        y = int(frame_h * fy)
        w = int(frame_w * fw)
        h = int(frame_h * fh)

        # Even dimensions
        w = w - (w % 2)
        h = h - (h % 2)
        x = x - (x % 2)
        y = y - (y % 2)

        return {"x": x, "y": y, "w": w, "h": h}
