"""
AutoClipper - AI-Powered Stream & Video Auto-Clipper
Transforms YouTube streams and gameplay videos into viral 9:16 vertical Shorts/TikToks.
"""

__version__ = "1.0.0"

from autoclipper.downloader import VideoDownloader
from autoclipper.analyzer import ViralMomentAnalyzer
from autoclipper.face_tracker import FaceCamDetector
from autoclipper.cropper import VerticalSplitScreenCropper
from autoclipper.transcriber import CaptionGenerator
from autoclipper.metadata_generator import ViralMetadataGenerator
from autoclipper.pipeline import AutoClipperPipeline

__all__ = [
    "VideoDownloader",
    "ViralMomentAnalyzer",
    "FaceCamDetector",
    "VerticalSplitScreenCropper",
    "CaptionGenerator",
    "ViralMetadataGenerator",
    "AutoClipperPipeline",
]
