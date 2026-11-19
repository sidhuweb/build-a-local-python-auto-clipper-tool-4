"""
Command-Line Interface (CLI) for AutoClipper.
Run with: python -m autoclipper clip <URL_OR_FILE> [OPTIONS]
"""

import argparse
import os
import sys
import time
from pathlib import Path

from autoclipper.pipeline import AutoClipperPipeline

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_banner():
    banner = f"""
{CYAN}{BOLD}
   █████╗ ██╗   ██╗████████╗ ██████╗  ██████╗██╗     ██╗██████╗ ██████╗ ███████╗██████╗ 
  ██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗██╔════╝██║     ██║██╔══██╗██╔══██╗██╔════╝██╔══██╗
  ███████║██║   ██║   ██║   ██║   ██║██║     ██║     ██║██████╔╝██████╔╝█████╗  ██████╔╝
  ██╔══██║██║   ██║   ██║   ██║   ██║██║     ██║     ██║██╔═══╝ ██╔═══╝ ██╔══╝  ██╔══██╗
  ██║  ██║╚██████╔╝   ██║   ╚██████╔╝╚██████╗███████╗██║██║     ██║     ███████╗██║  ██║
  ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝  ╚═════╝╚══════╝╚═╝╚═╝     ╚═╝     ╚══════╝╚═╝  ╚═╝
{RESET}{YELLOW}       🔥 AI-Powered Viral Moments, 9:16 Split-Screen & Bold Animated Captions 🔥{RESET}
"""
    print(banner)


def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="AutoClipper: AI Stream & Video Auto-Clipper into 9:16 Vertical Shorts/TikToks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="YouTube stream/video URL, or local MP4 video file path",
    )
    parser.add_argument(
        "--url", "-u",
        dest="url_flag",
        help="YouTube stream/video URL (alternative to positional argument)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Directory to save generated clips and metadata (default: output/)",
    )
    parser.add_argument(
        "--clip-duration", "-d",
        type=float,
        default=45.0,
        help="Duration of each viral clip in seconds (default: 45.0)",
    )
    parser.add_argument(
        "--whisper-model", "-m",
        default="tiny",
        choices=["tiny", "base", "small", "medium"],
        help="faster-whisper model size (default: tiny for ultra-fast CPU inference)",
    )
    parser.add_argument(
        "--facecam",
        default="top-right",
        choices=["auto", "top-right", "top-left", "bottom-right", "bottom-left", "top-center"],
        help="Streamer facecam webcam location (default: top-right)",
    )
    parser.add_argument(
        "--cookies",
        help="Path to cookies.txt file for YouTube authentication",
    )
    parser.add_argument(
        "--cookies-from-browser",
        choices=["chrome", "firefox", "brave", "edge", "safari"],
        help="Extract YouTube cookies from local browser",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run on built-in realistic mock stream sample (instant local offline test)",
    )

    args = parser.parse_args()
    target = args.target or args.url_flag

    if args.demo:
        print(f"{YELLOW}[DEMO MODE]{RESET} Running test on realistic sample stream...")
        # Check if sample video exists, if not generate it
        sample_path = Path("samples/sample_stream.mp4").resolve()
        if not sample_path.exists():
            print(f"{CYAN}Generating sample stream with 3 viral moments...{RESET}")
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
    print(f"{BOLD}Whisper Model:{RESET} {args.whisper_model}")
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
        bar_len = 28
        filled = int(bar_len * (pct / 100.0))
        bar = "█" * filled + "░" * (bar_len - filled)
        sys.stdout.write(f"\r{CYAN}[{bar}] {pct:3.0f}%{RESET} | {stage:<55}")
        sys.stdout.flush()
        if pct >= 100:
            sys.stdout.write("\n")

    try:
        results = pipeline.process(
            url_or_path=target,
            progress_callback=on_progress,
            top_k=3,
        )

        print(f"\n{GREEN}{BOLD}══════════════════════════════════════════════════════════════════════════════{RESET}")
        print(f"{GREEN}{BOLD} 🎉 SUCCESS! Top 3 Viral Clips Generated in {results['processing_time_sec']}s{RESET}")
        print(f"{GREEN}{BOLD}══════════════════════════════════════════════════════════════════════════════{RESET}\n")

        for clip in results["clips"]:
            rank = clip["rank"]
            vid_path = clip["video_file"]
            txt_path = clip.get("file_path", "")
            print(f"{YELLOW}▶ CLIP #{rank}:{RESET} {BOLD}{clip['title']}{RESET}")
            print(f"  • Timecode   : {clip['start_time']:.1f}s - {clip['end_time']:.1f}s (Viral Score: {clip['viral_score']}/100)")
            print(f"  • Reason     : {clip['primary_reason']}")
            print(f"  • Video (MP4): {CYAN}file://{vid_path}{RESET}")
            print(f"  • Metadata   : {CYAN}file://{txt_path}{RESET}")
            print(f"  • Hashtags   : {clip['hashtags_string']}")
            print(f"  • Transcript : \"{clip['transcript'][:80]}...\"\n")

        summary_file = os.path.join(args.output_dir, "all_clips_summary.json")
        print(f"{BOLD}Full JSON Summary:{RESET} {CYAN}file://{os.path.abspath(summary_file)}{RESET}\n")

    except Exception as e:
        print(f"\n{RED}{BOLD}AutoClipper Error:{RESET} {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
