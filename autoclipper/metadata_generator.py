"""
Viral metadata generator module for AutoClipper.
Generates high-CTR viral hook titles, SEO descriptions, and trending hashtags
for each extracted clip and saves formatted metadata text files.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("autoclipper.metadata_generator")


class ViralMetadataGenerator:
    """Generates viral titles, descriptions, and hashtags tailored for Shorts/TikTok/Reels."""

    EMOJIS = ["💀", "🤯", "🔥", "😱", "😭", "⚡", "👀", "🚨", "🚀", "💥"]

    HASHTAG_POOLS = {
        "core": ["#shorts", "#fyp", "#viral", "#trending", "#gaming"],
        "stream": ["#streamer", "#twitch", "#twitchclips", "#youtubeclips", "#streamerhighlights"],
        "hype": ["#insane", "#clutch", "#ragequit", "#funnygaming", "#gamermoments", "#epicwin"],
    }

    def generate_clip_metadata(
        self,
        clip_info: Dict[str, Any],
        transcript: str,
        channel: str = "Streamer",
        stream_title: str = "Stream",
        output_txt_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Creates viral title, descriptions, hashtags, and saves to clip_{i}_metadata.txt.
        """
        rank = clip_info.get("rank", 1)
        start_time = clip_info.get("start_time", 0.0)
        end_time = clip_info.get("end_time", 0.0)
        duration = clip_info.get("duration", 45.0)
        viral_score = clip_info.get("viral_score", 90.0)
        reason = clip_info.get("primary_reason", "High Audio & Replay Activity")

        # Generate contextual titles based on transcript and reason
        hook_titles = self._generate_hook_titles(transcript, rank, stream_title, reason)
        primary_title = hook_titles[0]

        # Generate description
        description = self._generate_description(
            primary_title, transcript, channel, start_time, end_time
        )

        # Generate hashtags
        hashtags = self._generate_hashtags(transcript)
        hashtags_str = " ".join(hashtags)

        metadata_payload = {
            "clip_id": f"clip_{rank}",
            "rank": rank,
            "title": primary_title,
            "alternative_titles": hook_titles[1:],
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "viral_score": viral_score,
            "primary_reason": reason,
            "description": description,
            "hashtags": hashtags,
            "hashtags_string": hashtags_str,
            "transcript": transcript.strip(),
        }

        # Write formatted text file
        if output_txt_path:
            txt_path = Path(output_txt_path).resolve()
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_text_file(txt_path, metadata_payload)
            metadata_payload["file_path"] = str(txt_path)

        return metadata_payload

    def _generate_hook_titles(
        self,
        transcript: str,
        rank: int,
        stream_title: str,
        reason: str,
    ) -> List[str]:
        """Generates 4 high-CTR viral hook titles with varied psychological angles."""
        titles = []
        clean_transcript = transcript.strip().replace('"', "")

        # Look for catchy punchlines in the transcript
        quote_title = None
        if len(clean_transcript) > 5:
            # Grab first 4-7 words for quote
            words = clean_transcript.split()
            short_quote = " ".join(words[:min(6, len(words))]).upper()
            if len(short_quote) > 10:
                quote_title = f'"{short_quote}..." 💀 (HE LOST IT)'

        templates_rank_1 = [
            "HE REALLY THOUGHT HE WAS GOING TO SURVIVE THIS... 💀",
            "THE MOMENT EVERYTHING WENT COMPLETELY WRONG 🤯 #Shorts",
            "NO ONE COULD BELIEVE WHAT HAPPENED NEXT... 🔥",
            "BRO LITERALLY RAGED AFTER THIS 1-IN-A-MILLION LOSS 😭",
            "HOW DID HE EVEN PULL THIS OFF?! ⚡",
        ]
        templates_rank_2 = [
            "WHEN THE ENEMY DOES NOT REALIZE WHO THEY ARE PLAYING 💀",
            "WATCH TILL THE VERY END... YOU WILL NOT REGRET IT 😱",
            "THE MOST UNLUCKY TIMING IN GAMING HISTORY 😭",
            "HIS REACTION AT THE END IS ABSOLUTELY PRICELESS 🤣",
            "THE CLEANEST CLUTCH YOU WILL SEE TODAY 🔥",
        ]
        templates_rank_3 = [
            "HE DID NOT SEE THIS COMING AT ALL... 💀",
            "IS THIS THE GREATEST GAMING MOMENT OF 2026?! 🤯",
            "THEY NEVER EXPECTED HIM TO ACTUALLY DO THIS 🔥",
            "EVEN THE CHAT WAS COMPLETELY SPEECHLESS... 😱",
            "BRO WENT FULL GOBLIN MODE ON STREAM 💀",
        ]

        if rank == 1:
            pool = templates_rank_1
        elif rank == 2:
            pool = templates_rank_2
        else:
            pool = templates_rank_3

        if quote_title:
            titles.append(quote_title)

        for t in pool:
            if t not in titles:
                titles.append(t)
            if len(titles) >= 4:
                break

        return titles

    def _generate_description(
        self,
        title: str,
        transcript: str,
        channel: str,
        start_time: float,
        end_time: float,
    ) -> str:
        """Generates SEO-friendly video description with timestamps and calls to action."""
        start_fmt = f"{int(start_time // 60):02d}:{int(start_time % 60):02d}"
        end_fmt = f"{int(end_time // 60):02d}:{int(end_time % 60):02d}"

        quote_line = f'"{transcript[:120]}..."' if transcript else "Watch this insane stream moment!"

        return (
            f"{title}\n\n"
            f"You won't believe how this ended! Watch till the very last second! 😱\n\n"
            f"🎯 Clip Timestamp: {start_fmt} - {end_fmt}\n"
            f"🎙️ Streamer: {channel}\n"
            f"💬 Highlight Quote: {quote_line}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"👉 Drop a LIKE if you would have reacted the same way!\n"
            f"🔔 SUBSCRIBE for daily stream highlights, funniest gaming clips, and insane clutches!\n"
            f"💬 Drop a comment: What would you rate this play from 1 to 10?\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

    def _generate_hashtags(self, transcript: str) -> List[str]:
        """Assembles balanced hashtag set for viral discovery."""
        tags = list(self.HASHTAG_POOLS["core"])
        tags.extend(self.HASHTAG_POOLS["stream"][:3])
        tags.extend(self.HASHTAG_POOLS["hype"][:3])

        # Context-dependent tags
        lower = transcript.lower()
        if any(w in lower for w in ["cheat", "aimbot", "hacker"]):
            tags.append("#hacker")
        if any(w in lower for w in ["rage", "mad", "scream"]):
            tags.append("#rage")
        if any(w in lower for w in ["win", "won", "clutch", "ace"]):
            tags.append("#clutch")
        if any(w in lower for w in ["funny", "laugh", "lmao", "lol"]):
            tags.append("#funny")

        # Deduplicate while preserving order
        seen = set()
        unique_tags = []
        for t in tags:
            if t not in seen:
                seen.add(t)
                unique_tags.append(t)

        return unique_tags[:10]

    def _write_text_file(self, file_path: Path, meta: Dict[str, Any]):
        """Writes human-readable, copy-paste ready metadata text file."""
        content = f"""================================================================================
VIRAL CLIP #{meta['rank']} - PUBLISHING METADATA KIT
Generated by AutoClipper (Local AI Stream Clipper)
================================================================================

[RECOMMENDED VIRAL HOOK TITLE]
{meta['title']}

[ALTERNATIVE TITLES FOR A/B TESTING]
"""
        for i, alt in enumerate(meta.get("alternative_titles", []), 1):
            content += f"{i}. {alt}\n"

        content += f"""
--------------------------------------------------------------------------------
[VIDEO METRICS & SOURCE INFO]
Clip Duration : {meta['duration']} seconds ({meta['start_time']:.1f}s - {meta['end_time']:.1f}s)
Viral Score   : {meta['viral_score']} / 100
Primary Reason: {meta['primary_reason']}
Format        : 9:16 Vertical (1080x1920 Split-Screen with Animated Captions)

--------------------------------------------------------------------------------
[SEO OPTIMIZED DESCRIPTION]
{meta['description']}

--------------------------------------------------------------------------------
[TRENDING HASHTAGS (Click & Copy)]
{meta['hashtags_string']}

--------------------------------------------------------------------------------
[TRANSCRIPT EXCERPT]
"{meta['transcript']}"

================================================================================
Ready to upload to YouTube Shorts, TikTok, and Instagram Reels!
================================================================================
"""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Saved viral metadata text file: {file_path}")
