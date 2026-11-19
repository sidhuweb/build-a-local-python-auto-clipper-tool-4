"""
Sample Stream Generator for AutoClipper.
Generates a realistic 1920x1080 gaming stream video with a streamer facecam in top-right,
dynamic gameplay visuals, and real TTS speech with 3 distinct viral hype/audio spikes.
Also generates matching YouTube heatmap and live chat replay JSON files.
"""

import asyncio
import json
import os
import subprocess
from pathlib import Path
import numpy as np


async def create_speech_tracks(output_dir: Path):
    """Generates speech audio tracks using edge-tts."""
    import edge_tts

    dialogues = [
        # Intro (Calm)
        (0.0, "calm_intro", "Alright team, checking the radar. Let's push towards the B site carefully."),
        # Moment 1 (Hype Spike 1)
        (22.0, "spike_1", "Oh my god! He is one HP! Look at this shot! No way he hit that! He is definitely cheating bro!"),
        # Interlude (Calm)
        (42.0, "calm_mid", "Okay chill out chat, let me grab some armor and reload my weapon."),
        # Moment 2 (Laugh / Shock Spike 2)
        (56.0, "spike_2", "Wait what just happened? Did he actually jump off the roof? Are you serious right now?"),
        # Interlude 2 (Calm)
        (76.0, "calm_mid2", "Final round, 1 versus 4 situation. Time to lock in chat."),
        # Moment 3 (Clutch Spike 3)
        (88.0, "spike_3", "Lets go! One versus four clutch for the win! Clip that chat! Clip that right now!"),
    ]

    files = []
    for offset, tag, text in dialogues:
        fpath = output_dir / f"{tag}.mp3"
        tts = edge_tts.Communicate(text, "en-US-ChristopherNeural")
        await tts.save(str(fpath))
        files.append((offset, str(fpath), text))

    return files


def generate_sample_stream(output_mp4_path: str = "samples/sample_stream.mp4") -> str:
    """Generates the full realistic gaming stream video with audio spikes and webcam overlay."""
    out_path = Path(output_mp4_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = out_path.parent / "temp_sample"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print("Generating speech audio segments...")
    speech_items = asyncio.run(create_speech_tracks(temp_dir))

    total_duration = 115.0  # 115 seconds stream

    # Assemble full audio track with background game sound and boosted audio spikes
    print("Assembling audio track with distinct viral volume spikes...")
    # Generate background low hum/game ambience
    bg_audio = temp_dir / "bg_ambience.wav"
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"anoisesrc=d={total_duration}:c=pink:a=0.03",
        "-ar", "16000", "-ac", "1",
        str(bg_audio)
    ], capture_output=True, check=True)

    # Place each speech item at its time offset and amplify viral spikes (+8dB to +12dB)
    inputs = ["-i", str(bg_audio)]
    filter_parts = ["[0:a]volume=0.4[a0];"]
    amix_inputs = ["[a0]"]

    for idx, (offset_sec, fpath, text) in enumerate(speech_items, start=1):
        inputs.extend(["-i", fpath])
        # Spike clips get heavy volume boost (+8dB) to trigger audio spike detection
        is_spike = "spike" in fpath
        vol = "2.8" if is_spike else "1.1"
        delay_ms = int(offset_sec * 1000)
        filter_parts.append(
            f"[{idx}:a]volume={vol},adelay={delay_ms}|{delay_ms}[a{idx}];"
        )
        amix_inputs.append(f"[a{idx}]")

    filter_complex = "".join(filter_parts) + f"{''.join(amix_inputs)}amix=inputs={len(amix_inputs)}:normalize=0[outa]"

    full_audio = temp_dir / "full_sample_audio.wav"
    cmd_audio = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_complex,
        "-map", "[outa]",
        "-ar", "16000",
        "-ac", "1",
        "-t", str(total_duration),
        str(full_audio)
    ]
    subprocess.run(cmd_audio, capture_output=True, check=True)

    # Create streamer face image with webcam frame in top-right
    print("Generating video with streamer facecam in top-right corner...")
    import cv2
    face_img = np.zeros((360, 480, 3), dtype=np.uint8)
    # Background webcam room
    face_img[:, :] = (35, 40, 50)
    # Webcam border (neon purple border)
    cv2.rectangle(face_img, (0, 0), (480, 360), (220, 50, 160), 6)
    # Streamer Face oval
    cv2.ellipse(face_img, (240, 180), (85, 115), 0, 0, 360, (180, 210, 235), -1)
    # Eyes
    cv2.circle(face_img, (205, 160), 10, (50, 40, 30), -1)
    cv2.circle(face_img, (275, 160), 10, (50, 40, 30), -1)
    # Eyebrows
    cv2.line(face_img, (195, 140), (220, 145), (40, 30, 20), 4)
    cv2.line(face_img, (260, 145), (285, 140), (40, 30, 20), 4)
    # Nose
    cv2.line(face_img, (240, 175), (240, 200), (140, 160, 190), 3)
    # Smile
    cv2.ellipse(face_img, (240, 225), (35, 20), 0, 0, 180, (40, 40, 180), -1)
    # Hair
    cv2.ellipse(face_img, (240, 105), (95, 50), 0, 180, 360, (20, 20, 20), -1)
    # Headset
    cv2.ellipse(face_img, (240, 130), (110, 80), 0, 180, 360, (80, 80, 80), 8)
    cv2.rectangle(face_img, (125, 150), (145, 210), (30, 30, 30), -1)
    cv2.rectangle(face_img, (335, 150), (355, 210), (30, 30, 30), -1)
    # Streamer name tag
    cv2.rectangle(face_img, (150, 315), (330, 350), (0, 0, 0), -1)
    cv2.putText(face_img, "PRO_STREAMER", (160, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    webcam_png = temp_dir / "webcam_overlay.png"
    cv2.imwrite(str(webcam_png), face_img)

    # Render video: animated test gaming background + webcam overlay in top-right (x=1400, y=40)
    # Main gameplay has dynamic moving elements
    # Main gameplay has dynamic moving elements & HUD bars (health, ammo, crosshair)
    gameplay_filter = (
        f"[0:v]drawbox=x=860:y=440:w=200:h=200:color=red@0.5:t=4,"
        f"drawbox=x=958:y=530:w=4:h=20:color=green:t=fill,"
        f"drawbox=x=950:y=538:w=20:h=4:color=green:t=fill,"
        f"drawbox=x=60:y=60:w=320:h=45:color=black@0.7:t=fill,"
        f"drawbox=x=65:y=65:w=180:h=35:color=blue@0.8:t=fill,"
        f"drawbox=x=60:y=1000:w=350:h=45:color=black@0.7:t=fill,"
        f"drawbox=x=65:y=1005:w=280:h=35:color=green@0.8:t=fill[game];"
        f"[game][1:v]overlay=x=1400:y=40[outv]"
    )

    cmd_video = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc=size=1920x1080:rate=30:duration={total_duration}",
        "-loop", "1", "-i", str(webcam_png),
        "-i", str(full_audio),
        "-filter_complex", gameplay_filter,
        "-map", "[outv]",
        "-map", "2:a",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-t", str(total_duration),
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        str(out_path)
    ]
    subprocess.run(cmd_video, capture_output=True, check=True)

    # Generate accompanying mock YouTube heatmap JSON file
    heatmap_file = out_path.with_name(f"{out_path.stem}_heatmap.json")
    heatmap_data = []
    for sec in range(0, int(total_duration), 2):
        # 3 peaks at 28s, 64s, 95s
        v1 = np.exp(-((sec - 28) ** 2) / 40.0) * 0.95
        v2 = np.exp(-((sec - 64) ** 2) / 35.0) * 0.88
        v3 = np.exp(-((sec - 95) ** 2) / 45.0) * 1.00
        val = float(max(0.08, v1 + v2 + v3))
        heatmap_data.append({
            "start_time": float(sec),
            "end_time": float(sec + 2),
            "value": round(val, 3),
        })

    with open(heatmap_file, "w") as f:
        json.dump(heatmap_data, f, indent=2)

    # Generate accompanying mock live chat replay JSON file
    chat_file = out_path.with_name(f"{out_path.stem}_chat.json")
    chat_messages = []
    base_users = ["xX_Sniper_Xx", "GamerGirl99", "PogChamp123", "TwitchViewer", "KekwMaster", "SpeedyGamer"]

    # Background chat
    for t in range(0, int(total_duration), 4):
        chat_messages.append({"time_offset": float(t), "author": base_users[t % len(base_users)], "text": "gg guys"})

    # Peak 1 chat (25-35s)
    for t in [25.5, 26.2, 27.0, 27.8, 28.5, 29.1, 29.8, 31.0, 32.5]:
        chat_messages.append({"time_offset": t, "author": "HypeBot", "text": "OMG NO WAY KEKW POGGERS!!!"})
        chat_messages.append({"time_offset": t + 0.3, "author": "ChatLover", "text": "HE IS CHEATING LMAO"})

    # Peak 2 chat (60-70s)
    for t in [61.0, 62.5, 63.8, 64.5, 65.2, 66.0, 67.5]:
        chat_messages.append({"time_offset": t, "author": "Viewer99", "text": "HAHAHAHA DID HE JUMP?! WTF"})

    # Peak 3 chat (90-102s)
    for t in [92.0, 93.5, 94.2, 95.0, 95.8, 96.5, 97.2, 98.0, 99.5]:
        chat_messages.append({"time_offset": t, "author": "ModGod", "text": "CLIP THAT RIGHT NOW!!! 🔥🔥🔥 W W W"})
        chat_messages.append({"time_offset": t + 0.2, "author": "Gamer1", "text": "LETS GOOOOOOOOOO 1V4 ACE!!!"})

    chat_messages.sort(key=lambda m: m["time_offset"])
    with open(chat_file, "w") as f:
        json.dump(chat_messages, f, indent=2)

    # Cleanup temp audio files
    for p in temp_dir.glob("*"):
        try:
            p.unlink()
        except Exception:
            pass
    try:
        temp_dir.rmdir()
    except Exception:
        pass

    print(f"Sample stream generated successfully at: {out_path}")
    print(f"Accompanied by heatmap: {heatmap_file}")
    print(f"Accompanied by chat: {chat_file}")
    return str(out_path)


if __name__ == "__main__":
    generate_sample_stream()
