"""
Viral moment analyzer module for AutoClipper.
Combines audio spikes (RMS / dB transitions), YouTube 'most replayed' heatmap peaks,
and live chat velocity/hype sentiment into a composite viral score to find top 3 viral moments.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

logger = logging.getLogger("autoclipper.analyzer")


class ViralMomentAnalyzer:
    """Analyzes audio signals, heatmaps, and chat density to find peak viral moments."""

    def __init__(
        self,
        clip_duration: float = 45.0,
        pre_peak_lead: float = 15.0,
        min_clip_distance: float = 50.0,
        sample_rate: int = 16000,
    ):
        self.clip_duration = clip_duration
        self.pre_peak_lead = pre_peak_lead
        self.min_clip_distance = min_clip_distance
        self.sample_rate = sample_rate

    def analyze(
        self,
        audio_wav_path: str,
        video_duration: float,
        heatmap_data: Optional[List[Dict[str, Any]]] = None,
        chat_messages: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Executes full viral moment detection pipeline.
        Returns top_k viral clips and complete timeline series for UI visualization.
        """
        # 1. Audio energy analysis
        time_grid, audio_scores, raw_rms = self._analyze_audio(audio_wav_path, video_duration)

        # 2. Replay heatmap analysis
        replay_scores = self._analyze_heatmap(heatmap_data, time_grid, video_duration)

        # 3. Chat velocity & hype analysis
        chat_scores = self._analyze_chat(chat_messages, time_grid, video_duration)

        # 4. Composite fusion scoring
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

        composite = (
            w_audio * audio_scores +
            w_replay * replay_scores +
            w_chat * chat_scores
        )

        # Smooth to remove micro-spikes (e.g. 2-second Gaussian window)
        smoothed = gaussian_filter1d(composite, sigma=2.0)
        # Normalize to 0 - 100 scale
        if np.max(smoothed) > 0:
            viral_curve = (smoothed / np.max(smoothed)) * 100.0
        else:
            viral_curve = np.zeros_like(smoothed)

        # 5. Peak picking with non-maximum suppression / distance enforcement
        step_sec = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 1.0
        # Dynamically adapt distance to video duration so top_k distinct peaks are always available
        adapted_min_dist = min(self.min_clip_distance, max(12.0, video_duration / float(top_k + 1.2)))
        min_distance_steps = max(1, int(adapted_min_dist / step_sec))

        peak_indices, properties = find_peaks(
            viral_curve,
            distance=min_distance_steps,
            prominence=4.0,
            height=12.0,
        )

        # If not enough peaks found with high prominence, relax prominence
        if len(peak_indices) < top_k:
            peak_indices, _ = find_peaks(
                viral_curve,
                distance=max(1, min_distance_steps // 2),
                height=8.0,
            )

        # If still not enough, pick top N highest points with distance separation
        if len(peak_indices) < top_k:
            sorted_indices = np.argsort(viral_curve)[::-1]
            chosen = []
            for idx in sorted_indices:
                if all(abs(idx - c) >= max(1, min_distance_steps // 2) for c in chosen):
                    chosen.append(idx)
                    if len(chosen) >= top_k:
                        break
            peak_indices = np.array(chosen)

        # Sort selected peaks by score descending
        if len(peak_indices) > 0:
            peak_scores = viral_curve[peak_indices]
            top_order = np.argsort(peak_scores)[::-1][:top_k]
            selected_peaks = peak_indices[top_order]
        else:
            # Fallback for completely flat video: spread clips evenly
            t1 = min(video_duration * 0.25, 30.0)
            t2 = min(video_duration * 0.50, 90.0)
            t3 = min(video_duration * 0.75, 150.0)
            fallback_times = [t1, t2, t3][:top_k]
            selected_peaks = [int(t / step_sec) for t in fallback_times if t / step_sec < len(time_grid)]

        # 6. Format clips
        clips = []
        for rank, peak_idx in enumerate(selected_peaks, start=1):
            peak_time = float(time_grid[peak_idx])
            peak_score = float(viral_curve[peak_idx])

            # Calculate start and end with lead-in anticipation
            start_time = max(0.0, peak_time - self.pre_peak_lead)
            end_time = min(video_duration, start_time + self.clip_duration)
            # Adjust if near end of video
            if end_time - start_time < self.clip_duration and start_time > 0:
                start_time = max(0.0, end_time - self.clip_duration)

            # Determine dominant reason for viral moment
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
                "viral_score": round(peak_score, 1),
                "audio_score": round(a_val * 100, 1),
                "replay_score": round(r_val * 100, 1),
                "chat_score": round(c_val * 100, 1),
                "primary_reason": " + ".join(reasons),
                "formatted_time": f"{int(start_time // 60):02d}:{int(start_time % 60):02d} - {int(end_time // 60):02d}:{int(end_time % 60):02d}",
            })

        # Subsample timeline for frontend charts (e.g. max 300 data points)
        stride = max(1, len(time_grid) // 300)
        timeline = [
            {
                "time": round(float(time_grid[i]), 1),
                "viral_score": round(float(viral_curve[i]), 1),
                "audio": round(float(audio_scores[i] * 100), 1),
                "replay": round(float(replay_scores[i] * 100), 1),
                "chat": round(float(chat_scores[i] * 100), 1),
            }
            for i in range(0, len(time_grid), stride)
        ]

        return {
            "top_clips": clips,
            "weights": {"audio": w_audio, "replay": w_replay, "chat": w_chat},
            "timeline": timeline,
            "video_duration": video_duration,
        }

    def _analyze_audio(
        self,
        audio_wav_path: str,
        video_duration: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates RMS audio energy and sudden decibel spikes over sliding windows."""
        sr, data = wavfile.read(audio_wav_path)
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Normalize int16/int32 to float in [-1.0, 1.0]
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        else:
            data = data.astype(np.float32)

        total_audio_sec = len(data) / float(sr)
        duration = min(video_duration, total_audio_sec) if video_duration > 0 else total_audio_sec

        # 0.5s window with 0.5s step
        window_size = int(sr * 1.0)
        step_size = int(sr * 0.5)

        num_steps = max(1, int((len(data) - window_size) / step_size))
        times = np.zeros(num_steps, dtype=np.float32)
        rms_vals = np.zeros(num_steps, dtype=np.float32)

        for i in range(num_steps):
            start = i * step_size
            chunk = data[start:start + window_size]
            rms = np.sqrt(np.mean(chunk ** 2) + 1e-9)
            rms_vals[i] = rms
            times[i] = (start + window_size / 2) / float(sr)

        # Convert to dB relative to 1.0
        db_vals = 20.0 * np.log10(rms_vals + 1e-6)

        # Compute rolling baseline (30s window) to isolate spikes from ambient noise
        baseline_steps = max(1, int(30.0 / 0.5))
        rolling_baseline = np.zeros_like(db_vals)
        for i in range(len(db_vals)):
            w_start = max(0, i - baseline_steps // 2)
            w_end = min(len(db_vals), i + baseline_steps // 2)
            rolling_baseline[i] = np.median(db_vals[w_start:w_end])

        # Delta dB: sudden loud reaction above background
        delta_db = np.maximum(0.0, db_vals - rolling_baseline)

        # Absolute energy normalization
        norm_rms = (rms_vals - np.min(rms_vals)) / (np.ptp(rms_vals) + 1e-6)
        norm_delta = (delta_db - np.min(delta_db)) / (np.ptp(delta_db) + 1e-6)

        # 60% weight on sudden loud spikes (screams/laughs), 40% on overall intensity
        audio_score = 0.60 * norm_delta + 0.40 * norm_rms
        audio_score = np.clip(audio_score, 0.0, 1.0)

        return times, audio_score, rms_vals

    def _analyze_heatmap(
        self,
        heatmap_data: Optional[List[Dict[str, Any]]],
        time_grid: np.ndarray,
        duration: float,
    ) -> np.ndarray:
        """Interpolates YouTube 'most replayed' heatmap values onto time_grid."""
        scores = np.zeros_like(time_grid)
        if not heatmap_data:
            return scores

        try:
            hp_times = []
            hp_vals = []
            for pt in heatmap_data:
                mid = (pt.get("start_time", 0.0) + pt.get("end_time", 0.0)) / 2.0
                val = float(pt.get("value", 0.0))
                hp_times.append(mid)
                hp_vals.append(val)

            if len(hp_times) > 1:
                # Linear interpolation onto time_grid
                scores = np.interp(time_grid, hp_times, hp_vals, left=0.0, right=0.0)
                # Normalize to 0 - 1
                max_v = np.max(scores)
                if max_v > 0:
                    scores = scores / max_v
        except Exception as e:
            logger.warning(f"Error parsing heatmap: {e}")

        return np.clip(scores, 0.0, 1.0)

    def _analyze_chat(
        self,
        chat_messages: Optional[List[Dict[str, Any]]],
        time_grid: np.ndarray,
        duration: float,
    ) -> np.ndarray:
        """Calculates live chat message velocity and hype keyword intensity over time."""
        scores = np.zeros_like(time_grid)
        if not chat_messages:
            return scores

        hype_keywords = {
            "w", "l", "omg", "pog", "poggers", "kekw", "lol", "lmao",
            "wtf", "hacker", "clutch", "clip", "no way", "bro", "sheesh",
            "dead", "gg", "what", "insane", "aimbot"
        }

        # Bin chat messages into 10s buckets
        bucket_size = 10.0
        num_buckets = max(1, int(duration / bucket_size) + 1)
        velocity = np.zeros(num_buckets, dtype=np.float32)
        hype_weight = np.zeros(num_buckets, dtype=np.float32)

        for msg in chat_messages:
            t = float(msg.get("time_offset", 0.0))
            if 0 <= t < duration:
                b_idx = int(t / bucket_size)
                if b_idx < num_buckets:
                    velocity[b_idx] += 1.0
                    txt = str(msg.get("text", "")).lower()
                    # Check for caps / exclamation
                    raw_txt = str(msg.get("text", ""))
                    caps_bonus = 1.5 if raw_txt.isupper() and len(raw_txt) > 3 else 1.0
                    excl_bonus = 1.3 if "!" in raw_txt else 1.0

                    # Hype keyword check
                    words = set(raw_txt.lower().split())
                    matched = len(words.intersection(hype_keywords))
                    hype_weight[b_idx] += (1.0 + matched * 0.8) * caps_bonus * excl_bonus

        # Map buckets onto time_grid
        bucket_times = np.arange(num_buckets) * bucket_size + bucket_size / 2.0
        combined = velocity * (1.0 + (hype_weight / (velocity + 1e-6)))

        if len(bucket_times) > 1 and np.max(combined) > 0:
            scores = np.interp(time_grid, bucket_times, combined, left=0.0, right=0.0)
            scores = scores / np.max(scores)

        return np.clip(scores, 0.0, 1.0)
