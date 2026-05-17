from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

from pydub import AudioSegment, effects, silence

from src.utils.config import ensure_dir


class AudioAligner:
    def __init__(
        self,
        output_dir: str,
        max_speedup: float = 1.65,
        trim_if_too_long: bool = False,
        silence_padding_ms: int = 30,
        min_gap_ms: int = 20,
    ):
        self.output_dir = ensure_dir(output_dir)
        self.max_speedup = float(os.getenv("AUDIO_ALIGN_MAX_SPEEDUP", max_speedup))
        self.trim_if_too_long = str(os.getenv("AUDIO_ALIGN_TRIM_IF_TOO_LONG", str(trim_if_too_long))).lower() in {"1", "true", "yes", "on"}
        self.silence_padding_ms = int(os.getenv("AUDIO_ALIGN_PADDING_MS", silence_padding_ms))
        self.min_gap_ms = int(os.getenv("AUDIO_ALIGN_MIN_GAP_MS", min_gap_ms))
        self.strict_timeline = str(os.getenv("AUDIO_ALIGN_STRICT_TIMELINE", "1")).lower() not in {"0", "false", "no", "off"}
        self.trim_silence = str(os.getenv("AUDIO_ALIGN_TRIM_SILENCE", "1")).lower() not in {"0", "false", "no", "off"}
        self.silence_thresh_db = float(os.getenv("AUDIO_ALIGN_SILENCE_THRESH_DB", "-45"))

    def _trim_edge_silence(self, audio: AudioSegment) -> AudioSegment:
        if not self.trim_silence or len(audio) <= 0:
            return audio
        ranges = silence.detect_nonsilent(audio, min_silence_len=80, silence_thresh=self.silence_thresh_db)
        if not ranges:
            return audio
        start = max(0, ranges[0][0] - 25)
        end = min(len(audio), ranges[-1][1] + 35)
        return audio[start:end]

    def _fit_audio_to_duration(self, audio: AudioSegment, target_duration_ms: int) -> AudioSegment:
        if target_duration_ms <= 0:
            return audio

        audio = self._trim_edge_silence(audio)
        current_ms = len(audio)
        if current_ms <= target_duration_ms:
            return audio

        required_speed = current_ms / target_duration_ms
        speed = min(required_speed, self.max_speedup)
        try:
            sped_up = effects.speedup(audio, playback_speed=speed, chunk_size=60, crossfade=15)
            audio = sped_up
        except Exception:
            pass

        # Không cắt chữ theo mặc định. Chỉ cắt nếu người dùng bật env AUDIO_ALIGN_TRIM_IF_TOO_LONG=1.
        if len(audio) > target_duration_ms and self.trim_if_too_long:
            keep_ms = max(200, target_duration_ms - 80)
            audio = audio[:keep_ms]

        return audio

    def _prepare_segments(self, segments: List[Dict]) -> Tuple[List[Tuple[int, AudioSegment, Dict]], int]:
        prepared: List[Tuple[int, AudioSegment, Dict]] = []
        cursor_ms = 0

        for idx, item in enumerate(segments, start=1):
            tts_audio_path = item.get("tts_audio_path")
            if not tts_audio_path:
                print(f"[ALIGN][WARN] Segment {idx} has no tts_audio_path -> skipped")
                continue

            original_start_ms = int(float(item["start"]) * 1000)
            original_end_ms = int(float(item["end"]) * 1000)
            target_ms = max(1, original_end_ms - original_start_ms - self.silence_padding_ms)

            segment_audio = AudioSegment.from_file(tts_audio_path)
            segment_audio = self._fit_audio_to_duration(segment_audio, target_ms)

            if self.strict_timeline:
                # Quan trọng: bám timestamp gốc, KHÔNG đẩy cả các segment sau đi xa.
                # Cách cũ dùng cursor_ms làm một segment dài kéo lệch timeline và tạo nhiều khoảng trống khó chịu.
                placement_ms = original_start_ms
            else:
                placement_ms = max(original_start_ms, cursor_ms)

            prepared.append((placement_ms, segment_audio, item))
            cursor_ms = max(cursor_ms, placement_ms + len(segment_audio) + self.min_gap_ms)

            print(
                f"[ALIGN] seg={item.get('id') or item.get('segment_id') or idx} "
                f"start={original_start_ms}ms place={placement_ms}ms "
                f"target={target_ms}ms audio={len(segment_audio)}ms"
            )

        return prepared, cursor_ms

    def build_dubbed_audio(self, segments: List[Dict], video_duration_sec: float, video_stem: str) -> str:
        prepared, final_cursor_ms = self._prepare_segments(segments)
        original_video_ms = int(video_duration_sec * 1000)
        total_duration_ms = max(original_video_ms + 500, final_cursor_ms + 500)
        base = AudioSegment.silent(duration=total_duration_ms)

        for placement_ms, segment_audio, item in prepared:
            base = base.overlay(segment_audio, position=placement_ms)

        output_path = self.output_dir / f"{video_stem}_vi_dubbed.wav"
        base.export(output_path, format="wav")
        print(f"[ALIGN] Export dubbed audio: {output_path} | duration={len(base)}ms | segments={len(prepared)}")
        return str(output_path)
