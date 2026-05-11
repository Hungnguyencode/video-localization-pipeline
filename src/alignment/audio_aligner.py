from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from pydub import AudioSegment, effects

from src.utils.config import ensure_dir


class AudioAligner:
    def __init__(
        self,
        output_dir: str,
        max_speedup: float = 1.45,
        trim_if_too_long: bool = False,
        silence_padding_ms: int = 80,
        min_gap_ms: int = 60,
    ):
        self.output_dir = ensure_dir(output_dir)
        self.max_speedup = float(max_speedup)
        self.trim_if_too_long = bool(trim_if_too_long)
        self.silence_padding_ms = int(silence_padding_ms)
        self.min_gap_ms = int(min_gap_ms)

    def _fit_audio_to_duration(
        self,
        audio: AudioSegment,
        target_duration_ms: int,
    ) -> AudioSegment:
        """
        Cố gắng làm audio vừa với timestamp gốc, nhưng KHÔNG cắt cụt nội dung
        trừ khi trim_if_too_long=True.

        Với dubbing, mất vài trăm ms lệch thời gian còn chấp nhận được,
        nhưng mất chữ cuối câu thì rất tệ.
        """
        if target_duration_ms <= 0:
            return audio

        current_ms = len(audio)

        if current_ms <= target_duration_ms:
            return audio

        required_speed = current_ms / target_duration_ms

        # Nếu chỉ cần tăng tốc vừa phải thì tăng tốc.
        if required_speed <= self.max_speedup:
            try:
                sped_up = effects.speedup(
                    audio,
                    playback_speed=required_speed,
                    chunk_size=80,
                    crossfade=20,
                )
                return sped_up
            except Exception:
                return audio

        # Bản cũ cắt audio ở đây. Bản mới mặc định KHÔNG cắt.
        if self.trim_if_too_long:
            return audio[:target_duration_ms]

        return audio

    def _prepare_segments(
        self,
        segments: List[Dict],
    ) -> Tuple[List[Tuple[int, AudioSegment, Dict]], int]:
        """
        Chuẩn bị các đoạn TTS để overlay.

        Nguyên tắc:
        - Ưu tiên bám timestamp gốc.
        - Nếu audio tiếng Việt dài hơn làm lấn sang câu sau,
          câu sau sẽ được đẩy lùi để tránh đè tiếng.
        """
        prepared: List[Tuple[int, AudioSegment, Dict]] = []
        cursor_ms = 0

        for item in segments:
            tts_audio_path = item.get("tts_audio_path")
            if not tts_audio_path:
                continue

            original_start_ms = int(float(item["start"]) * 1000)
            original_end_ms = int(float(item["end"]) * 1000)
            target_ms = max(
                1,
                original_end_ms - original_start_ms - self.silence_padding_ms,
            )

            segment_audio = AudioSegment.from_file(tts_audio_path)
            segment_audio = self._fit_audio_to_duration(segment_audio, target_ms)

            # Không cho câu sau đè câu trước.
            placement_ms = max(original_start_ms, cursor_ms)

            prepared.append((placement_ms, segment_audio, item))

            cursor_ms = placement_ms + len(segment_audio) + self.min_gap_ms

        return prepared, cursor_ms

    def build_dubbed_audio(
        self,
        segments: List[Dict],
        video_duration_sec: float,
        video_stem: str,
    ) -> str:
        prepared, final_cursor_ms = self._prepare_segments(segments)

        original_video_ms = int(video_duration_sec * 1000)

        # Nếu audio Việt dài hơn video một chút, cứ tạo audio dài hơn.
        # Khi ghép với video, phần vượt quá cuối video có thể bị cắt,
        # nhưng ít nhất các câu giữa video không bị cắt cụt.
        total_duration_ms = max(original_video_ms + 500, final_cursor_ms + 500)

        base = AudioSegment.silent(duration=total_duration_ms)

        for placement_ms, segment_audio, item in prepared:
            base = base.overlay(segment_audio, position=placement_ms)

        output_path = self.output_dir / f"{video_stem}_vi_dubbed.wav"
        base.export(output_path, format="wav")

        return str(output_path)