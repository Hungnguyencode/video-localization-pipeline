from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

from pydub import AudioSegment, effects, silence

from src.utils.config import ensure_dir


class AudioAligner:
    def __init__(
        self,
        output_dir: str,
        max_speedup: float = 1.65,
        trim_if_too_long: bool = True,
        silence_padding_ms: int = 30,
        min_gap_ms: int = 20,
        sample_rate: int = 48000,
    ):
        self.output_dir = ensure_dir(output_dir)

        # Tốc độ tối đa khi tự ép TTS cho vừa slot.
        # 1.65 nghĩa là cho phép tăng tốc tối đa 65%.
        self.max_speedup = float(os.getenv("AUDIO_ALIGN_MAX_SPEEDUP", max_speedup))

        # Để tránh mất câu cuối / đè timeline, nên bật trim fallback.
        self.trim_if_too_long = str(
            os.getenv("AUDIO_ALIGN_TRIM_IF_TOO_LONG", str(trim_if_too_long))
        ).lower() in {"1", "true", "yes", "on"}

        self.silence_padding_ms = int(os.getenv("AUDIO_ALIGN_PADDING_MS", silence_padding_ms))
        self.min_gap_ms = int(os.getenv("AUDIO_ALIGN_MIN_GAP_MS", min_gap_ms))
        self.sample_rate = int(os.getenv("AUDIO_ALIGN_SAMPLE_RATE", sample_rate))

        # Bám timestamp gốc, không để 1 segment dài kéo lệch các segment sau.
        self.strict_timeline = str(
            os.getenv("AUDIO_ALIGN_STRICT_TIMELINE", "1")
        ).lower() not in {"0", "false", "no", "off"}

        self.trim_silence = str(
            os.getenv("AUDIO_ALIGN_TRIM_SILENCE", "1")
        ).lower() not in {"0", "false", "no", "off"}

        self.silence_thresh_db = float(os.getenv("AUDIO_ALIGN_SILENCE_THRESH_DB", "-45"))

    def _trim_edge_silence(self, audio: AudioSegment) -> AudioSegment:
        """
        Cắt im lặng ở đầu/cuối file TTS.
        Việc này giúp segment ngắn hơn mà không phải sửa text thủ công.
        """
        if not self.trim_silence or len(audio) <= 0:
            return audio

        ranges = silence.detect_nonsilent(
            audio,
            min_silence_len=80,
            silence_thresh=self.silence_thresh_db,
        )

        if not ranges:
            return audio

        start = max(0, ranges[0][0] - 25)
        end = min(len(audio), ranges[-1][1] + 35)

        return audio[start:end]

    def _ensure_audio_duration(self, audio: AudioSegment, target_ms: int) -> AudioSegment:
        """
        Đảm bảo audio output đúng độ dài video.
        Thiếu thì thêm silence, dư thì cắt.
        """
        target_ms = int(target_ms)

        if len(audio) < target_ms:
            pad = AudioSegment.silent(
                duration=target_ms - len(audio),
                frame_rate=audio.frame_rate,
            ).set_channels(audio.channels)

            return audio + pad

        if len(audio) > target_ms:
            return audio[:target_ms]

        return audio

    def _build_atempo_filter(self, speed: float) -> str:
        """
        Tạo filter atempo cho ffmpeg.
        ffmpeg atempo chạy ổn trong khoảng 0.5 đến 2.0.
        Nếu speed vượt khoảng này thì tách thành nhiều filter.
        """
        speed = float(speed)
        factors = []

        while speed > 2.0:
            factors.append(2.0)
            speed /= 2.0

        while speed < 0.5:
            factors.append(0.5)
            speed /= 0.5

        factors.append(speed)

        return ",".join(f"atempo={factor:.6f}" for factor in factors)

    def _change_audio_tempo(self, audio: AudioSegment, speed: float) -> AudioSegment:
        """
        Tăng tốc audio bằng ffmpeg atempo để giữ giọng tự nhiên hơn.
        Nếu ffmpeg lỗi thì fallback về pydub effects.speedup.
        """
        speed = float(speed)

        if abs(speed - 1.0) < 0.01:
            return audio

        if shutil.which("ffmpeg"):
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpdir = Path(tmpdir)
                    input_wav = tmpdir / "input.wav"
                    output_wav = tmpdir / "output.wav"

                    audio.export(input_wav, format="wav")

                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(input_wav),
                        "-filter:a",
                        self._build_atempo_filter(speed),
                        "-vn",
                        str(output_wav),
                    ]

                    subprocess.run(
                        cmd,
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

                    fixed = AudioSegment.from_file(output_wav)
                    fixed = fixed.set_frame_rate(self.sample_rate).set_channels(2)

                    return fixed
            except Exception as exc:
                print(f"[ALIGN][WARN] ffmpeg atempo failed, fallback pydub: {exc}")

        try:
            return effects.speedup(
                audio,
                playback_speed=speed,
                chunk_size=60,
                crossfade=15,
            )
        except Exception as exc:
            print(f"[ALIGN][WARN] pydub speedup failed: {exc}")
            return audio

    def _fit_audio_to_duration(
        self,
        audio: AudioSegment,
        target_duration_ms: int,
        segment_id: str | int = "?",
    ) -> AudioSegment:
        """
        Ép TTS vừa thời lượng slot.

        Đây là phần sửa lỗi chính:
        - Nếu TTS dài hơn slot, tự tăng tốc.
        - Nếu tăng tốc xong vẫn dài, cắt nhẹ phần dư với fade_out.
        - Nhờ vậy segment cuối không bị rơi ra khỏi timeline video.
        """
        target_duration_ms = int(target_duration_ms)

        if target_duration_ms <= 0:
            return AudioSegment.silent(duration=1, frame_rate=self.sample_rate).set_channels(2)

        audio = audio.set_frame_rate(self.sample_rate).set_channels(2)
        audio = self._trim_edge_silence(audio)

        current_ms = len(audio)

        if current_ms <= target_duration_ms:
            return audio

        required_speed = current_ms / target_duration_ms
        speed = min(required_speed, self.max_speedup)

        before_ms = len(audio)
        audio = self._change_audio_tempo(audio, speed)
        after_speed_ms = len(audio)

        print(
            f"[ALIGN][AUTO-FIT] Segment {segment_id}: "
            f"{before_ms}ms -> {after_speed_ms}ms, "
            f"target={target_duration_ms}ms, speed={speed:.3f}"
        )

        # Sau khi speedup vẫn dài thì cắt phần dư.
        # Đây là fallback để không bị mất câu cuối ở bước mux video.
        if len(audio) > target_duration_ms:
            if self.trim_if_too_long:
                fade_ms = min(80, max(20, target_duration_ms // 4))
                audio = audio[:target_duration_ms].fade_out(fade_ms)

                print(
                    f"[ALIGN][TRIM] Segment {segment_id}: "
                    f"trim to {target_duration_ms}ms"
                )
            else:
                print(
                    f"[ALIGN][WARN] Segment {segment_id} still too long: "
                    f"{len(audio)}ms > {target_duration_ms}ms"
                )

        return audio

    def _prepare_segments(
        self,
        segments: List[Dict],
        video_duration_sec: float,
    ) -> Tuple[List[Tuple[int, AudioSegment, Dict]], int]:
        prepared: List[Tuple[int, AudioSegment, Dict]] = []

        video_duration_ms = int(float(video_duration_sec) * 1000)
        cursor_ms = 0

        # Sắp xếp theo start để tính slot chính xác.
        segments = sorted(
            segments,
            key=lambda item: float(item.get("start", 0)),
        )

        for idx, item in enumerate(segments, start=1):
            segment_id = item.get("id") or item.get("segment_id") or idx
            tts_audio_path = item.get("tts_audio_path")

            if not tts_audio_path:
                print(f"[ALIGN][WARN] Segment {segment_id} has no tts_audio_path -> skipped")
                continue

            tts_audio_path = Path(tts_audio_path)

            if not tts_audio_path.exists():
                print(f"[ALIGN][WARN] Segment {segment_id} audio not found: {tts_audio_path}")
                continue

            original_start_ms = int(float(item["start"]) * 1000)
            original_end_ms = int(float(item["end"]) * 1000)

            if original_start_ms >= video_duration_ms:
                print(
                    f"[ALIGN][SKIP] Segment {segment_id}: "
                    f"start={original_start_ms}ms vượt video={video_duration_ms}ms"
                )
                continue

            is_last_segment = idx == len(segments)

            # Tính hard_end_ms:
            # - Segment giữa: không cho đè sang segment sau.
            # - Segment cuối: cho dùng tới sát cuối video.
            if is_last_segment:
                hard_end_ms = video_duration_ms - 30
            else:
                next_start_ms = int(float(segments[idx].get("start", item["end"])) * 1000)
                hard_end_ms = min(original_end_ms, next_start_ms - self.min_gap_ms)

            # Nếu hard_end lỗi hoặc quá sát start thì fallback về end gốc.
            if hard_end_ms <= original_start_ms:
                hard_end_ms = min(original_end_ms, video_duration_ms - 30)

            # Với segment cuối, không trừ padding nhiều nữa,
            # vì đoạn cuối cần tận dụng sát hết video để không mất câu.
            if is_last_segment:
                target_ms = max(1, hard_end_ms - original_start_ms)
            else:
                target_ms = max(1, hard_end_ms - original_start_ms - self.silence_padding_ms)

            if target_ms <= 80:
                print(
                    f"[ALIGN][SKIP] Segment {segment_id}: "
                    f"slot quá ngắn target={target_ms}ms"
                )
                continue

            segment_audio = AudioSegment.from_file(tts_audio_path)
            segment_audio = segment_audio.set_frame_rate(self.sample_rate).set_channels(2)

            segment_audio = self._fit_audio_to_duration(
                segment_audio,
                target_duration_ms=target_ms,
                segment_id=segment_id,
            )

            if self.strict_timeline:
                placement_ms = original_start_ms
            else:
                placement_ms = max(original_start_ms, cursor_ms)

            # Chặn tuyệt đối audio rơi khỏi cuối video.
            available_until_video_end = video_duration_ms - placement_ms - 20

            if available_until_video_end <= 80:
                print(
                    f"[ALIGN][SKIP] Segment {segment_id}: "
                    f"không còn đủ chỗ ở cuối video"
                )
                continue

            if len(segment_audio) > available_until_video_end:
                segment_audio = self._fit_audio_to_duration(
                    segment_audio,
                    target_duration_ms=available_until_video_end,
                    segment_id=segment_id,
                )

            prepared.append((placement_ms, segment_audio, item))
            cursor_ms = max(cursor_ms, placement_ms + len(segment_audio) + self.min_gap_ms)

            print(
                f"[ALIGN] seg={segment_id} "
                f"start={original_start_ms}ms place={placement_ms}ms "
                f"target={target_ms}ms audio={len(segment_audio)}ms "
                f"video={video_duration_ms}ms"
            )

        return prepared, cursor_ms

    def build_dubbed_audio(
        self,
        segments: List[Dict],
        video_duration_sec: float,
        video_stem: str,
    ) -> str:
        """
        Build audio dub tiếng Việt đúng bằng duration video.

        Quan trọng:
        - Không tạo audio dài hơn video +500ms nữa.
        - Output audio đúng bằng video_duration.
        - Segment cuối nếu dài quá sẽ tự fit vào phần còn lại.
        """
        original_video_ms = int(float(video_duration_sec) * 1000)

        prepared, final_cursor_ms = self._prepare_segments(
            segments=segments,
            video_duration_sec=video_duration_sec,
        )

        if original_video_ms <= 0:
            total_duration_ms = final_cursor_ms
        else:
            total_duration_ms = original_video_ms

        base = AudioSegment.silent(
            duration=total_duration_ms,
            frame_rate=self.sample_rate,
        ).set_channels(2)

        for placement_ms, segment_audio, item in prepared:
            segment_id = item.get("id") or item.get("segment_id") or "?"
            available_ms = total_duration_ms - placement_ms

            if available_ms <= 0:
                print(f"[ALIGN][SKIP] Segment {segment_id}: placement beyond audio duration")
                continue

            if len(segment_audio) > available_ms:
                segment_audio = segment_audio[:available_ms].fade_out(60)

            base = base.overlay(segment_audio, position=placement_ms)

        base = self._ensure_audio_duration(base, total_duration_ms)
        base = base.fade_out(60)

        output_path = self.output_dir / f"{video_stem}_vi_dubbed.wav"
        base.export(output_path, format="wav")

        print(
            f"[ALIGN] Export dubbed audio: {output_path} "
            f"| duration={len(base)}ms "
            f"| video={total_duration_ms}ms "
            f"| segments={len(prepared)}"
        )

        return str(output_path)