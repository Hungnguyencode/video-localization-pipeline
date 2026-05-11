from __future__ import annotations

from pathlib import Path

from src.utils.config import ensure_dir
from src.utils.media import check_ffmpeg, run_command


class AudioExtractor:
    def __init__(self, output_dir: str, sample_rate: int = 16000, channels: int = 1):
        self.output_dir = ensure_dir(output_dir)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)

    def extract(self, video_path: str | Path, video_stem: str) -> str:
        check_ffmpeg()

        video_path = Path(video_path)
        output_path = self.output_dir / f"{video_stem}.wav"

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            str(self.channels),
            str(output_path),
        ]

        run_command(cmd)

        if not output_path.exists():
            raise RuntimeError(f"Audio extraction failed: {output_path}")

        return str(output_path)