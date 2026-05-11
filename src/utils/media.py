from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path


def safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    stem = stem.strip().lower()
    stem = re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem or "video"


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Please install FFmpeg and add it to PATH.")

    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found. Please install FFmpeg/FFprobe and add them to PATH.")


def run_command(cmd: list[str]) -> None:
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n\nSTDERR:\n"
            + result.stderr
        )


def get_video_duration_seconds(video_path: str | Path) -> float:
    check_ffmpeg()

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )

    data = json.loads(result.stdout)
    duration = float(data["format"]["duration"])
    return duration