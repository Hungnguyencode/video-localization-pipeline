from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Any

from src.utils.config import ensure_dir
from src.utils.media import safe_stem


ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


class LocalVideoIngestor:
    def __init__(self, input_dir: str):
        self.input_dir = ensure_dir(input_dir)

    def ingest(self, video_path: str | Path) -> Dict[str, Any]:
        source_path = Path(video_path)

        if not source_path.exists():
            raise FileNotFoundError(f"Video not found: {source_path}")

        suffix = source_path.suffix.lower()
        if suffix not in ALLOWED_VIDEO_EXTENSIONS:
            raise ValueError(
                f"Unsupported video extension: {suffix}. "
                f"Allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}"
            )

        safe_name = safe_stem(source_path.name)
        target_path = self.input_dir / f"{safe_name}{suffix}"

        if source_path.resolve() != target_path.resolve():
            shutil.copy2(source_path, target_path)

        return {
            "video_name": target_path.name,
            "video_stem": safe_name,
            "video_path": str(target_path),
            "source_path": str(source_path),
        }