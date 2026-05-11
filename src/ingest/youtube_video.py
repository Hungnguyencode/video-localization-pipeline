from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from yt_dlp import YoutubeDL

from src.utils.config import ensure_dir


class YouTubeVideoDownloader:
    """
    Tải 1 video YouTube về thư mục input.

    Mục tiêu:
    - Không batch playlist.
    - Ưu tiên mp4 để dễ xử lý bằng FFmpeg.
    - Giới hạn độ phân giải để phù hợp laptop 16GB RAM / RTX 2050.
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = ensure_dir(output_dir)

    def download(
        self,
        url: str,
        max_height: int = 720,
    ) -> Dict[str, Any]:
        url = (url or "").strip()

        if not url:
            raise ValueError("YouTube URL is empty.")

        max_height = int(max_height)

        output_template = str(
            Path(self.output_dir) / "%(title).80s_%(id)s.%(ext)s"
        )

        if max_height > 0:
            format_selector = (
                f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/"
                f"b[height<={max_height}][ext=mp4]/"
                f"best[height<={max_height}]/best"
            )
        else:
            format_selector = "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best"

        ydl_opts = {
            "format": format_selector,
            "outtmpl": output_template,
            "merge_output_format": "mp4",
            "noplaylist": True,
            "restrictfilenames": True,
            "quiet": False,
            "no_warnings": False,
        }

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        video_id = info.get("id")
        title = info.get("title") or "youtube_video"

        downloaded_path = self._find_downloaded_file(video_id)

        if downloaded_path is None:
            raise FileNotFoundError(
                f"Downloaded file not found for YouTube id: {video_id}"
            )

        return {
            "video_path": str(downloaded_path),
            "video_id": video_id,
            "title": title,
            "duration": info.get("duration"),
            "webpage_url": info.get("webpage_url") or url,
            "max_height": max_height,
        }

    def _find_downloaded_file(self, video_id: str | None) -> Path | None:
        if not video_id:
            return None

        candidates = list(Path(self.output_dir).glob(f"*{video_id}*"))

        if not candidates:
            return None

        mp4_candidates = [p for p in candidates if p.suffix.lower() == ".mp4"]

        if mp4_candidates:
            return max(mp4_candidates, key=lambda p: p.stat().st_mtime)

        return max(candidates, key=lambda p: p.stat().st_mtime)