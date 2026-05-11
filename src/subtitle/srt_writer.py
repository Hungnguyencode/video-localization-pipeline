from __future__ import annotations

import textwrap
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import srt


class SubtitleWriter:
    def write_source_srt(
        self,
        segments: List[Dict],
        output_path: str | Path,
    ) -> str:
        subtitles = []

        for idx, item in enumerate(segments, start=1):
            subtitles.append(
                srt.Subtitle(
                    index=idx,
                    start=timedelta(seconds=float(item["start"])),
                    end=timedelta(seconds=float(item["end"])),
                    content=(item.get("text") or "").strip(),
                )
            )

        return self._write(subtitles, output_path)

    def write_vietnamese_srt(
        self,
        segments: List[Dict],
        output_path: str | Path,
        max_chars_per_line: int | None = None,
    ) -> str:
        subtitles = []

        for idx, item in enumerate(segments, start=1):
            content = (
                item.get("subtitle_text")
                or item.get("vi_text")
                or ""
            ).strip()

            content = self._wrap_subtitle_text(
                content,
                max_chars_per_line=max_chars_per_line,
            )

            subtitles.append(
                srt.Subtitle(
                    index=idx,
                    start=timedelta(seconds=float(item["start"])),
                    end=timedelta(seconds=float(item["end"])),
                    content=content,
                )
            )

        return self._write(subtitles, output_path)

    def _wrap_subtitle_text(
        self,
        text: str,
        max_chars_per_line: int | None = None,
    ) -> str:
        text = (text or "").strip()

        if not text or not max_chars_per_line:
            return text

        max_chars_per_line = int(max_chars_per_line)

        if max_chars_per_line <= 0:
            return text

        wrapped_lines = textwrap.wrap(
            text,
            width=max_chars_per_line,
            break_long_words=False,
            break_on_hyphens=False,
        )

        return "\n".join(wrapped_lines)

    def _write(self, subtitles: List[srt.Subtitle], output_path: str | Path) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        content = srt.compose(subtitles)

        with path.open("w", encoding="utf-8") as f:
            f.write(content)

        return str(path)