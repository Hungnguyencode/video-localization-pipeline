from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict

from src.utils.config import ensure_dir
from src.utils.media import check_ffmpeg, run_command


class VideoRenderer:
    def __init__(
        self,
        output_dir: str,
        output_video_suffix: str = "_vi_dubbed",
    ):
        self.output_dir = ensure_dir(output_dir)
        self.output_video_suffix = output_video_suffix

    def _has_audio_stream(self, video_path: str | Path) -> bool:
        check_ffmpeg()

        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
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
        )

        if result.returncode != 0:
            return False

        try:
            data = json.loads(result.stdout)
            return len(data.get("streams", [])) > 0
        except Exception:
            return False

    def _escape_subtitle_path_for_ffmpeg(self, subtitle_path: str | Path) -> str:
        path = Path(subtitle_path).resolve().as_posix()
        path = path.replace(":", r"\:")
        path = path.replace("'", r"\'")
        return path

    def _build_subtitle_filter(
        self,
        subtitle_path: str | Path,
        subtitle_style: Dict[str, Any] | None = None,
    ) -> str:
        escaped_subtitle = self._escape_subtitle_path_for_ffmpeg(subtitle_path)

        style = {
            "font_name": "Arial",
            "font_size": 26,
            "primary_colour": "&H00FFFFFF",
            "outline_colour": "&H00000000",
            "border_style": 1,
            "outline": 2,
            "shadow": 1,
            "alignment": 2,
            "margin_v": 45,
        }

        if subtitle_style:
            style.update(subtitle_style)

        force_style = (
            f"FontName={style['font_name']},"
            f"FontSize={style['font_size']},"
            f"PrimaryColour={style['primary_colour']},"
            f"OutlineColour={style['outline_colour']},"
            f"BorderStyle={style['border_style']},"
            f"Outline={style['outline']},"
            f"Shadow={style['shadow']},"
            f"Alignment={style['alignment']},"
            f"MarginV={style['margin_v']}"
        )

        return f"subtitles='{escaped_subtitle}':force_style='{force_style}'"

    def _build_output_path(
        self,
        video_stem: str,
        audio_mode: str,
        burn_subtitle: bool,
    ) -> Path:
        mode_suffix = {
            "replace": "replace",
            "mix_low_original": "mix",
            "subtitle_only": "subtitle_only",
        }.get(audio_mode, "replace")

        subtitle_suffix = "sub" if burn_subtitle else "nosub"

        return self.output_dir / (
            f"{video_stem}{self.output_video_suffix}_{mode_suffix}_{subtitle_suffix}.mp4"
        )

    def render_localized_video(
        self,
        input_video_path: str | Path,
        dubbed_audio_path: str | Path | None,
        video_stem: str,
        subtitle_path: str | Path | None = None,
        audio_mode: str = "replace",
        original_audio_volume_db: float = -18.0,
        burn_subtitle: bool = False,
        subtitle_style: Dict[str, Any] | None = None,
    ) -> str:
        """
        audio_mode:
        - replace: thay audio gốc bằng audio tiếng Việt.
        - mix_low_original: giữ audio gốc nhỏ lại và trộn với audio tiếng Việt.
        - subtitle_only: giữ nguyên audio gốc, chỉ chèn phụ đề tiếng Việt.
        """
        check_ffmpeg()

        input_video_path = Path(input_video_path)

        if not input_video_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_video_path}")

        if audio_mode not in {"replace", "mix_low_original", "subtitle_only"}:
            raise ValueError(
                "Unsupported audio_mode. Use 'replace', 'mix_low_original', or 'subtitle_only'."
            )

        if burn_subtitle:
            if subtitle_path is None:
                raise ValueError("subtitle_path is required when burn_subtitle=True")

            subtitle_path = Path(subtitle_path)

            if not subtitle_path.exists():
                raise FileNotFoundError(f"Subtitle not found: {subtitle_path}")

        if audio_mode in {"replace", "mix_low_original"}:
            if dubbed_audio_path is None:
                raise ValueError("dubbed_audio_path is required for dubbing modes.")

            dubbed_audio_path = Path(dubbed_audio_path)

            if not dubbed_audio_path.exists():
                raise FileNotFoundError(f"Dubbed audio not found: {dubbed_audio_path}")

        if audio_mode == "subtitle_only":
            burn_subtitle = True

            output_path = self._build_output_path(
                video_stem=video_stem,
                audio_mode=audio_mode,
                burn_subtitle=burn_subtitle,
            )

            cmd = self._build_subtitle_only_cmd(
                input_video_path=input_video_path,
                output_path=output_path,
                subtitle_path=subtitle_path,
                subtitle_style=subtitle_style,
            )

            run_command(cmd)

            if not output_path.exists():
                raise RuntimeError(f"Video rendering failed: {output_path}")

            return str(output_path)

        output_path = self._build_output_path(
            video_stem=video_stem,
            audio_mode=audio_mode,
            burn_subtitle=burn_subtitle,
        )

        if audio_mode == "mix_low_original" and not self._has_audio_stream(input_video_path):
            print("[RENDER] Input video has no audio stream. Fallback to replace mode.")
            audio_mode = "replace"

        if audio_mode == "replace":
            cmd = self._build_replace_audio_cmd(
                input_video_path=input_video_path,
                dubbed_audio_path=dubbed_audio_path,
                output_path=output_path,
                subtitle_path=subtitle_path,
                burn_subtitle=burn_subtitle,
                subtitle_style=subtitle_style,
            )
        else:
            cmd = self._build_mix_audio_cmd(
                input_video_path=input_video_path,
                dubbed_audio_path=dubbed_audio_path,
                output_path=output_path,
                subtitle_path=subtitle_path,
                original_audio_volume_db=original_audio_volume_db,
                burn_subtitle=burn_subtitle,
                subtitle_style=subtitle_style,
            )

        run_command(cmd)

        if not output_path.exists():
            raise RuntimeError(f"Video rendering failed: {output_path}")

        return str(output_path)

    def _build_subtitle_only_cmd(
        self,
        input_video_path: Path,
        output_path: Path,
        subtitle_path: str | Path,
        subtitle_style: Dict[str, Any] | None = None,
    ) -> list[str]:
        subtitle_filter = self._build_subtitle_filter(
            subtitle_path=subtitle_path,
            subtitle_style=subtitle_style,
        )

        return [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-vf",
            subtitle_filter,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "copy",
            str(output_path),
        ]

    def _build_replace_audio_cmd(
        self,
        input_video_path: Path,
        dubbed_audio_path: Path,
        output_path: Path,
        subtitle_path: str | Path | None,
        burn_subtitle: bool,
        subtitle_style: Dict[str, Any] | None = None,
    ) -> list[str]:
        if burn_subtitle:
            subtitle_filter = self._build_subtitle_filter(
                subtitle_path=subtitle_path,
                subtitle_style=subtitle_style,
            )

            return [
                "ffmpeg",
                "-y",
                "-i",
                str(input_video_path),
                "-i",
                str(dubbed_audio_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-vf",
                subtitle_filter,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-shortest",
                str(output_path),
            ]

        return [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video_path),
            "-i",
            str(dubbed_audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]

    def _build_mix_audio_cmd(
        self,
        input_video_path: Path,
        dubbed_audio_path: Path,
        output_path: Path,
        subtitle_path: str | Path | None,
        original_audio_volume_db: float,
        burn_subtitle: bool,
        subtitle_style: Dict[str, Any] | None = None,
    ) -> list[str]:
        volume_expr = f"{float(original_audio_volume_db)}dB"

        if burn_subtitle:
            subtitle_filter = self._build_subtitle_filter(
                subtitle_path=subtitle_path,
                subtitle_style=subtitle_style,
            )

            filter_complex = (
                f"[0:v]{subtitle_filter}[vout];"
                f"[0:a]volume={volume_expr}[a0];"
                f"[1:a]volume=1.0[a1];"
                f"[a0][a1]amix=inputs=2:duration=longest:dropout_transition=0[aout]"
            )

            return [
                "ffmpeg",
                "-y",
                "-i",
                str(input_video_path),
                "-i",
                str(dubbed_audio_path),
                "-filter_complex",
                filter_complex,
                "-map",
                "[vout]",
                "-map",
                "[aout]",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-shortest",
                str(output_path),
            ]

        filter_complex = (
            f"[0:a]volume={volume_expr}[a0];"
            f"[1:a]volume=1.0[a1];"
            f"[a0][a1]amix=inputs=2:duration=longest:dropout_transition=0[aout]"
        )

        return [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video_path),
            "-i",
            str(dubbed_audio_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]

    def copy_subtitle_to_output(
        self,
        subtitle_path: str | Path,
        video_stem: str,
    ) -> str:
        source = Path(subtitle_path)
        output_path = self.output_dir / f"{video_stem}_vi.srt"

        output_path.write_text(
            source.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        return str(output_path)