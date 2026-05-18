from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict

from src.alignment.audio_aligner import AudioAligner
from src.asr.whisper_asr import FasterWhisperASR
from src.audio.extractor import AudioExtractor
from src.ingest.local_video import LocalVideoIngestor
from src.subtitle.segment_merger import merge_segments_for_translation
from src.subtitle.srt_writer import SubtitleWriter
from src.translation.translator import build_translator
from src.tts.edge_tts_engine import EdgeTTSEngine
from src.utils.config import ensure_dir, load_config, resolve_path
from src.utils.io import load_json, save_json
from src.utils.media import get_video_duration_seconds
from src.video_render.renderer import VideoRenderer


class VideoLocalizationPipeline:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        paths = self.config["paths"]
        for path_value in paths.values():
            ensure_dir(path_value)

        self.ingestor = LocalVideoIngestor(paths["input_dir"])
        self.audio_extractor = AudioExtractor(
            output_dir=paths["audio_dir"],
            sample_rate=self.config["audio"].get("sample_rate", 16000),
            channels=self.config["audio"].get("channels", 1),
        )

        asr_cfg = self.config["asr"]
        self.asr = FasterWhisperASR(
            model_name=asr_cfg.get("model_name", "small"),
            device=asr_cfg.get("device", "auto"),
            compute_type=asr_cfg.get("compute_type", "int8_float16"),
            language=asr_cfg.get("language", "auto"),
            fallback_to_cpu=asr_cfg.get("fallback_to_cpu", True),
        )

        self.translator = build_translator(self.config["translation"])
        self.subtitle_writer = SubtitleWriter()

        tts_cfg = self.config["tts"]
        self.tts_engine = EdgeTTSEngine(
            output_dir=paths["tts_segments_dir"],
            voice=tts_cfg.get("voice", "fpt:banmai"),
            rate=tts_cfg.get("rate", "+15%"),
            volume=tts_cfg.get("volume", "+0%"),
        )

        align_cfg = self.config["alignment"]
        self.audio_aligner = AudioAligner(
            output_dir=paths["audio_dir"],
            max_speedup=align_cfg.get("max_speedup", 1.45),
            trim_if_too_long=align_cfg.get("trim_if_too_long", False),
            silence_padding_ms=align_cfg.get("silence_padding_ms", 80),
            min_gap_ms=align_cfg.get("min_gap_ms", 60),
        )

        render_cfg = self.config["render"]
        self.renderer = VideoRenderer(
            output_dir=paths["output_dir"],
            output_video_suffix=render_cfg.get("output_video_suffix", "_vi_dubbed"),
        )

    def _path(self, key: str, filename: str | None = None) -> Path:
        base = resolve_path(self.config["paths"][key])
        return base / filename if filename else base

    def prepare_translation(self, video_path: str | Path) -> Dict[str, Any]:
        ingest_result = self.ingestor.ingest(video_path)
        video_stem = ingest_result["video_stem"]
        local_video_path = ingest_result["video_path"]
        video_name = ingest_result["video_name"]
        video_duration = get_video_duration_seconds(local_video_path)

        audio_path = self.audio_extractor.extract(video_path=local_video_path, video_stem=video_stem)
        asr_result = self.asr.transcribe(audio_path=audio_path, video_name=video_name)
        raw_segments = asr_result["segments"]

        transcript_path = self._path("transcripts_dir", f"{video_stem}_source_transcript.json")
        save_json(asr_result, transcript_path)

        source_srt_path = self._path("subtitles_dir", f"{video_stem}_source.srt")
        self.subtitle_writer.write_source_srt(raw_segments, source_srt_path)

        merge_cfg = self.config["translation"].get("merge_segments", {})
        segments_for_translation = merge_segments_for_translation(
            raw_segments,
            enabled=merge_cfg.get("enabled", True),
            max_merged_duration_sec=merge_cfg.get("max_duration_sec", 10.0),
            max_merged_chars=merge_cfg.get("max_chars", 240),
            continuation_max_duration_sec=merge_cfg.get(
                "continuation_max_duration_sec",
                merge_cfg.get("max_duration_sec", 12.0),
            ),
            continuation_max_chars=merge_cfg.get(
                "continuation_max_chars",
                merge_cfg.get("max_chars", 280),
            ),
            max_gap_sec=merge_cfg.get("max_gap_sec", 0.8),
        )

        merged_source_srt_path = self._path("subtitles_dir", f"{video_stem}_source_merged.srt")
        self.subtitle_writer.write_source_srt(segments_for_translation, merged_source_srt_path)

        merged_transcript_path = self._path("transcripts_dir", f"{video_stem}_source_merged_transcript.json")
        save_json(
            {
                "video_name": video_name,
                "video_duration_sec": video_duration,
                "raw_segments_count": len(raw_segments),
                "merged_segments_count": len(segments_for_translation),
                "segments": segments_for_translation,
            },
            merged_transcript_path,
        )

        translated_segments = self.translator.translate_segments(segments_for_translation)
        bilingual_data = {
            "video_name": video_name,
            "video_stem": video_stem,
            "video_duration_sec": video_duration,
            "input_video_path": local_video_path,
            "audio_path": audio_path,
            "source_language": asr_result.get("language"),
            "target_language": "vi",
            "raw_segments_count": len(raw_segments),
            "merged_segments_count": len(segments_for_translation),
            "segments": translated_segments,
        }
        bilingual_path = self._path("transcripts_dir", f"{video_stem}_bilingual.json")
        save_json(bilingual_data, bilingual_path)

        vi_srt_path = self._path("subtitles_dir", f"{video_stem}_vi.srt")
        self.subtitle_writer.write_vietnamese_srt(translated_segments, vi_srt_path)

        return {
            "stage": "translation_prepared",
            "video_name": video_name,
            "video_stem": video_stem,
            "video_duration_sec": video_duration,
            "input_video_path": local_video_path,
            "audio_path": audio_path,
            "source_transcript_path": str(transcript_path),
            "source_srt_path": str(source_srt_path),
            "merged_source_srt_path": str(merged_source_srt_path),
            "merged_transcript_path": str(merged_transcript_path),
            "bilingual_transcript_path": str(bilingual_path),
            "vi_srt_path": str(vi_srt_path),
            "raw_segments_count": len(raw_segments),
            "merged_segments_count": len(segments_for_translation),
            "segments_count": len(translated_segments),
        }

    def render_from_bilingual(
        self,
        bilingual_path: str | Path,
        clear_tts_cache: bool = True,
        audio_mode: str | None = None,
        burn_subtitle: bool | None = None,
        original_audio_volume_db: float | None = None,
        subtitle_style: Dict[str, Any] | None = None,
        subtitle_max_chars_per_line: int | None = None,
    ) -> Dict[str, Any]:
        bilingual_path = Path(bilingual_path)
        bilingual_data = load_json(bilingual_path)
        video_name = bilingual_data["video_name"]
        video_stem = bilingual_data.get("video_stem") or Path(video_name).stem
        video_duration = float(bilingual_data["video_duration_sec"])
        input_video_path = bilingual_data["input_video_path"]
        translated_segments = bilingual_data["segments"]

        render_cfg = self.config.get("render", {})
        audio_mode = audio_mode or render_cfg.get("audio_mode", "replace")
        burn_subtitle = bool(burn_subtitle) if burn_subtitle is not None else bool(render_cfg.get("burn_subtitle", False))
        original_audio_volume_db = (
            float(original_audio_volume_db)
            if original_audio_volume_db is not None
            else float(render_cfg.get("original_audio_volume_db", -18.0))
        )
        subtitle_max_chars_per_line = (
            int(subtitle_max_chars_per_line)
            if subtitle_max_chars_per_line is not None
            else int(render_cfg.get("subtitle_max_chars_per_line", 42))
        )

        vi_srt_path = self._path("subtitles_dir", f"{video_stem}_vi.srt")
        self.subtitle_writer.write_vietnamese_srt(
            translated_segments,
            vi_srt_path,
            max_chars_per_line=subtitle_max_chars_per_line,
        )

        if audio_mode == "subtitle_only":
            output_video_path = self.renderer.render_localized_video(
                input_video_path=input_video_path,
                dubbed_audio_path=None,
                video_stem=video_stem,
                subtitle_path=vi_srt_path,
                audio_mode="subtitle_only",
                original_audio_volume_db=original_audio_volume_db,
                burn_subtitle=True,
                subtitle_style=subtitle_style,
            )
            output_subtitle_path = self.renderer.copy_subtitle_to_output(vi_srt_path, video_stem)
            return {
                "stage": "render_finished",
                "video_name": video_name,
                "video_stem": video_stem,
                "video_duration_sec": video_duration,
                "input_video_path": input_video_path,
                "bilingual_transcript_path": str(bilingual_path),
                "vi_srt_path": str(vi_srt_path),
                "tts_segments_path": None,
                "dubbed_audio_path": None,
                "output_video_path": str(output_video_path),
                "output_subtitle_path": str(output_subtitle_path),
                "segments_count": len(translated_segments),
                "audio_mode": audio_mode,
                "burn_subtitle": True,
                "original_audio_volume_db": original_audio_volume_db,
                "subtitle_style": subtitle_style,
                "subtitle_max_chars_per_line": subtitle_max_chars_per_line,
            }

        use_smart_cache = not clear_tts_cache
        if clear_tts_cache:
            tts_dir = self._path("tts_segments_dir", video_stem)
            if tts_dir.exists():
                shutil.rmtree(tts_dir)

        tts_segments = self.tts_engine.synthesize_segments(
            translated_segments,
            video_stem=video_stem,
            use_smart_cache=use_smart_cache,
        )
        tts_data_path = self._path("transcripts_dir", f"{video_stem}_tts_segments.json")
        save_json(tts_segments, tts_data_path)

        dubbed_audio_path = self.audio_aligner.build_dubbed_audio(
            segments=tts_segments,
            video_duration_sec=video_duration,
            video_stem=video_stem,
        )
        output_video_path = self.renderer.render_localized_video(
            input_video_path=input_video_path,
            dubbed_audio_path=dubbed_audio_path,
            video_stem=video_stem,
            subtitle_path=vi_srt_path,
            audio_mode=audio_mode,
            original_audio_volume_db=original_audio_volume_db,
            burn_subtitle=burn_subtitle,
            subtitle_style=subtitle_style,
        )
        output_subtitle_path = self.renderer.copy_subtitle_to_output(vi_srt_path, video_stem)

        return {
            "stage": "render_finished",
            "video_name": video_name,
            "video_stem": video_stem,
            "video_duration_sec": video_duration,
            "input_video_path": input_video_path,
            "bilingual_transcript_path": str(bilingual_path),
            "vi_srt_path": str(vi_srt_path),
            "tts_segments_path": str(tts_data_path),
            "dubbed_audio_path": str(dubbed_audio_path),
            "output_video_path": str(output_video_path),
            "output_subtitle_path": str(output_subtitle_path),
            "segments_count": len(translated_segments),
            "audio_mode": audio_mode,
            "burn_subtitle": burn_subtitle,
            "original_audio_volume_db": original_audio_volume_db,
            "subtitle_style": subtitle_style,
            "subtitle_max_chars_per_line": subtitle_max_chars_per_line,
        }

    def process_video(self, video_path: str | Path) -> Dict[str, Any]:
        prepared = self.prepare_translation(video_path)
        rendered = self.render_from_bilingual(prepared["bilingual_transcript_path"], clear_tts_cache=True)
        return {**prepared, **rendered}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run video localization pipeline")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    pipeline = VideoLocalizationPipeline(config_path=args.config)
    result = pipeline.process_video(args.video)
    print("=== Pipeline finished ===")
    for key, value in result.items():
        print(f"{key}: {value}")
