import json
from pathlib import Path
from pydub import AudioSegment, silence
from typing import List, Dict, Optional


class AudioTimelinePostprocessor:
    """
    Căn chỉnh segment TTS + audio:
    - Merge segment gần nhau
    - Fill silence nếu audio quá ngắn
    - Cập nhật start/end chuẩn
    - Xuất JSON và SRT
    """

    def __init__(self, min_gap_sec: float = 0.3, fill_silence_db: float = -50):
        """
        min_gap_sec: khoảng cách nhỏ hơn sẽ merge segment
        fill_silence_db: âm lượng dB để fill silence
        """
        self.min_gap_sec = min_gap_sec
        self.fill_silence_db = fill_silence_db

    @staticmethod
    def get_audio_duration_ms(path: Path) -> int:
        if not path.exists():
            return 0
        try:
            audio = AudioSegment.from_file(path)
            return len(audio)
        except Exception:
            return 0

    @staticmethod
    def create_silence(duration_ms: int, db: float = -50) -> AudioSegment:
        return AudioSegment.silent(duration=duration_ms).apply_gain(db)

    def merge_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Merge các segment gần nhau nếu gap < min_gap_sec
        """
        if not segments:
            return []

        merged: List[Dict] = []
        prev = segments[0].copy()
        for seg in segments[1:]:
            gap = seg["start"] - prev["end"]
            if gap <= self.min_gap_sec:
                # Merge segment
                prev["vi_text"] += " " + seg.get("vi_text", "")
                prev["tts_audio_path"] = prev.get("tts_audio_path") or seg.get(
                    "tts_audio_path"
                )
                prev["end"] = seg["end"]
            else:
                merged.append(prev)
                prev = seg.copy()
        merged.append(prev)
        return merged

    def adjust_audio_duration(self, segments: List[Dict]) -> List[Dict]:
        """
        Fill silence cuối segment nếu audio < target duration
        """
        for seg in segments:
            audio_path = seg.get("tts_audio_path")
            if not audio_path:
                continue
            audio_path = Path(audio_path)
            audio_ms = self.get_audio_duration_ms(audio_path)
            seg["audio_duration_ms"] = audio_ms

            # target duration từ timestamp
            target_ms = int((seg["end"] - seg["start"]) * 1000)
            seg["target_duration_ms"] = target_ms

            if audio_ms < target_ms:
                # Fill silence
                fill_ms = target_ms - audio_ms
                try:
                    audio = AudioSegment.from_file(audio_path)
                    silence_segment = self.create_silence(fill_ms, self.fill_silence_db)
                    audio += silence_segment
                    audio.export(audio_path, format="mp3")
                    seg["audio_duration_ms"] = len(audio)
                except Exception as e:
                    print(f"[WARN] Fill silence failed for {audio_path}: {e}")
        return segments

    def postprocess(
        self, bilingual_json_path: Path, output_json_path: Path, output_srt_path: Optional[Path] = None
    ):
        """
        Main function:
        1. Load JSON
        2. Merge segment gần nhau
        3. Fill silence để khớp timestamp
        4. Xuất JSON và SRT
        """
        segments = json.loads(bilingual_json_path.read_text(encoding="utf-8"))

        segments = self.merge_segments(segments)
        segments = self.adjust_audio_duration(segments)

        # Xuất JSON
        output_json_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")

        # Xuất SRT nếu cần
        if output_srt_path:
            self.write_srt(segments, output_srt_path)

    @staticmethod
    def write_srt(segments: List[Dict], srt_path: Path):
        def ms_to_srt_time(ms: int) -> str:
            h = ms // 3600000
            m = (ms % 3600000) // 60000
            s = (ms % 60000) // 1000
            ms_rem = ms % 1000
            return f"{h:02d}:{m:02d}:{s:02d},{ms_rem:03d}"

        lines = []
        for idx, seg in enumerate(segments, start=1):
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)
            text = seg.get("vi_text", "").replace("\n", " ").strip()
            lines.append(f"{idx}")
            lines.append(f"{ms_to_srt_time(start_ms)} --> {ms_to_srt_time(end_ms)}")
            lines.append(text)
            lines.append("")

        srt_path.write_text("\n".join(lines), encoding="utf-8")