from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List


AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
RATE_LADDER = ["+0%", "+10%", "+15%", "+20%"]


def normalize_rate_value(rate: str | None) -> str:
    """
    Chuẩn hóa rate về format +0%, +10%, +15%, +20%.
    """
    rate = str(rate or "").strip()

    if rate in RATE_LADDER:
        return rate

    aliases = {
        "0%": "+0%",
        "default": "+0%",
        "mặc định": "+0%",
        "mac dinh": "+0%",
        "+5%": "+10%",
    }

    return aliases.get(rate.lower(), "+0%")


def next_rate_value(current_rate: str | None) -> str:
    """
    Tăng rate lên một nấc an toàn.
    """
    current_rate = normalize_rate_value(current_rate)

    try:
        idx = RATE_LADDER.index(current_rate)
    except ValueError:
        return "+10%"

    if idx >= len(RATE_LADDER) - 1:
        return RATE_LADDER[-1]

    return RATE_LADDER[idx + 1]


def suggest_alignment_rate_fix(
    offset_sec: float,
    current_rate: str | None,
    heavy_threshold_sec: float = 0.75,
    light_threshold_sec: float = 0.25,
) -> tuple[str, str]:
    """
    Level 3B: đề xuất auto-fix rate.

    Chỉ auto-fix khi TTS dài hơn subtitle.
    TTS ngắn hơn subtitle không gây đè audio nên không tăng rate.
    """
    current_rate = normalize_rate_value(current_rate)

    if offset_sec <= light_threshold_sec:
        return current_rate, "Không cần auto-fix"

    suggested_rate = next_rate_value(current_rate)

    if suggested_rate == current_rate and current_rate == "+20%":
        if offset_sec > heavy_threshold_sec:
            return current_rate, "Đã +20%, cần rút gọn hoặc split text"
        return current_rate, "Đã +20%, kiểm tra thủ công"

    if offset_sec > heavy_threshold_sec:
        return suggested_rate, f"TTS dài hơn {offset_sec:.2f}s, tăng rate {current_rate} -> {suggested_rate}"

    return suggested_rate, f"TTS dài hơn nhẹ {offset_sec:.2f}s, tăng rate {current_rate} -> {suggested_rate}"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def probe_audio_duration(audio_path: str | Path) -> float:
    """
    Đo duration audio bằng ffprobe.
    Project của bạn đã dùng ffmpeg để render video, nên ffprobe thường có sẵn cùng ffmpeg.
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        return 0.0

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return max(float(result.stdout.strip()), 0.0)
    except Exception:
        return 0.0


def _extract_segment_number_from_name(path: Path) -> int | None:
    """
    Cố gắng lấy segment id từ tên file audio.
    Hỗ trợ các dạng:
    - segment_0001.mp3
    - segment_1.mp3
    - tts_0001_xxx.mp3
    - 0001.mp3
    """
    stem = path.stem.lower()

    patterns = [
        r"segment[_\- ]?(\d+)",
        r"seg[_\- ]?(\d+)",
        r"tts[_\- ]?(\d+)",
        r"^(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None

    return None


def collect_tts_audio_files(tts_root: str | Path) -> Dict[int, Path]:
    """
    Quét toàn bộ thư mục data/tts_segments để tìm audio TTS theo segment id.
    Nếu có nhiều file cùng id, chọn file mới nhất.
    """
    tts_root = Path(tts_root)

    if not tts_root.exists():
        return {}

    candidates: Dict[int, List[Path]] = {}

    for path in tts_root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        # Bỏ qua preview audio
        if "_preview" in path.parts:
            continue

        segment_id = _extract_segment_number_from_name(path)
        if segment_id is None:
            continue

        candidates.setdefault(segment_id, []).append(path)

    selected: Dict[int, Path] = {}

    for segment_id, files in candidates.items():
        # Chọn file mới nhất để tránh lấy nhầm cache cũ
        files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
        selected[segment_id] = files[0]

    return selected


def classify_alignment_offset(offset_sec: float, abs_offset_sec: float) -> str:
    """
    Phân loại mức lệch.
    offset_sec = audio_duration - subtitle_duration
    """
    if abs_offset_sec <= 0.25:
        return "OK"

    if abs_offset_sec <= 0.75:
        if offset_sec > 0:
            return "TTS dài hơn nhẹ"
        return "TTS ngắn hơn nhẹ"

    if offset_sec > 0:
        return "TTS dài hơn nhiều"

    return "TTS ngắn hơn nhiều"


def build_tts_alignment_report(
    bilingual_path: str | Path,
    tts_root: str | Path,
    output_dir: str | Path,
    video_stem: str,
    tts_segments_path: str | Path | None = None,
) -> dict:
    """
    Tạo report kiểm tra lệch giữa duration segment và duration audio TTS.

    Output gồm:
    - report_json_path
    - report_csv_path
    - rows
    - summary
    """
    bilingual_path = Path(bilingual_path)
    tts_root = Path(tts_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(bilingual_path.read_text(encoding="utf-8"))
    segments = data.get("segments", [])

    if video_stem is None:
        video_stem = (
            data.get("video_stem")
            or Path(data.get("video_name", bilingual_path.stem)).stem
        )
    audio_by_id = {}

    if tts_segments_path and Path(tts_segments_path).exists():
        tts_segments_data = json.loads(Path(tts_segments_path).read_text(encoding="utf-8"))
        if isinstance(tts_segments_data, list):
            tts_segments = tts_segments_data
        else:
            tts_segments = tts_segments_data.get("segments", [])

        for idx, item in enumerate(tts_segments, start=1):
            sid = int(item.get("id") or item.get("segment_id") or idx)
            audio_path = item.get("tts_audio_path")
            if audio_path:
                audio_by_id[sid] = Path(audio_path)
    else:
        audio_by_id = collect_tts_audio_files(tts_root)

    # Quan trọng: nếu có tts_segments_path thì report phải dùng đúng audio
    # của lần render vừa xong, không được scan lại cả folder cache cũ.
    audio_by_segment_id = audio_by_id

    rows: List[Dict[str, Any]] = []

    for idx, segment in enumerate(segments, start=1):
        segment_id = int(segment.get("segment_id") or segment.get("id") or idx)

        start = _safe_float(segment.get("start"), 0.0)
        end = _safe_float(segment.get("end"), start + 0.1)
        subtitle_duration = max(end - start, 0.1)

        audio_path = audio_by_segment_id.get(segment_id)
        audio_duration = probe_audio_duration(audio_path) if audio_path else 0.0

        offset_sec = audio_duration - subtitle_duration if audio_duration > 0 else 0.0
        abs_offset_sec = abs(offset_sec) if audio_duration > 0 else 0.0

        status = (
            classify_alignment_offset(offset_sec, abs_offset_sec)
            if audio_duration > 0
            else "Chưa tìm thấy audio TTS"
        )
        current_rate = str(
            segment.get("rate")
            or segment.get("tts_rate")
            or "+0%"
        ).strip()

        suggested_rate, fix_action = suggest_alignment_rate_fix(
            offset_sec=offset_sec,
            current_rate=current_rate,
        )

        rows.append(
            {
                "segment_id": segment_id,
                "start": round(start, 3),
                "end": round(end, 3),
                "subtitle_duration": round(subtitle_duration, 3),
                "tts_audio_duration": round(audio_duration, 3),
                "offset_sec": round(offset_sec, 3),
                "abs_offset_sec": round(abs_offset_sec, 3),
                "alignment_status": status,

                # Level 3B
                "current_rate": current_rate,
                "suggested_rate": suggested_rate,
                "fix_action": fix_action,

                "tts_audio_path": str(audio_path) if audio_path else "",
                "vi_text": str(segment.get("vi_text", "")).strip(),
            }
        )

    total = len(rows)
    missing_audio = sum(1 for row in rows if row["alignment_status"] == "Chưa tìm thấy audio TTS")
    ok_count = sum(1 for row in rows if row["alignment_status"] == "OK")
    light_offset = sum(
        1
        for row in rows
        if row["alignment_status"] in {"TTS dài hơn nhẹ", "TTS ngắn hơn nhẹ"}
    )
    heavy_too_long = sum(
        1
        for row in rows
        if row["alignment_status"] == "TTS dài hơn nhiều"
    )

    heavy_too_short = sum(
        1
        for row in rows
        if row["alignment_status"] == "TTS ngắn hơn nhiều"
    )

    max_abs_offset = max((row["abs_offset_sec"] for row in rows), default=0.0)
    avg_abs_offset = (
        sum(row["abs_offset_sec"] for row in rows) / total
        if total > 0
        else 0.0
    )

    summary = {
        "total_segments": total,
        "ok_count": ok_count,
        "light_offset_count": light_offset,
        "heavy_offset_count": heavy_too_long + heavy_too_short,
        "tts_too_long_heavy_count": heavy_too_long,
        "tts_too_short_heavy_count": heavy_too_short,
        "auto_fix_candidate_count": heavy_too_long,
        "missing_audio_count": missing_audio,
        "max_abs_offset_sec": round(max_abs_offset, 3),
        "avg_abs_offset_sec": round(avg_abs_offset, 3),
    }

    report = {
        "video_stem": video_stem,
        "bilingual_path": str(bilingual_path),
        "tts_root": str(tts_root),
        "tts_segments_path": str(tts_segments_path) if tts_segments_path else None,
        "summary": summary,
        "rows": rows,
    }

    json_path = output_dir / f"{video_stem}_tts_alignment_report.json"
    csv_path = output_dir / f"{video_stem}_tts_alignment_report.csv"

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "segment_id",
            "start",
            "end",
            "subtitle_duration",
            "tts_audio_duration",
            "offset_sec",
            "abs_offset_sec",
            "alignment_status",

            # Level 3B
            "current_rate",
            "suggested_rate",
            "fix_action",

            "tts_audio_path",
            "vi_text",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "report_json_path": str(json_path),
        "report_csv_path": str(csv_path),
        "summary": summary,
        "rows": rows,
    }