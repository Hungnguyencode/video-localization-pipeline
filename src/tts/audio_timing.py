import shutil
import subprocess
from pathlib import Path


def probe_duration_ms(audio_path: str | Path) -> int:
    """
    Đo duration audio bằng ffprobe.
    Trả về millisecond.
    """

    audio_path = Path(audio_path)

    if not audio_path.exists():
        return 0

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
        duration_sec = float(result.stdout.strip())
        return int(duration_sec * 1000)
    except Exception:
        return 0


def _build_atempo_filter(tempo: float) -> str:
    """
    ffmpeg atempo ổn nhất trong khoảng 0.5..2.0.
    Ở đây mình chỉ dùng 0.86..0.98 nên không cần chain phức tạp,
    nhưng vẫn giữ hàm riêng cho sạch.
    """

    tempo = max(0.5, min(2.0, float(tempo)))
    return f"atempo={tempo:.3f}"


def stretch_audio_if_too_short(
    audio_path: str | Path,
    target_ms: int,
    *,
    min_ratio: float = 0.82,
    min_tempo: float = 0.86,
    max_tempo: float = 0.98,
) -> dict:
    """
    Nếu FPT đọc quá nhanh khiến audio ngắn hơn target nhiều,
    làm chậm audio nhẹ bằng ffmpeg atempo.

    Ví dụ:
    target = 10000ms
    audio = 6500ms
    ratio = 0.65

    Không kéo thẳng 0.65 vì giọng sẽ méo.
    Chỉ kéo nhẹ tối đa min_tempo=0.86.
    """

    audio_path = Path(audio_path)

    info = {
        "changed": False,
        "before_ms": 0,
        "after_ms": 0,
        "target_ms": int(target_ms or 0),
        "tempo": 1.0,
        "reason": "",
    }

    if not audio_path.exists():
        info["reason"] = "audio_not_found"
        return info

    if not target_ms or target_ms <= 0:
        info["reason"] = "invalid_target"
        return info

    before_ms = probe_duration_ms(audio_path)
    info["before_ms"] = before_ms

    if before_ms <= 0:
        info["reason"] = "invalid_audio_duration"
        return info

    ratio = before_ms / target_ms

    # Audio không quá ngắn thì không cần kéo
    if ratio >= min_ratio:
        info["reason"] = "not_short_enough"
        info["after_ms"] = before_ms
        return info

    # tempo < 1 nghĩa là làm chậm audio
    tempo = max(min_tempo, min(max_tempo, ratio))
    info["tempo"] = tempo

    tmp_path = audio_path.with_suffix(audio_path.suffix + ".stretch.tmp.mp3")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-filter:a",
        _build_atempo_filter(tempo),
        "-vn",
        str(tmp_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        if not tmp_path.exists() or tmp_path.stat().st_size <= 0:
            info["reason"] = "stretch_output_empty"
            return info

        shutil.move(str(tmp_path), str(audio_path))

        after_ms = probe_duration_ms(audio_path)
        info["after_ms"] = after_ms
        info["changed"] = True
        info["reason"] = "stretched"

        return info

    except Exception as exc:
        info["reason"] = f"ffmpeg_failed: {exc}"
        return info

    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass