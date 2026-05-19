from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List, Sequence


_TEXT_KEYS = ("vi_text", "target_text", "text", "translation")
_VOICE_KEYS = (
    "speaker",
    "voice",
    "voice_key",
    "voice_label",
    "voice_provider",
    "provider",
    "rate",
    "volume",
)

_DANGLING_END_RE = re.compile(
    r"(?:\b(?:và|hoặc|nhưng|rằng|là|của|cho|với|từ|để|trong|vào|ở|một|có thể là|bao gồm cả)\b)\s*[.!?…]*$",
    re.IGNORECASE,
)
_VI_LOWER_FIRST_CHARS = "a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"

# Không dùng re.IGNORECASE ở đây.
# Nếu dùng IGNORECASE, mọi segment bắt đầu bằng chữ đều bị coi là câu nối tiếp,
# dẫn tới merge toàn bộ video thành 1 segment rất dài.
_CONTINUATION_START_RE = re.compile(
    r"^\s*[\"'“”‘’()\[\]]*"
    rf"(?:[{_VI_LOWER_FIRST_CHARS}]|(?:trong|đến|từ|các|những|một|về|với|cho|của|ở|vào|tuần|tháng|sân bay|thời gian|tác động|hoạt động|hy vọng|lo ngại|khi|nếu|vì vậy|do đó)\b)"
)
_TERMINAL_RE = re.compile(r"[.!?…]+$")


def _get_text(seg: Dict[str, Any]) -> str:
    for key in _TEXT_KEYS:
        value = seg.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _set_text(seg: Dict[str, Any], text: str) -> None:
    target_key = None
    for key in _TEXT_KEYS:
        if key in seg:
            target_key = key
            break
    if target_key is None:
        target_key = "vi_text"
    seg[target_key] = text.strip()


def _duration(seg: Dict[str, Any]) -> float:
    try:
        return max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))
    except Exception:
        return 0.0


def _gap(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    try:
        return float(right.get("start", 0.0)) - float(left.get("end", 0.0))
    except Exception:
        return 999.0


def clean_vi_text_for_output(text: str, *, force_terminal: bool = False) -> str:
    """Clean Vietnamese text without forcing fake sentence endings.

    The old behavior usually added a final dot to every segment. That is risky for
    dubbed video because ASR segments can cut a sentence in the middle. A forced
    dot makes FPT/Edge pause hard and can create text like "trong." / "để.".
    """
    text = (text or "").strip()
    if not text:
        return ""

    text = text.replace("\u200b", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?…])", r"\1", text)
    text = re.sub(r"([,;:])(?=\S)", r"\1 ", text)
    text = re.sub(r"\.\s*\.\s*\.\s*", "… ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Remove terminal punctuation from obviously dangling fragments.
    if _DANGLING_END_RE.search(text):
        text = _TERMINAL_RE.sub("", text).strip()

    # Do not force a period for continuation fragments.
    if force_terminal and text and not re.search(r"[.!?…]$", text):
        if not _CONTINUATION_START_RE.search(text) and not _DANGLING_END_RE.search(text):
            text += "."

    return text


def _looks_continuation_start(text: str) -> bool:
    text = clean_vi_text_for_output(text)
    return bool(text and _CONTINUATION_START_RE.search(text))


def _ends_dangling(text: str) -> bool:
    text = clean_vi_text_for_output(text)
    return bool(text and _DANGLING_END_RE.search(text))


def _voice_signature(seg: Dict[str, Any]) -> tuple:
    return tuple(str(seg.get(k, "") or "") for k in _VOICE_KEYS if k in seg)


def _same_voice_or_unset(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    ls = _voice_signature(left)
    rs = _voice_signature(right)
    if not ls or not rs:
        return True
    return ls == rs


def _copy_voice_fields(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for key in _VOICE_KEYS:
        if key in src and src.get(key) not in (None, ""):
            dst[key] = src.get(key)


def _join_text(left_text: str, right_text: str) -> str:
    left_text = clean_vi_text_for_output(left_text)
    right_text = clean_vi_text_for_output(right_text)
    if not left_text:
        return right_text
    if not right_text:
        return left_text

    # If the next segment starts as a continuation, remove the fake terminal dot.
    if _looks_continuation_start(right_text) or _ends_dangling(left_text):
        left_text = _TERMINAL_RE.sub("", left_text).strip()
    return clean_vi_text_for_output(f"{left_text} {right_text}")


def _merge_pair(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(left)
    merged["start"] = left.get("start", right.get("start", 0.0))
    merged["end"] = right.get("end", left.get("end", 0.0))
    merged["duration"] = _duration(merged)
    _set_text(merged, _join_text(_get_text(left), _get_text(right)))

    # Keep the voice of the non-tiny/longer part. This prevents a 0.2s repair
    # fragment from accidentally changing the whole merged segment voice.
    if _duration(right) > _duration(left):
        _copy_voice_fields(merged, right)
    else:
        _copy_voice_fields(merged, left)

    merged_ids = []
    for seg in (left, right):
        if "merged_from" in seg and isinstance(seg["merged_from"], list):
            merged_ids.extend(seg["merged_from"])
        elif "id" in seg:
            merged_ids.append(seg.get("id"))
    if merged_ids:
        merged["merged_from"] = merged_ids
    return merged


def _should_merge(
    left: Dict[str, Any],
    right: Dict[str, Any],
    *,
    max_gap_sec: float,
    min_duration_sec: float,
    min_chars: int,
    protect_voice_boundary: bool,
) -> bool:
    left_text = _get_text(left)
    right_text = _get_text(right)
    gap = _gap(left, right)
    if gap > max_gap_sec:
        return False

    left_tiny = _duration(left) < min_duration_sec or len(left_text) < min_chars
    right_tiny = _duration(right) < min_duration_sec or len(right_text) < min_chars
    semantic_continuation = _looks_continuation_start(right_text) or _ends_dangling(left_text)

    if protect_voice_boundary and not _same_voice_or_unset(left, right):
        # Still allow merging truly tiny pieces, otherwise TTS providers can fail.
        return left_tiny or right_tiny

    return left_tiny or right_tiny or semantic_continuation


def repair_tiny_tts_segments(
    segments: Sequence[Dict[str, Any]],
    *,
    min_duration_sec: float = 0.65,
    min_chars: int = 5,
    max_gap_sec: float = 1.20,
    protect_voice_boundary: bool = True,
) -> List[Dict[str, Any]]:
    """Repair subtitle/TTS segments before synthesis.

    Fixes two demo-killing cases:
    1. near-zero/tiny segments such as 0.22s that make Edge/FPT fail or go silent;
    2. semantic continuations split across subtitles, for example:
       "... Istanbul và Munich." + "các sân bay."

    The function returns a new list and renumbers `id` from 1..N.
    """
    rows: List[Dict[str, Any]] = []
    for seg in segments or []:
        item = deepcopy(dict(seg))
        _set_text(item, clean_vi_text_for_output(_get_text(item)))
        try:
            item["start"] = float(item.get("start", 0.0))
            item["end"] = float(item.get("end", item.get("start", 0.0)))
        except Exception:
            pass
        rows.append(item)

    rows.sort(key=lambda x: float(x.get("start", 0.0) or 0.0))

    changed = True
    while changed and len(rows) > 1:
        changed = False
        out: List[Dict[str, Any]] = []
        i = 0
        while i < len(rows):
            if i < len(rows) - 1 and _should_merge(
                rows[i],
                rows[i + 1],
                max_gap_sec=max_gap_sec,
                min_duration_sec=min_duration_sec,
                min_chars=min_chars,
                protect_voice_boundary=protect_voice_boundary,
            ):
                out.append(_merge_pair(rows[i], rows[i + 1]))
                i += 2
                changed = True
            else:
                out.append(rows[i])
                i += 1
        rows = out

    for idx, seg in enumerate(rows, start=1):
        seg["id"] = idx
        seg["duration"] = _duration(seg)
        _set_text(seg, clean_vi_text_for_output(_get_text(seg)))
    return rows
