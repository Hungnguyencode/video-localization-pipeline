from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

TERMINAL_PUNCTUATION = (".", "?", "!", "…")


DEFAULT_WEAK_START_WORDS = {
    # Vietnamese
    "về", "và", "nhưng", "nên", "rồi", "thì", "để", "cho", "bằng", "của",
    "khi", "nếu", "mà", "hoặc", "hay", "từ", "trong", "với", "như", "là",
    "ở", "đến", "cùng", "còn", "cũng", "các", "những", "giai", "sự",

    # English
    "about", "and", "but", "so", "because", "of", "to", "for", "with", "that",
    "which", "where", "when", "why", "how", "in", "on", "at", "by", "as",
    "from", "into", "around", "between", "airports", "busiest", "bit",
}


DEFAULT_WEAK_END_WORDS = {
    # Vietnamese
    "và", "nhưng", "vì", "để", "cho", "bằng", "của", "khi", "nếu", "về",
    "từ", "trong", "với", "như", "là", "ở", "đến", "cùng", "còn", "cũng",
    "một", "các", "những", "cao", "nhất",

    # English
    "and", "but", "because", "of", "to", "for", "with", "that", "which", "where",
    "when", "why", "how", "in", "on", "at", "by", "as", "from", "into",
    "around", "between", "their", "a", "an", "the", "this", "these", "those",
    "some", "many", "most", "profitable",
}


# Cụm ở ĐẦU segment sau nhưng thật ra thuộc về CUỐI segment trước.
# Ví dụ Whisper tách: "Istanbul and Munich" / "airports so really..."
CARRY_BACK_PATTERNS: List[Tuple[re.Pattern[str], int]] = [
    (re.compile(r"^(airports)\b[\s,]*", re.IGNORECASE), 1),
    (re.compile(r"^(airport)\b[\s,]*", re.IGNORECASE), 1),
    (re.compile(r"^(busiest\s+and\s+peak\s+periods)\b[\s,]*", re.IGNORECASE), 4),
    (re.compile(r"^(peak\s+periods)\b[\s,]*", re.IGNORECASE), 2),
]


# Cụm ở CUỐI segment trước nhưng thật ra là phần mở đầu câu sau.
# Ví dụ: "... summer holiday routes. So there's a" / "bit of reassurance..."
TRAILING_TAIL_MAX_WORDS = 6


def _clean_text(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _get_source_text(segment: Dict[str, Any]) -> str:
    return _clean_text(
        segment.get("source_text")
        or segment.get("text")
        or segment.get("sentence")
        or ""
    )


def _set_source_text(segment: Dict[str, Any], text: str) -> None:
    text = _clean_text(text)
    segment["text"] = text
    segment["source_text"] = text


def _duration(segment: Dict[str, Any]) -> float:
    return max(0.0, float(segment.get("end", 0.0)) - float(segment.get("start", 0.0)))


def _gap_between(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    return float(b.get("start", 0.0)) - float(a.get("end", 0.0))


def _word_count(text: str) -> int:
    return len(re.findall(r"[\wÀ-ỹ']+", _clean_text(text), flags=re.UNICODE))


def _first_word(text: str) -> str:
    text = _clean_text(text).lower()
    match = re.match(r"^[\W_]*([\wÀ-ỹ']+)", text, flags=re.UNICODE)
    return match.group(1) if match else ""


def _last_word(text: str) -> str:
    words = re.findall(r"[\wÀ-ỹ']+", _clean_text(text).lower(), flags=re.UNICODE)
    return words[-1] if words else ""


def _starts_lowercase(text: str) -> bool:
    text = _clean_text(text)
    if not text:
        return False
    first = text[0]
    return first.isalpha() and first.islower()


def _ends_with_terminal_punctuation(text: str) -> bool:
    return bool(re.search(r'[.!?…]["”\')\]]?\s*$', str(text or "").strip()))


def _starts_with_weak_word(text: str, weak_start_words: set[str]) -> bool:
    return _first_word(text) in weak_start_words


def _ends_with_weak_word(text: str, weak_end_words: set[str]) -> bool:
    return _last_word(text) in weak_end_words


def _estimate_leading_duration_sec(segment: Dict[str, Any], moved_words: int) -> float:
    text = _get_source_text(segment)
    total_words = max(1, _word_count(text))
    ratio = max(0.0, min(0.4, moved_words / total_words))
    # Giới hạn để không làm lệch timestamp quá nhiều.
    return max(0.15, min(1.25, _duration(segment) * ratio))


def _move_leading_phrase_back(current: Dict[str, Any], nxt: Dict[str, Any]) -> bool:
    """
    Chuyển cụm đầu của segment sau về cuối segment trước nếu đó là cụm bị Whisper cắt sai.
    Trường hợp chính: "Istanbul and Munich" / "airports so really...".
    """
    current_text = _get_source_text(current)
    next_text = _get_source_text(nxt)
    if not current_text or not next_text:
        return False

    # Chỉ sửa khi ranh giới trông có vẻ bị cắt câu.
    if _ends_with_terminal_punctuation(current_text):
        return False
    if not _starts_lowercase(next_text):
        return False

    for pattern, moved_words in CARRY_BACK_PATTERNS:
        match = pattern.match(next_text)
        if not match:
            continue

        moved = _clean_text(match.group(1))
        rest = _clean_text(next_text[match.end():])
        if not moved or not rest:
            return False

        _set_source_text(current, f"{current_text} {moved}")
        _set_source_text(nxt, rest)

        # Dịch ranh giới thời gian nhẹ theo số từ đã chuyển.
        shift = _estimate_leading_duration_sec(nxt, moved_words)
        old_end = float(current.get("end", 0.0))
        next_end = float(nxt.get("end", old_end))
        new_boundary = min(next_end - 0.2, old_end + shift)
        if new_boundary > old_end:
            current["end"] = new_boundary
            nxt["start"] = new_boundary
        return True

    return False


def _move_trailing_weak_tail_forward(current: Dict[str, Any], nxt: Dict[str, Any], weak_end_words: set[str]) -> bool:
    """
    Chuyển phần đuôi câu cụt sang segment sau.
    Ví dụ: "... routes. So there's a" + "bit of reassurance..."
    -> "... routes." + "So there's a bit of reassurance..."
    """
    current_text = _get_source_text(current)
    next_text = _get_source_text(nxt)
    if not current_text or not next_text:
        return False

    # Bắt phần sau dấu câu cuối cùng.
    match = re.match(r"^(.*[.!?…])\s+([^.!?…]+)$", current_text)
    if not match:
        return False

    head = _clean_text(match.group(1))
    tail = _clean_text(match.group(2))
    if not head or not tail:
        return False

    # Chỉ chuyển tail ngắn và kết thúc yếu.
    if _word_count(tail) > TRAILING_TAIL_MAX_WORDS:
        return False
    if not _ends_with_weak_word(tail, weak_end_words):
        return False

    _set_source_text(current, head)
    _set_source_text(nxt, f"{tail} {next_text}")
    return True


def _looks_like_continuation(
    current: Dict[str, Any],
    nxt: Dict[str, Any],
    max_gap_sec: float,
    weak_start_words: set[str],
    weak_end_words: set[str],
) -> bool:
    current_text = _get_source_text(current)
    next_text = _get_source_text(nxt)

    if not current_text or not next_text:
        return False

    gap = _gap_between(current, nxt)
    if gap > max_gap_sec:
        return False

    # Nếu segment hiện tại đã kết thúc rõ ràng thì không nối bừa.
    if _ends_with_terminal_punctuation(current_text):
        return False

    if _starts_lowercase(next_text):
        return True

    if _starts_with_weak_word(next_text, weak_start_words):
        return True

    if _ends_with_weak_word(current_text, weak_end_words):
        return True

    return False


def _merge_two_segments(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    a_text = _get_source_text(a)
    b_text = _get_source_text(b)
    merged_text = _clean_text(f"{a_text} {b_text}")

    merged = dict(a)
    merged["text"] = merged_text
    merged["source_text"] = merged_text
    merged["end"] = b.get("end", a.get("end", 0.0))
    return merged


def _repair_boundaries_pass(
    segments: List[Dict[str, Any]],
    weak_end_words: set[str],
) -> List[Dict[str, Any]]:
    """
    Pass nhẹ sau khi merge để sửa các ranh giới cụt nhưng không nhất thiết phải merge cả segment.
    """
    if len(segments) < 2:
        return segments

    repaired = [dict(item) for item in segments]
    for i in range(len(repaired) - 1):
        current = repaired[i]
        nxt = repaired[i + 1]
        _move_leading_phrase_back(current, nxt)
        _move_trailing_weak_tail_forward(current, nxt, weak_end_words)

    return repaired


def merge_segments_for_translation(
    segments: List[Dict[str, Any]],
    enabled: bool = True,
    max_merged_duration_sec: float = 10.0,
    max_merged_chars: int = 240,
    continuation_max_duration_sec: float = 17.0,
    continuation_max_chars: int = 360,
    merge_weak_boundaries: bool = True,
    max_gap_sec: float = 0.8,
    weak_start_words: List[str] | None = None,
    weak_end_words: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Gộp segment ASR trước khi dịch.

    Bản này sửa thêm các lỗi Whisper/VAD hay cắt sai theo âm thanh, ví dụ:
    - "Istanbul and Munich" / "airports so really..."
    - "... routes. So there's a" / "bit of reassurance..."

    Mục tiêu là tạo segment có nghĩa hơn trước khi đưa sang Gemini/TTS.
    """
    if not enabled:
        return segments

    if not segments:
        return []

    # Ép ngưỡng tối thiểu để không bị config cũ 12s làm hỏng các câu tin tức dài.
    continuation_max_duration_sec = max(float(continuation_max_duration_sec), 17.0)
    continuation_max_chars = max(int(continuation_max_chars), 360)

    weak_start_set = set(weak_start_words or DEFAULT_WEAK_START_WORDS)
    weak_end_set = set(weak_end_words or DEFAULT_WEAK_END_WORDS)

    sorted_segments = sorted(
        [dict(item) for item in segments],
        key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))),
    )

    output: List[Dict[str, Any]] = []
    current = dict(sorted_segments[0])

    for nxt in sorted_segments[1:]:
        nxt = dict(nxt)

        merged_candidate = _merge_two_segments(current, nxt)
        candidate_duration = _duration(merged_candidate)
        candidate_chars = len(_get_source_text(merged_candidate))

        should_merge = False
        if merge_weak_boundaries:
            should_merge = _looks_like_continuation(
                current=current,
                nxt=nxt,
                max_gap_sec=max_gap_sec,
                weak_start_words=weak_start_set,
                weak_end_words=weak_end_set,
            )

        if should_merge:
            can_merge = (
                candidate_duration <= continuation_max_duration_sec
                and candidate_chars <= continuation_max_chars
            )
        else:
            can_merge = (
                candidate_duration <= float(max_merged_duration_sec)
                and candidate_chars <= int(max_merged_chars)
                and not _ends_with_terminal_punctuation(_get_source_text(current))
                and _gap_between(current, nxt) <= max_gap_sec
            )

        if can_merge:
            current = merged_candidate
        else:
            # Trước khi chốt current, sửa các cụm bị cắt nhưng không cần merge nguyên segment.
            _move_leading_phrase_back(current, nxt)
            _move_trailing_weak_tail_forward(current, nxt, weak_end_set)
            output.append(current)
            current = nxt

    output.append(current)

    # Pass cuối để sửa tail/leading còn sót sau vòng merge.
    output = _repair_boundaries_pass(output, weak_end_set)

    for idx, item in enumerate(output, start=1):
        item["id"] = idx
        item["segment_id"] = idx

    return output
