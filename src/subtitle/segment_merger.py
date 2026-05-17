from __future__ import annotations

import re
from typing import Any, Dict, List

TERMINAL_PUNCTUATION = (".", "?", "!", "…")


# Các từ/cụm bắt đầu yếu: nếu segment sau bắt đầu bằng các từ này
# thì nhiều khả năng segment đó là phần tiếp của câu trước.
DEFAULT_WEAK_START_WORDS = {
    # Vietnamese
    "về",
    "và",
    "nhưng",
    "nên",
    "rồi",
    "thì",
    "để",
    "cho",
    "bằng",
    "của",
    "khi",
    "nếu",
    "mà",
    "hoặc",
    "hay",
    "từ",
    "trong",
    "với",
    "như",
    "là",
    "ở",
    "đến",
    "cùng",
    "còn",
    "cũng",

    # English
    "about",
    "and",
    "but",
    "so",
    "because",
    "of",
    "to",
    "for",
    "with",
    "that",
    "which",
    "where",
    "when",
    "why",
    "how",
    "in",
    "on",
    "at",
    "by",
    "as",
    "from",
    "into",
    "around",
    "between",
}


# Các từ/cụm kết thúc yếu: nếu segment hiện tại kết thúc bằng các từ này
# thì nên nối với segment sau.
DEFAULT_WEAK_END_WORDS = {
    # Vietnamese
    "và",
    "nhưng",
    "vì",
    "để",
    "cho",
    "bằng",
    "của",
    "khi",
    "nếu",
    "về",
    "từ",
    "trong",
    "với",
    "như",
    "là",
    "ở",
    "đến",
    "cùng",
    "còn",
    "cũng",

    # English
    "and",
    "but",
    "because",
    "of",
    "to",
    "for",
    "with",
    "that",
    "which",
    "where",
    "when",
    "why",
    "how",
    "in",
    "on",
    "at",
    "by",
    "as",
    "from",
    "into",
    "around",
    "between",
}


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


def _duration(segment: Dict[str, Any]) -> float:
    return max(
        0.0,
        float(segment.get("end", 0.0)) - float(segment.get("start", 0.0)),
    )


def _gap_between(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    return float(b.get("start", 0.0)) - float(a.get("end", 0.0))


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
    return _clean_text(text).endswith(TERMINAL_PUNCTUATION)


def _starts_with_weak_word(text: str, weak_start_words: set[str]) -> bool:
    first_word = _first_word(text)
    return first_word in weak_start_words


def _ends_with_weak_word(text: str, weak_end_words: set[str]) -> bool:
    last_word = _last_word(text)
    return last_word in weak_end_words


def _looks_like_continuation(
    current: Dict[str, Any],
    nxt: Dict[str, Any],
    max_gap_sec: float,
    weak_start_words: set[str],
    weak_end_words: set[str],
) -> bool:
    """
    Xác định ranh giới hiện tại có giống một câu bị cắt lẻ không.

    Ví dụ cần merge:
    - Segment trước: "Nhân tiện, mình rất muốn nghe ý kiến của bạn."
    - Segment sau:   "về cách mình có thể làm video tốt hơn nữa."

    Hoặc:
    - Segment sau bắt đầu bằng chữ thường.
    - Segment trước chưa có dấu kết thúc câu.
    - Segment sau bắt đầu bằng từ nối/yếu.
    - Segment trước kết thúc bằng từ nối/yếu.
    """
    current_text = _get_source_text(current)
    next_text = _get_source_text(nxt)

    if not current_text or not next_text:
        return False

    gap = _gap_between(current, nxt)

    # Nếu khoảng cách quá xa thì không nối, tránh nối sai ý.
    if gap > max_gap_sec:
        return False

    # Case 1: segment sau bắt đầu bằng chữ thường.
    # Đây là dấu hiệu rất mạnh rằng nó là phần tiếp của câu trước.
    if _starts_lowercase(next_text):
        return True

    # Case 2: segment sau bắt đầu bằng từ yếu như "về", "và", "nhưng"...
    if _starts_with_weak_word(next_text, weak_start_words):
        return True

    # Case 3: segment trước kết thúc bằng từ yếu như "và", "về", "để"...
    if _ends_with_weak_word(current_text, weak_end_words):
        return True

    # Case 4: segment trước chưa có dấu kết thúc câu,
    # và segment sau nhìn giống continuation.
    if (
        not _ends_with_terminal_punctuation(current_text)
        and (
            _starts_lowercase(next_text)
            or _starts_with_weak_word(next_text, weak_start_words)
        )
    ):
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


def merge_segments_for_translation(
    segments: List[Dict[str, Any]],
    enabled: bool = True,
    max_merged_duration_sec: float = 22.0,
    max_merged_chars: int = 520,
    continuation_max_duration_sec: float = 24.0,
    continuation_max_chars: int = 600,
    merge_weak_boundaries: bool = True,
    max_gap_sec: float = 1.2,
    weak_start_words: List[str] | None = None,
    weak_end_words: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Gộp segment ASR trước khi dịch.

    Mục tiêu:
    - Giảm lỗi segment bị cắt lẻ.
    - Gộp câu bị tách vụn trước khi đưa cho Gemini/translator.
    - Không gộp quá dài để tránh phụ đề/TTS quá nặng.

    Tham số tương thích với code cũ:
    - enabled
    - max_merged_duration_sec
    - max_merged_chars
    - continuation_max_duration_sec
    - continuation_max_chars

    Tham số mới:
    - merge_weak_boundaries
    - max_gap_sec
    - weak_start_words
    - weak_end_words
    """
    if not enabled:
        return segments

    if not segments:
        return []

    weak_start_set = set(weak_start_words or DEFAULT_WEAK_START_WORDS)
    weak_end_set = set(weak_end_words or DEFAULT_WEAK_END_WORDS)

    sorted_segments = sorted(
        [dict(item) for item in segments],
        key=lambda item: (
            float(item.get("start", 0.0)),
            float(item.get("end", 0.0)),
        ),
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

        # Nếu đúng là continuation rõ ràng thì cho phép ngưỡng rộng hơn một chút.
        if should_merge:
            can_merge = (
                candidate_duration <= float(continuation_max_duration_sec)
                and candidate_chars <= int(continuation_max_chars)
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
            output.append(current)
            current = nxt

    output.append(current)

    # Đánh lại id cho đẹp sau khi merge.
    for idx, item in enumerate(output, start=1):
        item["id"] = idx
        item["segment_id"] = idx

    return output