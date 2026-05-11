from __future__ import annotations

import re
from typing import Any, Dict, List


TERMINAL_PUNCTUATION = (".", "?", "!", "…")

# Cụm kết thúc yếu: nếu segment hiện tại kết thúc bằng các cụm này thì nên nối với segment sau.
WEAK_ENDINGS = {
    "a",
    "an",
    "the",
    "to",
    "of",
    "for",
    "with",
    "without",
    "that",
    "which",
    "who",
    "whom",
    "whose",
    "where",
    "when",
    "why",
    "how",
    "in",
    "on",
    "at",
    "from",
    "by",
    "as",
    "into",
    "about",
    "around",
    "between",
    "and",
    "or",
    "but",
    "so",
    "because",
    "due",
    "due to",
    "rising",
    "fuel",
    "travel",
    "coming",
    "coming days",
    "there is a",
    "there's a",
    "is a",
    "are a",
    "a little",
    "carry",
    "carry out",
    "implement",
    "mitigate",
}

# Cụm bắt đầu yếu: nếu segment sau bắt đầu bằng các cụm này thì nhiều khả năng là phần tiếp của câu trước.
WEAK_STARTS = {
    "prices",
    "price",
    "weeks",
    "months",
    "days",
    "plans",
    "routes",
    "hubs",
    "airports",
    "flights",
    "airlines",
    "fuel",
    "middle",
    "east",
    "from",
    "to",
    "of",
    "for",
    "with",
    "that",
    "which",
    "who",
    "where",
    "when",
    "why",
    "how",
    "in",
    "on",
    "at",
    "by",
    "as",
    "and",
    "or",
    "but",
    "so",
    "because",
    "little",
    "plans",
    "going",
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
    return max(0.0, float(segment.get("end", 0.0)) - float(segment.get("start", 0.0)))


def _starts_lowercase(text: str) -> bool:
    text = _clean_text(text)

    if not text:
        return False

    first = text[0]
    return first.isalpha() and first.islower()


def _ends_with_terminal_punctuation(text: str) -> bool:
    text = _clean_text(text)

    if not text:
        return False

    text = text.rstrip('"”’)]}')
    return text.endswith(TERMINAL_PUNCTUATION)


def _last_words(text: str, n: int = 3) -> str:
    words = re.findall(r"[A-Za-z']+", text.lower())
    return " ".join(words[-n:])


def _first_word(text: str) -> str:
    words = re.findall(r"[A-Za-z']+", text.lower())
    return words[0] if words else ""


def _ends_with_weak_phrase(text: str) -> bool:
    text = _clean_text(text).lower()
    text = text.rstrip(".,!?;: ")

    for n in (3, 2, 1):
        phrase = _last_words(text, n=n)
        if phrase in WEAK_ENDINGS:
            return True

    return False


def _starts_with_weak_phrase(text: str) -> bool:
    first = _first_word(text)

    if not first:
        return False

    return first in WEAK_STARTS


def _looks_like_continuation(prev_text: str, next_text: str) -> bool:
    """
    Chỉ merge khi có dấu hiệu continuation rõ ràng.

    Không còn dùng rule quá rộng: "câu trước không có dấu chấm thì merge".
    Rule đó làm news/interview bị over-merge thành segment 20-30 giây.
    """

    prev_text = _clean_text(prev_text)
    next_text = _clean_text(next_text)

    if not prev_text or not next_text:
        return False

    if prev_text.endswith((",", ";", ":")):
        return True

    if _starts_lowercase(next_text):
        return True

    if _ends_with_weak_phrase(prev_text):
        return True

    if _starts_with_weak_phrase(next_text):
        return True

    return False


def _short_unpunctuated_bridge_allowed(current: Dict[str, Any], nxt: Dict[str, Any]) -> bool:
    """
    Một số ASR không chấm câu tốt. Cho phép nối khi segment hiện tại rất ngắn,
    chưa có dấu kết thúc và segment sau có vẻ là continuation.
    Không dùng cho segment dài để tránh over-merge.
    """

    current_text = _get_source_text(current)
    next_text = _get_source_text(nxt)

    if _ends_with_terminal_punctuation(current_text):
        return False

    if _duration(current) > 5.5:
        return False

    return _starts_lowercase(next_text) or _starts_with_weak_phrase(next_text) or _ends_with_weak_phrase(current_text)


def _merge_two_segments(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    a_text = _get_source_text(a)
    b_text = _get_source_text(b)

    merged_text = _clean_text(f"{a_text} {b_text}")

    merged = dict(a)
    merged["start"] = float(a["start"])
    merged["end"] = float(b["end"])
    merged["text"] = merged_text
    merged["source_text"] = merged_text

    merged_from = []

    if "merged_from" in a:
        merged_from.extend(a["merged_from"])
    else:
        merged_from.append(
            {
                "start": float(a["start"]),
                "end": float(a["end"]),
                "text": a_text,
            }
        )

    if "merged_from" in b:
        merged_from.extend(b["merged_from"])
    else:
        merged_from.append(
            {
                "start": float(b["start"]),
                "end": float(b["end"]),
                "text": b_text,
            }
        )

    merged["merged_from"] = merged_from
    merged["merged_count"] = len(merged_from)

    return merged


def merge_segments_for_translation(
    segments: List[Dict[str, Any]],
    enabled: bool = True,
    max_merged_duration_sec: float = 16.0,
    max_merged_chars: int = 420,
    continuation_max_duration_sec: float = 18.0,
    continuation_max_chars: int = 480,
) -> List[Dict[str, Any]]:
    """
    Gộp segment trước khi dịch, nhưng không merge quá tham.

    Mục tiêu:
    - Gộp các chỗ bị cắt cụt: "rising fuel" + "prices...".
    - Tránh over-merge news/interview thành segment 20-30 giây.
    - Không cố gộp qua nhiều câu hỏi/đáp vì sẽ làm mất khả năng gán giọng từng người.
    """

    if not enabled:
        return segments

    if not segments:
        return []

    normalized_segments: List[Dict[str, Any]] = []

    for item in segments:
        new_item = dict(item)
        text = _get_source_text(new_item)
        new_item["text"] = text
        new_item["source_text"] = text
        normalized_segments.append(new_item)

    output: List[Dict[str, Any]] = []
    i = 0

    while i < len(normalized_segments):
        current = normalized_segments[i]
        i += 1

        while i < len(normalized_segments):
            nxt = normalized_segments[i]

            current_text = _get_source_text(current)
            next_text = _get_source_text(nxt)

            candidate_duration = float(nxt["end"]) - float(current["start"])
            candidate_chars = len(_clean_text(f"{current_text} {next_text}"))

            continuation = _looks_like_continuation(current_text, next_text)
            short_bridge = _short_unpunctuated_bridge_allowed(current, nxt)

            normal_merge_allowed = (
                (continuation or short_bridge)
                and candidate_duration <= float(max_merged_duration_sec)
                and candidate_chars <= int(max_merged_chars)
            )

            # Chỉ nới giới hạn khi continuation rất rõ ràng.
            clear_continuation = (
                _ends_with_weak_phrase(current_text)
                or _starts_with_weak_phrase(next_text)
                or current_text.endswith((",", ";", ":"))
            )

            continuation_merge_allowed = (
                clear_continuation
                and candidate_duration <= float(continuation_max_duration_sec)
                and candidate_chars <= int(continuation_max_chars)
            )

            if normal_merge_allowed or continuation_merge_allowed:
                current = _merge_two_segments(current, nxt)
                i += 1
                continue

            break

        output.append(current)

    for idx, item in enumerate(output, start=1):
        item["segment_id"] = idx

    return output
