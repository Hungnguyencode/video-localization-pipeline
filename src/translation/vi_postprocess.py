from __future__ import annotations

import re
from typing import Iterable, List, Sequence

from src.translation.glossary_loader import apply_replacements

_DANGLING_END_RE = re.compile(
    r"(?:\b(?:và|hoặc|nhưng|rằng|là|của|cho|với|từ|để|trong|vào|ở|một|có thể là|bao gồm cả)\b)\s*[.!?…]*$",
    re.IGNORECASE,
)
_CONTINUATION_START_RE = re.compile(
    r"^\s*[\"'“”‘’()\[\]]*(?:[a-zà-ỹ]|(?:trong|đến|từ|các|những|một|về|với|cho|của|ở|vào|tuần|tháng|sân bay|thời gian|tác động|hoạt động|hy vọng|lo ngại)\b)",
    re.IGNORECASE,
)
_TERMINAL_RE = re.compile(r"[.!?…]+$")


def _basic_cleanup(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("\u200b", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?…])", r"\1", text)
    text = re.sub(r"([,;:])(?=\S)", r"\1 ", text)
    text = re.sub(r"\.\s*\.\s*\.\s*", "… ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _should_force_final_punctuation(text: str, source_text: str | None = None) -> bool:
    if not text or re.search(r"[.!?…]$", text):
        return False
    if _CONTINUATION_START_RE.search(text) or _DANGLING_END_RE.search(text):
        return False
    if source_text:
        src = source_text.strip()
        # If the source segment itself has no sentence-ending punctuation, do not
        # invent a hard stop in Vietnamese. The segment merger will repair it later.
        if src and not re.search(r"[.!?…]$", src):
            return False
    return True


def postprocess_vi_text(
    text: str,
    *,
    glossary_replacements: Sequence[dict] | None = None,
    source_text: str | None = None,
    force_terminal: bool = False,
) -> str:
    text = _basic_cleanup(text)
    if glossary_replacements:
        text = apply_replacements(text, glossary_replacements)
        text = _basic_cleanup(text)

    # Remove fake terminal punctuation from dangling fragments.
    if _DANGLING_END_RE.search(text):
        text = _TERMINAL_RE.sub("", text).strip()

    if force_terminal and _should_force_final_punctuation(text, source_text):
        text += "."
    return text


def postprocess_vi_segments(
    segments: Iterable[dict],
    *,
    glossary_replacements: Sequence[dict] | None = None,
    force_terminal: bool = False,
) -> List[dict]:
    out: List[dict] = []
    for seg in segments:
        item = dict(seg)
        vi = item.get("vi_text") or item.get("target_text") or item.get("text") or ""
        src = item.get("source_text") or item.get("en_text") or item.get("original_text") or ""
        cleaned = postprocess_vi_text(
            vi,
            glossary_replacements=glossary_replacements,
            source_text=src,
            force_terminal=force_terminal,
        )
        if "vi_text" in item:
            item["vi_text"] = cleaned
        elif "target_text" in item:
            item["target_text"] = cleaned
        else:
            item["text"] = cleaned
        out.append(item)
    return out

# ============================================================
# Backward compatibility for src/translation/translator.py
# translator.py cũ đang import VietnamesePostProcessor và build_vi_postprocessor
# nên cần giữ lại 2 API này để app không crash.
# ============================================================

class VietnamesePostProcessor:
    """
    Wrapper tương thích với translator.py cũ.
    Dùng các hàm postprocess hiện có trong file này để làm sạch tiếng Việt.
    """

    def __init__(self, replacements=None, enabled: bool = True, **kwargs):
        self.replacements = replacements or []
        self.enabled = enabled

    def process(self, text: str) -> str:
        if not self.enabled:
            return text or ""

        value = str(text or "")

        # Áp glossary/replacements nếu có
        try:
            value = apply_replacements(value, self.replacements)
        except Exception:
            pass

        # Nếu file đang có hàm postprocess_vi_text thì dùng
        fn = globals().get("postprocess_vi_text")
        if callable(fn):
            try:
                return fn(value)
            except TypeError:
                try:
                    return fn(value, replacements=self.replacements)
                except Exception:
                    return value
            except Exception:
                return value

        # Fallback dọn nhẹ nếu không có postprocess_vi_text
        import re
        value = re.sub(r"\s+", " ", value).strip()
        value = re.sub(r"\s+([,.;:!?…])", r"\1", value)
        value = re.sub(r"([,.;:!?…])([^\s\d])", r"\1 \2", value)
        value = re.sub(r"(\d+)\s*\.\s*(\d{3})(?!\d)", r"\1.\2", value)
        value = re.sub(r"(\d+)\s*,\s*(\d+)", r"\1,\2", value)
        return value

    def postprocess(self, text: str) -> str:
        return self.process(text)

    def __call__(self, text: str) -> str:
        return self.process(text)

    def process_segment(self, segment: dict) -> dict:
        if not isinstance(segment, dict):
            return segment

        item = dict(segment)

        # Ưu tiên các field bản dịch tiếng Việt
        for key in (
            "translated_text",
            "translation",
            "text_vi",
            "vi_text",
            "target_text",
            "text",
        ):
            if key in item and item.get(key):
                item[key] = self.process(item.get(key))
                break

        return item

    def postprocess_segment(self, segment: dict) -> dict:
        return self.process_segment(segment)

    def process_segments(self, segments):
        if not self.enabled:
            return segments

        # Nếu file đang có hàm postprocess_vi_segments thì dùng luôn
        fn = globals().get("postprocess_vi_segments")
        if callable(fn):
            try:
                return fn(segments, replacements=self.replacements)
            except TypeError:
                try:
                    return fn(segments)
                except Exception:
                    pass
            except Exception:
                pass

        return [self.process_segment(seg) for seg in (segments or [])]

    def postprocess_segments(self, segments):
        return self.process_segments(segments)


def build_vi_postprocessor(config=None, replacements=None, enabled: bool = True, **kwargs):
    """
    Factory tương thích với translator.py cũ.
    """

    final_replacements = replacements or []

    # Nếu config truyền vào có replacements thì lấy thêm
    try:
        if isinstance(config, dict):
            cfg_replacements = config.get("replacements") or config.get("glossary_replacements") or []
            if cfg_replacements:
                final_replacements = cfg_replacements
    except Exception:
        pass

    return VietnamesePostProcessor(
        replacements=final_replacements,
        enabled=enabled,
        **kwargs,
    )
