from __future__ import annotations

import re
from typing import Any, Dict, List

from src.translation.glossary_loader import (
    deduplicate_replacements,
    load_replacements_from_files,
)


class VietnamesePostProcessor:
    """
    Hậu xử lý tiếng Việt theo hướng generic.

    Nguyên tắc:
    - Không hard-code theo một video cụ thể.
    - Thuật ngữ/idiom theo lĩnh vực đưa vào glossary YAML.
    - File Python chỉ giữ logic xử lý chung.
    """

    def __init__(
        self,
        enabled: bool = True,
        style: str = "natural_spoken",
        shorten_for_tts: bool = False,
        max_chars_per_second: float = 17.0,
        replacements: List[List[str]] | None = None,
        glossary_files: List[str] | None = None,
    ):
        self.enabled = bool(enabled)
        self.style = style
        self.shorten_for_tts = bool(shorten_for_tts)
        self.max_chars_per_second = float(max_chars_per_second)

        glossary_replacements = load_replacements_from_files(glossary_files or [])
        config_replacements = replacements or []

        self.replacements = deduplicate_replacements(
            glossary_replacements + config_replacements
        )

    def _fix_number_separators(self, text: str) -> str:
        """
        Sửa lỗi số bị tách khoảng trắng:
        13. 000 -> 13.000
        850. 000 -> 850.000
        13 , 000 -> 13,000
        """

        text = str(text or "")

        for _ in range(3):
            text = re.sub(
                r"(?<=\d)\s*([,.])\s*(?=\d{3}(\D|$))",
                r"\1",
                text,
            )

        return text

    def normalize_text(self, text: str) -> str:
        text = (text or "").strip()

        if not text:
            return ""

        text = self._fix_number_separators(text)
        text = re.sub(r"\s+", " ", text)

        # Xóa khoảng trắng trước dấu câu, nhưng không phá số 13.000.
        text = re.sub(r"(?<!\d)\s+([,.!?;:])", r"\1", text)

        # Thêm khoảng trắng sau dấu câu, nhưng không thêm vào giữa số 13.000.
        text = re.sub(r"([,.!?;:])(?!\d)([^\s])", r"\1 \2", text)

        text = self._fix_number_separators(text)
        text = re.sub(r"\s+", " ", text)

        # Dọn lỗi dấu câu thừa hay gặp sau dịch/merge.
        text = text.replace(",.", ".")
        text = text.replace(" ,", ",")
        text = text.replace(" .", ".")

        return text.strip()

    def apply_replacements(self, text: str) -> str:
        sorted_rules = sorted(
            self.replacements,
            key=lambda pair: len(str(pair[0])),
            reverse=True,
        )

        for src, dst in sorted_rules:
            src = str(src).strip()
            dst = str(dst).strip()

            if not src:
                continue

            text = text.replace(src, dst)

        return text

    def soften_spoken_style(self, text: str) -> str:
        if self.style != "natural_spoken":
            return text

        general_rules = [
            ("Chỉ cần đảm bảo rằng", "Chỉ cần nhớ rằng"),
            ("Trên thực tế,", "Thật ra,"),
            ("Thành thật mà nói,", "Thật ra,"),
            ("Từ kinh nghiệm của tôi,", "Theo kinh nghiệm của mình,"),
            ("Tôi khuyên bạn nên", "Bạn nên"),
            ("Tôi sử dụng", "Mình dùng"),
            ("Tôi đã từng", "Mình từng"),
            ("Tôi hơi", "Mình hơi"),
            ("video của tôi", "video của mình"),
            ("cho tôi biết", "cho mình biết"),
            ("Cảm ơn đã xem", "Cảm ơn mọi người đã xem"),
        ]

        for src, dst in general_rules:
            text = text.replace(src, dst)

        return text

    def shorten_text_for_tts(self, text: str, duration_sec: float | None) -> str:
        if not self.shorten_for_tts or duration_sec is None or duration_sec <= 0:
            return text

        max_chars = int(duration_sec * self.max_chars_per_second)

        if len(text) <= max_chars:
            return text

        generic_rules = [
            ("một việc rất phức tạp", "việc khó"),
            ("thực sự", ""),
            ("rất quan trọng", "quan trọng"),
            ("một cách nhanh chóng", "nhanh"),
            ("vào cùng một lúc", "một lúc"),
            ("khoảng ", ""),
        ]

        shortened = text

        for src, dst in generic_rules:
            shortened = shortened.replace(src, dst)
            shortened = self.normalize_text(shortened)

            if len(shortened) <= max_chars:
                break

        return shortened

    def ensure_final_punctuation(self, text: str) -> str:
        if not text:
            return text

        if text[-1] not in ".!?…":
            text += "."

        return text

    def postprocess_text(
        self,
        vi_text: str,
        source_text: str | None = None,
        duration_sec: float | None = None,
    ) -> str:
        if not self.enabled:
            return vi_text

        text = self.normalize_text(vi_text)
        text = self.apply_replacements(text)
        text = self.soften_spoken_style(text)
        text = self.shorten_text_for_tts(text, duration_sec)
        text = self.normalize_text(text)
        text = self.ensure_final_punctuation(text)

        return text

    def postprocess_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []

        for item in segments:
            new_item = dict(item)

            raw_vi_text = new_item.get("vi_text", "")
            source_text = new_item.get("source_text") or new_item.get("text") or ""

            start = float(new_item.get("start", 0.0))
            end = float(new_item.get("end", start))
            duration_sec = max(0.0, end - start)

            processed_vi_text = self.postprocess_text(
                vi_text=raw_vi_text,
                source_text=source_text,
                duration_sec=duration_sec,
            )

            new_item["raw_vi_text"] = raw_vi_text
            new_item["vi_text"] = processed_vi_text

            output.append(new_item)

        return output


def build_vi_postprocessor(config: Dict[str, Any]) -> VietnamesePostProcessor:
    cfg = config.get("postprocess", {})

    return VietnamesePostProcessor(
        enabled=cfg.get("enabled", True),
        style=cfg.get("style", "natural_spoken"),
        shorten_for_tts=cfg.get("shorten_for_tts", False),
        max_chars_per_second=cfg.get("max_chars_per_second", 17.0),
        replacements=cfg.get("replacements", []),
        glossary_files=cfg.get("glossary_files", []),
    )
