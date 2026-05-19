from __future__ import annotations

from pathlib import Path
from typing import List

import yaml
import re


Replacement = List[str]


def _normalize_replacement_pair(pair) -> Replacement | None:
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return None

    src = str(pair[0]).strip()
    dst = str(pair[1]).strip()

    if not src:
        return None

    return [src, dst]


def load_replacements_from_file(path: str | Path) -> List[Replacement]:
    file_path = Path(path)

    if not file_path.exists():
        print(f"[GLOSSARY] File not found, skip: {file_path}")
        return []

    with file_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_replacements = data.get("replacements", [])

    replacements: List[Replacement] = []

    for pair in raw_replacements:
        normalized = _normalize_replacement_pair(pair)
        if normalized:
            replacements.append(normalized)

    return replacements


def load_replacements_from_files(paths: list[str | Path]) -> List[Replacement]:
    all_replacements: List[Replacement] = []

    for path in paths:
        all_replacements.extend(load_replacements_from_file(path))

    return deduplicate_replacements(all_replacements)


def deduplicate_replacements(replacements: List[Replacement]) -> List[Replacement]:
    seen = set()
    output: List[Replacement] = []

    for pair in replacements:
        normalized = _normalize_replacement_pair(pair)
        if not normalized:
            continue

        src, dst = normalized
        key = src.lower()

        if key in seen:
            continue

        seen.add(key)
        output.append([src, dst])

    return output


def parse_quick_replacements(text: str) -> List[Replacement]:
    """
    Parse quick replacements từ UI.

    Hỗ trợ:
    rocket science => nghe có vẻ khó
    don't sweat it -> đừng lo
    pancake, pancake
    """

    text = text or ""
    replacements: List[Replacement] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if "=>" in line:
            src, dst = line.split("=>", 1)
        elif "->" in line:
            src, dst = line.split("->", 1)
        elif "," in line:
            src, dst = line.split(",", 1)
        else:
            continue

        normalized = _normalize_replacement_pair([src, dst])
        if normalized:
            replacements.append(normalized)

    return deduplicate_replacements(replacements)


def apply_replacements(text: str, replacements=None) -> str:
    """
    Áp dụng danh sách thay thế từ glossary/replacements lên text tiếng Việt.
    Hàm này được vi_postprocess.py gọi để sửa các lỗi tên riêng/cụm từ.
    Nếu không có replacements thì trả nguyên text, không làm hỏng pipeline.
    """
    if text is None:
        return ""

    result = str(text)

    if not replacements:
        return result

    for item in replacements:
        if not item:
            continue

        src = None
        dst = ""

        regex = False
        ignore_case = False

        if isinstance(item, dict):
            src = (
                item.get("from")
                or item.get("source")
                or item.get("old")
                or item.get("pattern")
            )
            dst = (
                item.get("to")
                or item.get("target")
                or item.get("new")
                or item.get("replacement")
                or ""
            )
            regex = bool(item.get("regex", False))
            ignore_case = bool(item.get("ignore_case", False))

        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            src = item[0]
            dst = item[1]

        if not src:
            continue

        flags = re.IGNORECASE if ignore_case else 0

        try:
            if regex:
                result = re.sub(str(src), str(dst), result, flags=flags)
            else:
                if ignore_case:
                    result = re.sub(re.escape(str(src)), str(dst), result, flags=flags)
                else:
                    result = result.replace(str(src), str(dst))
        except Exception:
            # Không để glossary làm hỏng toàn pipeline
            continue

    return result