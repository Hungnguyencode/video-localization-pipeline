from __future__ import annotations

import re


_DIGITS = {
    0: "không",
    1: "một",
    2: "hai",
    3: "ba",
    4: "bốn",
    5: "năm",
    6: "sáu",
    7: "bảy",
    8: "tám",
    9: "chín",
}


def _read_under_1000(n: int, full: bool = False) -> str:
    if n < 0 or n > 999:
        raise ValueError("n must be in range 0..999")

    hundred = n // 100
    rest = n % 100
    ten = rest // 10
    unit = rest % 10

    parts: list[str] = []

    if hundred > 0:
        parts.append(_DIGITS[hundred])
        parts.append("trăm")
    elif full and rest > 0:
        parts.append("không")
        parts.append("trăm")

    if rest == 0:
        return " ".join(parts).strip()

    if ten == 0:
        if hundred > 0 or full:
            parts.append("lẻ")
        parts.append(_DIGITS[unit])
        return " ".join(parts).strip()

    if ten == 1:
        parts.append("mười")
    else:
        parts.append(_DIGITS[ten])
        parts.append("mươi")

    if unit == 0:
        return " ".join(parts).strip()

    if unit == 1 and ten >= 2:
        parts.append("mốt")
    elif unit == 4 and ten >= 2:
        parts.append("tư")
    elif unit == 5:
        parts.append("lăm")
    else:
        parts.append(_DIGITS[unit])

    return " ".join(parts).strip()


def number_to_vietnamese(n: int) -> str:
    if n == 0:
        return "không"

    if n < 0:
        return "âm " + number_to_vietnamese(abs(n))

    if n > 999_999_999_999:
        return str(n)

    units = [
        ("tỷ", 1_000_000_000),
        ("triệu", 1_000_000),
        ("nghìn", 1_000),
        ("", 1),
    ]

    parts: list[str] = []
    remaining = n
    started = False

    for unit_name, unit_value in units:
        group = remaining // unit_value
        remaining = remaining % unit_value

        if group == 0:
            continue

        full = started and group < 100
        group_text = _read_under_1000(group, full=full)

        if unit_name:
            parts.append(f"{group_text} {unit_name}")
        else:
            parts.append(group_text)

        started = True

    return " ".join(parts).strip()


def _clean_number_token(token: str) -> int | None:
    token = token.strip()

    # 13,000 hoặc 13.000
    if re.fullmatch(r"\d{1,3}([,.]\d{3})+", token):
        return int(token.replace(",", "").replace(".", ""))

    # 13000
    if re.fullmatch(r"\d+", token):
        value = int(token)

        # Số nhỏ như 1, 2, 10 vẫn để TTS tự đọc cũng được.
        # Nhưng từ 1000 trở lên nên chuyển sang chữ.
        if value >= 1000:
            return value

    return None


def normalize_vietnamese_numbers_for_tts(text: str) -> str:
    """
    Chuyển các số lớn trong câu tiếng Việt sang chữ để edge-tts đọc tự nhiên hơn.

    Ví dụ:
    13.000 chuyến bay -> mười ba nghìn chuyến bay
    850,000 chuyến bay -> tám trăm năm mươi nghìn chuyến bay
    """

    text = str(text or "")

    # Xử lý số có dấu phân tách nghìn hoặc số lớn liền nhau.
    pattern = r"(?<!\w)(\d{1,3}(?:[,.]\d{3})+|\d{4,12})(?!\w)"

    def repl(match: re.Match) -> str:
        token = match.group(1)
        value = _clean_number_token(token)

        if value is None:
            return token

        return number_to_vietnamese(value)

    text = re.sub(pattern, repl, text)

    # Một vài chuẩn hóa riêng cho tin tức.
    text = text.replace("13 không không không", "mười ba nghìn")

    return text