from __future__ import annotations

VOICE_OPTIONS = {
    "Edge - Nữ Việt Nam - Hoài My": "vi-VN-HoaiMyNeural",
    "Edge - Nam Việt Nam - Nam Minh": "vi-VN-NamMinhNeural",
    "FPT - Nữ Bắc - Ban Mai": "fpt:banmai",
    "FPT - Nam Bắc - Lê Minh": "fpt:leminh",
    "FPT - Nữ Bắc - Thu Minh": "fpt:thuminh",
    "FPT - Nam Nam - Minh Quang": "fpt:minhquang",
    "FPT - Nữ Nam - Linh San": "fpt:linhsan",
    "FPT - Nữ Nam - Lan Nhi": "fpt:lannhi",
    "FPT - Nữ Trung - Mỹ An": "fpt:myan",
    "FPT - Nam Trung - Gia Huy": "fpt:giahuy",
    "FPT - Nữ Trung - Ngọc Lam": "fpt:ngoclam",
}

DEFAULT_VOICE_LABEL = "FPT - Nữ Bắc - Ban Mai"
DEFAULT_VOICE_VALUE = VOICE_OPTIONS[DEFAULT_VOICE_LABEL]


def canonical_voice_label(label: str | None, value: str | None = None) -> str:
    if label and str(label).strip() in VOICE_OPTIONS:
        return str(label).strip()
    if value:
        found = get_voice_label_from_value(value)
        if found:
            return found
    return DEFAULT_VOICE_LABEL


def get_voice_value(label: str | None) -> str:
    return VOICE_OPTIONS.get(canonical_voice_label(label), DEFAULT_VOICE_VALUE)


def get_voice_label_from_value(value: str | None) -> str:
    value = str(value or "").strip()
    for label, voice_value in VOICE_OPTIONS.items():
        if voice_value == value:
            return label
    if value.startswith("fpt:"):
        return f"FPT - {value.split(':', 1)[1]}"
    return DEFAULT_VOICE_LABEL
