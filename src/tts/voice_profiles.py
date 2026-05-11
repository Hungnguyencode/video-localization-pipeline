from __future__ import annotations

from collections import OrderedDict
from typing import Dict


VOICE_PROFILES: "OrderedDict[str, Dict[str, str]]" = OrderedDict(
    [
        (
            "Edge - Nữ Nam - HoaiMy",
            {
                "provider": "edge",
                "voice": "vi-VN-HoaiMyNeural",
                "gender": "female",
                "region": "south",
                "note": "Edge TTS, không cần API key.",
            },
        ),
        (
            "Edge - Nam Nam - NamMinh",
            {
                "provider": "edge",
                "voice": "vi-VN-NamMinhNeural",
                "gender": "male",
                "region": "south",
                "note": "Edge TTS, không cần API key.",
            },
        ),
        (
            "FPT - Nữ Bắc - Ban Mai",
            {
                "provider": "fpt",
                "voice": "fpt:banmai",
                "fpt_voice": "banmai",
                "gender": "female",
                "region": "north",
                "note": "FPT.AI TTS, cần FPT_AI_API_KEY.",
            },
        ),
        (
            "FPT - Nữ Bắc - Thu Minh",
            {
                "provider": "fpt",
                "voice": "fpt:thuminh",
                "fpt_voice": "thuminh",
                "gender": "female",
                "region": "north",
                "note": "FPT.AI TTS, cần FPT_AI_API_KEY.",
            },
        ),
        (
            "FPT - Nam Bắc - Lê Minh",
            {
                "provider": "fpt",
                "voice": "fpt:leminh",
                "fpt_voice": "leminh",
                "gender": "male",
                "region": "north",
                "note": "FPT.AI TTS, cần FPT_AI_API_KEY.",
            },
        ),
        (
            "FPT - Nữ Trung - Mỹ An",
            {
                "provider": "fpt",
                "voice": "fpt:myan",
                "fpt_voice": "myan",
                "gender": "female",
                "region": "central",
                "note": "FPT.AI TTS, cần FPT_AI_API_KEY.",
            },
        ),
        (
            "FPT - Nữ Trung - Ngọc Lam",
            {
                "provider": "fpt",
                "voice": "fpt:ngoclam",
                "fpt_voice": "ngoclam",
                "gender": "female",
                "region": "central",
                "note": "FPT.AI TTS, cần FPT_AI_API_KEY.",
            },
        ),
        (
            "FPT - Nam Trung - Gia Huy",
            {
                "provider": "fpt",
                "voice": "fpt:giahuy",
                "fpt_voice": "giahuy",
                "gender": "male",
                "region": "central",
                "note": "FPT.AI TTS, cần FPT_AI_API_KEY.",
            },
        ),
        (
            "FPT - Nữ Nam - Lan Nhi",
            {
                "provider": "fpt",
                "voice": "fpt:lannhi",
                "fpt_voice": "lannhi",
                "gender": "female",
                "region": "south",
                "note": "FPT.AI TTS, cần FPT_AI_API_KEY.",
            },
        ),
        (
            "FPT - Nữ Nam - Linh San",
            {
                "provider": "fpt",
                "voice": "fpt:linhsan",
                "fpt_voice": "linhsan",
                "gender": "female",
                "region": "south",
                "note": "FPT.AI TTS, cần FPT_AI_API_KEY.",
            },
        ),
        (
            "FPT - Nam Nam - Minh Quang",
            {
                "provider": "fpt",
                "voice": "fpt:minhquang",
                "fpt_voice": "minhquang",
                "gender": "male",
                "region": "south",
                "note": "FPT.AI TTS, cần FPT_AI_API_KEY.",
            },
        ),
    ]
)

VOICE_OPTIONS = {label: profile["voice"] for label, profile in VOICE_PROFILES.items()}

LEGACY_VOICE_LABELS = {
    "Nữ miền Bắc - HoaiMy": "Edge - Nữ Nam - HoaiMy",
    "Nam miền Bắc - NamMinh": "Edge - Nam Nam - NamMinh",
    "Nữ - HoaiMy": "Edge - Nữ Nam - HoaiMy",
    "Nam - NamMinh": "Edge - Nam Nam - NamMinh",
    "vi-VN-HoaiMyNeural": "Edge - Nữ Nam - HoaiMy",
    "vi-VN-NamMinhNeural": "Edge - Nam Nam - NamMinh",
    # Nhãn pitch cũ: chỉ map về giọng Edge gốc để dữ liệu cũ không lỗi.
    "Nữ - HoaiMy sáng hơn": "Edge - Nữ Nam - HoaiMy",
    "Nữ - HoaiMy trầm hơn": "Edge - Nữ Nam - HoaiMy",
    "Nữ - HoaiMy nhẹ": "Edge - Nữ Nam - HoaiMy",
    "Nam - NamMinh sáng hơn": "Edge - Nam Nam - NamMinh",
    "Nam - NamMinh trầm hơn": "Edge - Nam Nam - NamMinh",
    "Nam - NamMinh nhẹ": "Edge - Nam Nam - NamMinh",
}

DEFAULT_VOICE_LABEL = "Edge - Nữ Nam - HoaiMy"
DEFAULT_VOICE_VALUE = VOICE_OPTIONS[DEFAULT_VOICE_LABEL]
DEFAULT_VOICE_PITCH = "+0Hz"  # compatibility only; không còn dùng pitch giả.


def canonical_voice_label(label: str | None) -> str:
    label = str(label or "").strip()
    if label in LEGACY_VOICE_LABELS:
        return LEGACY_VOICE_LABELS[label]
    if label in VOICE_PROFILES:
        return label
    return DEFAULT_VOICE_LABEL


def get_voice_value(voice_label: str | None) -> str:
    return VOICE_OPTIONS[canonical_voice_label(voice_label)]


def get_voice_label_from_value(voice_value: str | None) -> str:
    value = str(voice_value or "").strip()

    for label, profile in VOICE_PROFILES.items():
        if profile["voice"] == value:
            return label

    if value in LEGACY_VOICE_LABELS:
        return LEGACY_VOICE_LABELS[value]

    if value.startswith("fpt:"):
        fpt_code = value.split(":", 1)[1]
        for label, profile in VOICE_PROFILES.items():
            if profile.get("fpt_voice") == fpt_code:
                return label

    return DEFAULT_VOICE_LABEL


def get_voice_provider(voice_label_or_value: str | None) -> str:
    raw = str(voice_label_or_value or "").strip()
    if raw.startswith("fpt:"):
        return "fpt"
    if raw in VOICE_PROFILES or raw in LEGACY_VOICE_LABELS:
        label = canonical_voice_label(raw)
        return VOICE_PROFILES[label].get("provider", "edge")
    return "edge"


def get_fpt_voice_code(voice_label_or_value: str | None) -> str:
    raw = str(voice_label_or_value or "").strip()
    if raw.startswith("fpt:"):
        return raw.split(":", 1)[1]
    label = canonical_voice_label(raw)
    return VOICE_PROFILES[label].get("fpt_voice", VOICE_PROFILES[label]["voice"])


def get_voice_gender(voice_label_or_value: str | None) -> str:
    raw = str(voice_label_or_value or "").strip()
    if raw.startswith("fpt:"):
        label = get_voice_label_from_value(raw)
    else:
        label = canonical_voice_label(raw)
    return VOICE_PROFILES[label].get("gender", "unknown")


def get_voice_region(voice_label_or_value: str | None) -> str:
    raw = str(voice_label_or_value or "").strip()
    if raw.startswith("fpt:"):
        label = get_voice_label_from_value(raw)
    else:
        label = canonical_voice_label(raw)
    return VOICE_PROFILES[label].get("region", "unknown")


def get_voice_pitch(voice_label: str | None) -> str:
    return DEFAULT_VOICE_PITCH
