from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env", override=True)

from main_pipeline import VideoLocalizationPipeline
from src.tts.voice_profiles import (
    DEFAULT_VOICE_LABEL,
    DEFAULT_VOICE_VALUE,
    VOICE_OPTIONS,
    canonical_voice_label,
    get_voice_label_from_value as get_voice_label,
    get_voice_value,
)
from src.ingest.youtube_video import YouTubeVideoDownloader
from src.translation.glossary_loader import (
    deduplicate_replacements,
    load_replacements_from_files,
    parse_quick_replacements,
)
from src.utils.io import load_json, save_json


st.set_page_config(
    page_title="Video Localization Pipeline",
    page_icon="🎬",
    layout="wide",
)


# =========================
# Options
# =========================

# VOICE_OPTIONS được lấy từ src/tts/voice_profiles.py.
# File đó chỉ giữ giọng thật: Edge TTS + FPT.AI, không dùng pitch giả nữa.

RATE_OPTIONS = {
    "Chậm hơn (-10%)": "-10%",
    "Mặc định": "+0%",
    "Nhanh hơn (+10%)": "+10%",
    "Nhanh hơn (+15%)": "+15%",
    "Nhanh hơn (+20%)": "+20%",
}
RATE_REVERSE_OPTIONS = {value: label for label, value in RATE_OPTIONS.items()}
DEFAULT_RATE_LABEL = "Mặc định"
DEFAULT_RATE_VALUE = RATE_OPTIONS[DEFAULT_RATE_LABEL]

AUDIO_MODE_OPTIONS = {
    "Thay audio gốc bằng giọng Việt": "replace",
    "Giữ âm gốc nhỏ + chèn giọng Việt": "mix_low_original",
    "Giữ nguyên audio gốc + chỉ chèn phụ đề tiếng Việt": "subtitle_only",
}

VOICE_RENDER_MODE_OPTIONS = {
    "Giữ giọng đã gán thủ công, còn lại dùng giọng mặc định": "manual_then_default",
    "Giữ nguyên voice/rate đang có trong bảng": "keep_table",
    "Ép toàn bộ video dùng giọng mặc định": "force_default",
}

FONT_OPTIONS = ["Arial", "Tahoma", "Verdana", "Times New Roman"]

GLOSSARY_DOMAIN_OPTIONS = {
    "Chung": ["configs/glossaries/common_vi.yaml"],
    "Nấu ăn": ["configs/glossaries/common_vi.yaml", "configs/glossaries/cooking_vi.yaml"],
    "Giáo dục": ["configs/glossaries/common_vi.yaml", "configs/glossaries/education_vi.yaml"],
    "Tin tức": ["configs/glossaries/common_vi.yaml", "configs/glossaries/news_vi.yaml"],
    "Công nghệ / AI": ["configs/glossaries/common_vi.yaml", "configs/glossaries/technology_vi.yaml"],
}

SPEAKER_ROLE_OPTIONS = [
    "Mặc định",
    "Người dẫn / Speaker A",
    "Khách mời / Speaker B",
    "Speaker C",
    "Tùy chỉnh",
]


# =========================
# Pipeline
# =========================

@st.cache_resource
def get_pipeline() -> VideoLocalizationPipeline:
    return VideoLocalizationPipeline()


# =========================
# Basic helpers
# =========================

def canonical_rate_label(rate_label: str | None, rate_value: str | None = None) -> str:
    if rate_label and str(rate_label).strip() in RATE_OPTIONS:
        return str(rate_label).strip()
    if rate_value and str(rate_value).strip() in RATE_REVERSE_OPTIONS:
        return RATE_REVERSE_OPTIONS[str(rate_value).strip()]
    return DEFAULT_RATE_LABEL


def get_rate_value(rate_label: str | None) -> str:
    return RATE_OPTIONS.get(canonical_rate_label(rate_label), DEFAULT_RATE_VALUE)


def rate_rank(rate_value: str) -> int:
    order = {"-10%": -1, "+0%": 0, "+10%": 10, "+15%": 15, "+20%": 20}
    return order.get(str(rate_value).strip(), 0)


def max_rate(a: str, b: str) -> str:
    return a if rate_rank(a) >= rate_rank(b) else b


# =========================
# Glossary + YouTube helpers
# =========================

def apply_runtime_glossary_to_pipeline(
    pipeline: VideoLocalizationPipeline,
    glossary_files: list[str],
    quick_replacement_text: str,
) -> list[list[str]]:
    glossary_replacements = load_replacements_from_files(glossary_files)
    quick_replacements = parse_quick_replacements(quick_replacement_text)
    final_replacements = deduplicate_replacements(glossary_replacements + quick_replacements)

    translator = pipeline.translator
    postprocessor = getattr(translator, "postprocessor", None)
    if postprocessor is not None:
        postprocessor.replacements = final_replacements

    return final_replacements


def save_glossary_metadata_to_bilingual(
    bilingual_path: str | Path,
    domain_name: str,
    glossary_files: list[str],
    replacements: list[list[str]],
) -> None:
    data = load_json(bilingual_path)
    data["glossary"] = {
        "domain": domain_name,
        "files": glossary_files,
        "replacements_count": len(replacements),
        "replacements": replacements,
    }
    save_json(data, bilingual_path)


def download_youtube_to_input(url: str, max_height: int) -> Dict[str, Any]:
    input_dir = PROJECT_ROOT / "data" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    downloader = YouTubeVideoDownloader(output_dir=input_dir)
    return downloader.download(url=url, max_height=max_height)


# =========================
# Dataframe / editing helpers
# =========================

def estimate_tts_risk(text: str, duration_sec: float) -> tuple[str, float]:
    text = str(text or "").strip()
    duration_sec = max(float(duration_sec), 0.1)
    chars_per_sec = len(text) / duration_sec
    if chars_per_sec <= 14:
        return "OK", chars_per_sec
    if chars_per_sec <= 19:
        return "Hơi dài", chars_per_sec
    return "Quá dài", chars_per_sec


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def build_editor_dataframe(bilingual_data: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for idx, item in enumerate(bilingual_data.get("segments", []), start=1):
        start = _safe_float(item.get("start"), 0.0)
        end = _safe_float(item.get("end"), start + 0.1)
        duration = max(end - start, 0.1)
        vi_text = str(item.get("vi_text", "") or "")
        risk, cps = estimate_tts_risk(vi_text, duration)

        voice_value = item.get("voice") or item.get("tts_voice") or DEFAULT_VOICE_VALUE
        voice_label = item.get("voice_label") or get_voice_label(voice_value)
        voice_label = canonical_voice_label(voice_label)

        rate_value = item.get("rate") or item.get("tts_rate") or DEFAULT_RATE_VALUE
        rate_label = canonical_rate_label(item.get("rate_label"), rate_value)

        speaker_role = str(item.get("speaker_role") or "Mặc định")
        if speaker_role not in SPEAKER_ROLE_OPTIONS:
            speaker_role = "Tùy chỉnh"

        rows.append(
            {
                "segment_id": int(item.get("id") or item.get("segment_id") or idx),
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(duration, 2),
                "tts_risk": risk,
                "chars_per_sec": round(cps, 1),
                "speaker_role": speaker_role,
                "voice_label": voice_label,
                "voice": get_voice_value(voice_label),
                "voice_manual": bool(item.get("voice_manual", False)),
                "rate_label": rate_label,
                "rate": get_rate_value(rate_label),
                "rate_manual": bool(item.get("rate_manual", False)),
                "source_text": item.get("source_text") or item.get("text") or "",
                "raw_vi_text": item.get("raw_vi_text", item.get("vi_text", "")),
                "vi_text": vi_text,
            }
        )
    return normalize_editor_dataframe(pd.DataFrame(rows))


def refresh_risk_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    risks, cps_values = [], []
    for _, row in df.iterrows():
        risk, cps = estimate_tts_risk(row.get("vi_text", ""), _safe_float(row.get("duration"), 0.1))
        risks.append(risk)
        cps_values.append(round(cps, 1))
    df["tts_risk"] = risks
    df["chars_per_sec"] = cps_values
    return df


def normalize_editor_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    output = df.copy()
    for col, default in [
        ("source_text", ""),
        ("raw_vi_text", ""),
        ("vi_text", ""),
        ("speaker_role", "Mặc định"),
        ("voice_label", DEFAULT_VOICE_LABEL),
        ("rate_label", DEFAULT_RATE_LABEL),
    ]:
        if col not in output.columns:
            output[col] = default
    if "voice_manual" not in output.columns:
        output["voice_manual"] = False
    if "rate_manual" not in output.columns:
        output["rate_manual"] = False

    output["start"] = output["start"].astype(float)
    output["end"] = output["end"].astype(float)
    output = output.sort_values(["start", "end"]).reset_index(drop=True)
    output["segment_id"] = range(1, len(output) + 1)
    output["duration"] = (output["end"] - output["start"]).clip(lower=0.1).round(2)
    output["voice_label"] = output["voice_label"].apply(canonical_voice_label)
    output["voice"] = output["voice_label"].apply(get_voice_value)
    output["rate_label"] = output.apply(lambda r: canonical_rate_label(r.get("rate_label"), r.get("rate")), axis=1)
    output["rate"] = output["rate_label"].apply(get_rate_value)
    output["voice_manual"] = output["voice_manual"].astype(bool)
    output["rate_manual"] = output["rate_manual"].astype(bool)
    output = refresh_risk_columns(output)
    return output


def get_changed_count(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    return int((df["raw_vi_text"].astype(str).str.strip() != df["vi_text"].astype(str).str.strip()).sum())


def filter_editor_dataframe(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    keyword = (keyword or "").strip().lower()
    if not keyword:
        return df
    mask = (
        df["segment_id"].astype(str).str.contains(keyword, na=False)
        | df["source_text"].astype(str).str.lower().str.contains(keyword, na=False)
        | df["raw_vi_text"].astype(str).str.lower().str.contains(keyword, na=False)
        | df["vi_text"].astype(str).str.lower().str.contains(keyword, na=False)
        | df["tts_risk"].astype(str).str.lower().str.contains(keyword, na=False)
        | df["voice_label"].astype(str).str.lower().str.contains(keyword, na=False)
        | df["speaker_role"].astype(str).str.lower().str.contains(keyword, na=False)
    )
    return df[mask].copy()


def parse_segment_ranges(range_text: str) -> set[int]:
    result: set[int] = set()
    for part in str(range_text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text.strip()), int(end_text.strip())
            if start > end:
                start, end = end, start
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return result


def split_text_suggestion(text: str) -> tuple[str, str]:
    text = str(text or "").strip()
    if not text:
        return "", ""
    # Ưu tiên tách theo dấu câu gần giữa.
    mid = len(text) // 2
    candidates = [m.end() for m in re.finditer(r"[.!?。]\s+", text)]
    if candidates:
        best = min(candidates, key=lambda p: abs(p - mid))
        return text[:best].strip(), text[best:].strip()
    # Nếu không có dấu câu, tách theo khoảng trắng gần giữa.
    left = text.rfind(" ", 0, mid)
    right = text.find(" ", mid)
    cut = left if left > 0 and (right < 0 or mid - left <= right - mid) else right
    if cut <= 0:
        cut = mid
    return text[:cut].strip(), text[cut:].strip()


def apply_voice_to_segment_ranges(range_text: str, voice_label: str, speaker_role: str = "Tùy chỉnh") -> int:
    df = st.session_state.get("editor_df")
    if df is None or df.empty:
        return 0
    segment_ids = parse_segment_ranges(range_text)
    voice_label = canonical_voice_label(voice_label)
    mask = df["segment_id"].astype(int).isin(segment_ids)
    count = int(mask.sum())
    if count > 0:
        st.session_state["editor_df"].loc[mask, "voice_label"] = voice_label
        st.session_state["editor_df"].loc[mask, "voice"] = get_voice_value(voice_label)
        st.session_state["editor_df"].loc[mask, "speaker_role"] = speaker_role if speaker_role in SPEAKER_ROLE_OPTIONS else "Tùy chỉnh"
        st.session_state["editor_df"].loc[mask, "voice_manual"] = True
        st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
    return count


def apply_role_to_segment_ranges(range_text: str, speaker_role: str, role_voice_map: dict[str, str]) -> int:
    if speaker_role not in SPEAKER_ROLE_OPTIONS or speaker_role == "Mặc định":
        raise ValueError("Chọn một speaker role cụ thể, không chọn Mặc định.")
    voice_label = role_voice_map.get(speaker_role, DEFAULT_VOICE_LABEL)
    return apply_voice_to_segment_ranges(range_text, voice_label, speaker_role=speaker_role)


def update_segment_text(segment_id: int, new_text: str) -> None:
    df = st.session_state.get("editor_df")
    if df is None or df.empty:
        return
    mask = df["segment_id"] == int(segment_id)
    if mask.any():
        st.session_state["editor_df"].loc[mask, "vi_text"] = str(new_text).strip()
        st.session_state["editor_df"] = normalize_editor_dataframe(st.session_state["editor_df"])
        st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1


def update_segment_voice(segment_id: int, voice_label: str, speaker_role: str | None = None) -> None:
    df = st.session_state.get("editor_df")
    if df is None or df.empty:
        return
    mask = df["segment_id"] == int(segment_id)
    if mask.any():
        voice_label = canonical_voice_label(voice_label)
        st.session_state["editor_df"].loc[mask, "voice_label"] = voice_label
        st.session_state["editor_df"].loc[mask, "voice"] = get_voice_value(voice_label)
        st.session_state["editor_df"].loc[mask, "voice_manual"] = True
        if speaker_role:
            st.session_state["editor_df"].loc[mask, "speaker_role"] = speaker_role
        st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1


def update_segment_rate(segment_id: int, rate_label: str, manual: bool = True) -> None:
    df = st.session_state.get("editor_df")
    if df is None or df.empty:
        return
    mask = df["segment_id"] == int(segment_id)
    if mask.any():
        rate_label = canonical_rate_label(rate_label)
        st.session_state["editor_df"].loc[mask, "rate_label"] = rate_label
        st.session_state["editor_df"].loc[mask, "rate"] = get_rate_value(rate_label)
        st.session_state["editor_df"].loc[mask, "rate_manual"] = bool(manual)
        st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1


def merge_segment_ranges(range_text: str, voice_strategy: str = "first") -> int:
    df = st.session_state.get("editor_df")
    if df is None or df.empty:
        return 0
    ids = sorted(parse_segment_ranges(range_text))
    if len(ids) < 2:
        raise ValueError("Cần chọn ít nhất 2 segment để gộp.")
    selected = df[df["segment_id"].astype(int).isin(ids)].copy().sort_values("start")
    if len(selected) < 2:
        return 0
    first = selected.iloc[0]
    last = selected.iloc[-1]
    if voice_strategy == "last":
        voice_row = last
    else:
        voice_row = first
    merged_row = first.copy()
    merged_row["start"] = float(first["start"])
    merged_row["end"] = float(last["end"])
    merged_row["source_text"] = " ".join(selected["source_text"].astype(str).str.strip()).strip()
    merged_row["raw_vi_text"] = " ".join(selected["raw_vi_text"].astype(str).str.strip()).strip()
    merged_row["vi_text"] = " ".join(selected["vi_text"].astype(str).str.strip()).strip()
    merged_row["voice_label"] = voice_row.get("voice_label", DEFAULT_VOICE_LABEL)
    merged_row["voice"] = get_voice_value(merged_row["voice_label"])
    merged_row["speaker_role"] = voice_row.get("speaker_role", "Mặc định")
    merged_row["voice_manual"] = bool(voice_row.get("voice_manual", False))
    merged_row["rate_label"] = voice_row.get("rate_label", DEFAULT_RATE_LABEL)
    merged_row["rate"] = get_rate_value(merged_row["rate_label"])
    merged_row["rate_manual"] = bool(voice_row.get("rate_manual", False))

    remaining = df[~df["segment_id"].astype(int).isin(ids)].copy()
    new_df = pd.concat([remaining, pd.DataFrame([merged_row])], ignore_index=True)
    st.session_state["editor_df"] = normalize_editor_dataframe(new_df)
    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
    return len(selected)


def split_segment(
    segment_id: int,
    split_time: float,
    source_part_1: str,
    source_part_2: str,
    vi_part_1: str,
    vi_part_2: str,
) -> None:
    df = st.session_state.get("editor_df")
    if df is None or df.empty:
        return
    mask = df["segment_id"] == int(segment_id)
    if not mask.any():
        raise ValueError("Không tìm thấy segment cần tách.")
    row = df[mask].iloc[0].copy()
    start, end = float(row["start"]), float(row["end"])
    split_time = float(split_time)
    if split_time <= start or split_time >= end:
        raise ValueError("Thời điểm tách phải nằm giữa start và end của segment.")

    row1 = row.copy()
    row2 = row.copy()
    row1["end"] = split_time
    row2["start"] = split_time
    row1["source_text"] = source_part_1.strip()
    row2["source_text"] = source_part_2.strip()
    row1["raw_vi_text"] = vi_part_1.strip()
    row2["raw_vi_text"] = vi_part_2.strip()
    row1["vi_text"] = vi_part_1.strip()
    row2["vi_text"] = vi_part_2.strip()

    new_df = pd.concat([df[~mask].copy(), pd.DataFrame([row1, row2])], ignore_index=True)
    st.session_state["editor_df"] = normalize_editor_dataframe(new_df)
    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1


def apply_auto_rate(df: pd.DataFrame, default_rate_label: str) -> pd.DataFrame:
    output = normalize_editor_dataframe(df)
    default_rate = get_rate_value(default_rate_label)
    for idx, row in output.iterrows():
        if bool(row.get("rate_manual", False)):
            continue
        cps = _safe_float(row.get("chars_per_sec"), 0.0)
        if cps <= 14:
            target = default_rate
        elif cps <= 19:
            target = max_rate(default_rate, "+10%")
        elif cps <= 23:
            target = max_rate(default_rate, "+15%")
        else:
            target = max_rate(default_rate, "+20%")
        output.at[idx, "rate"] = target
        output.at[idx, "rate_label"] = RATE_REVERSE_OPTIONS.get(target, DEFAULT_RATE_LABEL)
    return output


def build_role_voice_map(role_a_voice: str, role_b_voice: str, role_c_voice: str) -> dict[str, str]:
    return {
        "Người dẫn / Speaker A": canonical_voice_label(role_a_voice),
        "Khách mời / Speaker B": canonical_voice_label(role_b_voice),
        "Speaker C": canonical_voice_label(role_c_voice),
    }


def build_render_dataframe(
    df: pd.DataFrame,
    default_voice_label: str,
    default_rate_label: str,
    voice_render_mode: str,
    auto_rate_enabled: bool,
    role_voice_map: dict[str, str],
) -> pd.DataFrame:
    render_df = normalize_editor_dataframe(df)
    default_voice_label = canonical_voice_label(default_voice_label)

    for idx, row in render_df.iterrows():
        role = str(row.get("speaker_role", "Mặc định"))
        manual = bool(row.get("voice_manual", False))
        if voice_render_mode == "force_default":
            voice_label = default_voice_label
            manual = False
        elif voice_render_mode == "manual_then_default":
            if manual:
                voice_label = role_voice_map.get(role, row.get("voice_label", default_voice_label))
            else:
                voice_label = default_voice_label
        else:  # keep_table
            voice_label = role_voice_map.get(role, row.get("voice_label", default_voice_label)) if role != "Mặc định" else row.get("voice_label", default_voice_label)
        voice_label = canonical_voice_label(voice_label)
        render_df.at[idx, "voice_label"] = voice_label
        render_df.at[idx, "voice"] = get_voice_value(voice_label)
        render_df.at[idx, "voice_manual"] = manual

    if auto_rate_enabled:
        render_df = apply_auto_rate(render_df, default_rate_label)
    else:
        default_rate = get_rate_value(default_rate_label)
        for idx, row in render_df.iterrows():
            if not bool(row.get("rate_manual", False)):
                render_df.at[idx, "rate"] = default_rate
                render_df.at[idx, "rate_label"] = RATE_REVERSE_OPTIONS.get(default_rate, DEFAULT_RATE_LABEL)
    return normalize_editor_dataframe(render_df)


def save_edited_bilingual(bilingual_path: str | Path, edited_df: pd.DataFrame) -> str:
    bilingual_path = Path(bilingual_path)
    bilingual_data = load_json(bilingual_path)
    normalized_df = normalize_editor_dataframe(edited_df)
    edited_segments = []
    for _, row in normalized_df.iterrows():
        source_text = str(row.get("source_text", "")).strip()
        raw_vi_text = str(row.get("raw_vi_text", "")).strip()
        vi_text = str(row.get("vi_text", "")).strip()
        voice_label = canonical_voice_label(row.get("voice_label"))
        rate_label = canonical_rate_label(row.get("rate_label"), row.get("rate"))
        start = _safe_float(row.get("start"), 0.0)
        end = _safe_float(row.get("end"), start + 0.1)
        edited_segments.append(
            {
                "id": int(row.get("segment_id", len(edited_segments) + 1)),
                "segment_id": int(row.get("segment_id", len(edited_segments) + 1)),
                "start": start,
                "end": end,
                "text": source_text,
                "source_text": source_text,
                "raw_vi_text": raw_vi_text,
                "vi_text": vi_text,
                "speaker_role": str(row.get("speaker_role", "Mặc định")),
                "voice": get_voice_value(voice_label),
                "voice_label": voice_label,
                "voice_manual": bool(row.get("voice_manual", False)),
                "rate": get_rate_value(rate_label),
                "rate_label": rate_label,
                "rate_manual": bool(row.get("rate_manual", False)),
                "manual_edited": vi_text != raw_vi_text,
            }
        )

    edited_data = dict(bilingual_data)
    edited_data["segments"] = edited_segments
    edited_data["edited"] = True
    edited_data["edit_note"] = "Edited from Streamlit human-in-the-loop interface."
    video_stem = edited_data.get("video_stem") or Path(edited_data["video_name"]).stem
    edited_path = PROJECT_ROOT / "data" / "transcripts" / f"{video_stem}_bilingual_edited.json"
    save_json(edited_data, edited_path)
    return str(edited_path)


# =========================
# Quality checker + report
# =========================

def quality_check_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df is None or df.empty:
        return pd.DataFrame(rows)
    weak_endings = ("vì", "để", "từ", "trong", "của", "thực hiện", "một", "và", "nhưng", "rằng")
    for _, row in normalize_editor_dataframe(df).iterrows():
        sid = int(row["segment_id"])
        duration = _safe_float(row["duration"], 0.0)
        cps = _safe_float(row["chars_per_sec"], 0.0)
        vi = str(row.get("vi_text", "")).strip()
        source = str(row.get("source_text", "")).strip()
        def add(level: str, issue: str, suggestion: str):
            rows.append({"segment_id": sid, "level": level, "issue": issue, "suggestion": suggestion})
        if duration >= 16:
            add("Cảnh báo", f"Segment dài {duration:.1f}s", "Nên tách segment để dễ gán giọng và giảm đùn audio.")
        if cps > 23:
            add("Nặng", f"TTS rất dài: {cps:.1f} ký tự/s", "Rút gọn câu hoặc bật auto speed +20%.")
        elif cps > 19:
            add("Cảnh báo", f"TTS dài: {cps:.1f} ký tự/s", "Bật auto speed hoặc rút gọn câu.")
        if vi and vi[-1] not in ".!?…":
            add("Nhẹ", "Thiếu dấu kết thúc câu", "Thêm dấu chấm/hỏi/chấm than để TTS ngắt tự nhiên.")
        if vi.lower().rstrip(" .,!?").endswith(weak_endings):
            add("Cảnh báo", "Câu có vẻ kết thúc cụt", "Cân nhắc gộp với segment sau hoặc sửa lại câu.")
        if re.search(r"\d+\s+[,.]\s+\d{3}", vi):
            add("Cảnh báo", "Số bị tách khoảng trắng", "Sửa 13. 000 thành 13.000 hoặc nhập chữ.")
        if vi and vi[0].islower():
            add("Nhẹ", "Bắt đầu bằng chữ thường", "Có thể là câu nối từ segment trước; cân nhắc gộp.")
        if source and not re.search(r"[.!?…]['\")\]]*$", source) and duration < 6:
            add("Nhẹ", "Source có thể bị cắt giữa câu", "Kiểm tra segment kế tiếp để gộp nếu cần.")
    return pd.DataFrame(rows)


def create_demo_report(
    render_result: Dict[str, Any],
    editor_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    selected_domain: str,
    replacements_count: int,
) -> dict[str, str]:
    output_dir = PROJECT_ROOT / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_stem = render_result.get("video_stem") or Path(render_result.get("video_name", "video")).stem
    normalized = normalize_editor_dataframe(editor_df)
    voices = normalized["voice_label"].value_counts().to_dict() if not normalized.empty else {}
    roles = normalized["speaker_role"].value_counts().to_dict() if "speaker_role" in normalized.columns else {}
    rates = normalized["rate_label"].value_counts().to_dict() if "rate_label" in normalized.columns else {}

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "video_name": render_result.get("video_name"),
        "video_duration_sec": render_result.get("video_duration_sec"),
        "segments_count": int(len(normalized)),
        "manual_edited_segments": int(get_changed_count(normalized)),
        "too_long_segments": int((normalized["tts_risk"] == "Quá dài").sum()) if not normalized.empty else 0,
        "long_segments": int((normalized["tts_risk"] == "Hơi dài").sum()) if not normalized.empty else 0,
        "quality_issues": int(len(quality_df)),
        "translation_domain": selected_domain,
        "replacements_count": int(replacements_count),
        "voices": voices,
        "speaker_roles": roles,
        "rates": rates,
        "audio_mode": render_result.get("audio_mode") or render_result.get("selected_audio_mode"),
        "burn_subtitle": render_result.get("burn_subtitle") or render_result.get("selected_burn_subtitle"),
        "output_video_path": render_result.get("output_video_path"),
        "output_subtitle_path": render_result.get("output_subtitle_path"),
        "bilingual_transcript_path": render_result.get("bilingual_transcript_path"),
    }
    json_path = output_dir / f"{video_stem}_demo_report.json"
    html_path = output_dir / f"{video_stem}_demo_report.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    issue_rows = "".join(
        f"<tr><td>{int(r.segment_id)}</td><td>{escape(str(r.level))}</td><td>{escape(str(r.issue))}</td><td>{escape(str(r.suggestion))}</td></tr>"
        for r in quality_df.itertuples()
    ) or "<tr><td colspan='4'>Không có cảnh báo nghiêm trọng.</td></tr>"
    html = f"""
<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><title>Video Localization Report</title>
<style>body{{font-family:Arial,sans-serif;max-width:980px;margin:32px auto;line-height:1.5}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:8px}}th{{background:#f3f3f3}}</style></head>
<body>
<h1>Video Localization Pipeline Report</h1>
<h2>Tổng quan</h2>
<table>
<tr><th>Trường</th><th>Giá trị</th></tr>
<tr><td>Video</td><td>{escape(str(report.get('video_name')))}</td></tr>
<tr><td>Thời lượng</td><td>{escape(str(report.get('video_duration_sec')))} giây</td></tr>
<tr><td>Số segment</td><td>{report['segments_count']}</td></tr>
<tr><td>Segment đã chỉnh</td><td>{report['manual_edited_segments']}</td></tr>
<tr><td>Domain dịch</td><td>{escape(str(selected_domain))}</td></tr>
<tr><td>Render mode</td><td>{escape(str(report.get('audio_mode')))}</td></tr>
<tr><td>Voices</td><td>{escape(json.dumps(voices, ensure_ascii=False))}</td></tr>
<tr><td>Rates</td><td>{escape(json.dumps(rates, ensure_ascii=False))}</td></tr>
</table>
<h2>Cảnh báo chất lượng</h2>
<table><tr><th>Segment</th><th>Mức</th><th>Vấn đề</th><th>Gợi ý</th></tr>{issue_rows}</table>
</body></html>
"""
    html_path.write_text(html, encoding="utf-8")
    return {"report_json_path": str(json_path), "report_html_path": str(html_path)}


# =========================
# Files / state helpers
# =========================

def show_download_button(path_value: str | None, label: str, mime: str) -> None:
    if not path_value:
        st.warning("Chưa có file.")
        return
    path = Path(path_value)
    if not path.exists():
        st.warning(f"Không tìm thấy file: {path}")
        return
    with path.open("rb") as f:
        st.download_button(label=label, data=f, file_name=path.name, mime=mime, use_container_width=True)


def reset_working_state() -> None:
    keys = [
        "prepare_result", "render_result", "edited_bilingual_path", "editor_df", "editor_version",
        "preview_audio_path", "youtube_download_result", "last_report_paths",
    ]
    for key in keys:
        st.session_state.pop(key, None)


def make_preview_audio(segment_id: int, text: str, voice: str, rate: str) -> str:
    pipeline = get_pipeline()
    pipeline.tts_engine.voice = voice
    pipeline.tts_engine.rate = rate
    preview_dir = PROJECT_ROOT / "data" / "tts_segments" / "_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    safe_voice = re.sub(r"[^a-zA-Z0-9_]+", "_", voice)
    safe_rate = re.sub(r"[^a-zA-Z0-9_+\-]+", "_", rate)
    output_path = preview_dir / f"segment_{segment_id:04d}_{safe_voice}_{safe_rate}.mp3"
    if output_path.exists():
        output_path.unlink()
    pipeline.tts_engine.synthesize_one(text=text, output_path=output_path, voice=voice, rate=rate, use_cache=False)
    return str(output_path)


def run_prepare_translation(input_path: Path, selected_domain: str, selected_glossary_files: list[str], quick_replacement_text: str) -> None:
    with st.spinner("Đang chạy ASR + merge segment + dịch + hậu xử lý..."):
        try:
            pipeline = get_pipeline()
            runtime_replacements = apply_runtime_glossary_to_pipeline(
                pipeline=pipeline,
                glossary_files=selected_glossary_files,
                quick_replacement_text=quick_replacement_text,
            )
            prepare_result = pipeline.prepare_translation(input_path)
            save_glossary_metadata_to_bilingual(
                bilingual_path=prepare_result["bilingual_transcript_path"],
                domain_name=selected_domain,
                glossary_files=selected_glossary_files,
                replacements=runtime_replacements,
            )
            bilingual_data = load_json(prepare_result["bilingual_transcript_path"])
            st.session_state["prepare_result"] = prepare_result
            st.session_state["editor_df"] = build_editor_dataframe(bilingual_data)
            st.session_state["editor_version"] = 0
            st.session_state["selected_domain"] = selected_domain
            st.session_state["selected_glossary_files"] = selected_glossary_files
            st.session_state["runtime_replacements_count"] = len(runtime_replacements)
            st.session_state.pop("render_result", None)
            st.session_state.pop("edited_bilingual_path", None)
            st.session_state.pop("preview_audio_path", None)
            st.success("Đã tạo xong transcript và bản dịch. Bạn có thể chỉnh sửa bên dưới.")
            st.rerun()
        except Exception as e:
            st.error("Lỗi ở bước tạo transcript/bản dịch.")
            st.exception(e)


# =========================
# UI
# =========================

st.title("🎬 Video Localization Pipeline")
st.caption("Việt hóa video ngắn bằng ASR, dịch phụ đề, hiệu chỉnh bản dịch, TTS và lồng tiếng tự động.")
st.info("Bản v1.5+FPT: split segment thủ công, speaker role, quality checker, auto speed từng segment, report demo và multi-provider TTS Edge/FPT.AI.")

with st.expander("🔑 Cấu hình FPT.AI TTS", expanded=False):
    if os.getenv("FPT_AI_API_KEY"):
        st.success("Đã phát hiện FPT_AI_API_KEY. Bạn có thể dùng các giọng FPT.AI trong danh sách voice.")
    else:
        st.warning(
            "Chưa phát hiện FPT_AI_API_KEY. Các giọng Edge vẫn chạy bình thường; "
            "nếu chọn giọng FPT.AI thì cần tạo file .env ở thư mục gốc project."
        )
        st.code("FPT_AI_API_KEY=your_fpt_ai_api_key_here", language="env")
    st.caption("Không đưa API key thật lên public GitHub. Nếu đã lộ key, nên tạo/reset key mới trong FPT.AI Console.")

with st.expander("📌 Hướng dẫn nhanh", expanded=False):
    st.markdown(
        """
        **Bước 1:** Chọn glossary, chọn nguồn video, chạy transcript + dịch.  
        **Bước 2:** Kiểm tra bảng dịch, gán speaker/voice, gộp hoặc tách segment nếu cần.  
        **Bước 3:** Xem tab kiểm tra chất lượng trước khi render.  
        **Bước 4:** Render video, sau đó tải video/SRT/JSON/report.
        """
    )

# Glossary UI
st.subheader("0. Cấu hình dịch và glossary")
with st.expander("🌐 Glossary theo lĩnh vực", expanded=True):
    selected_domain = st.selectbox("Chọn lĩnh vực video", options=list(GLOSSARY_DOMAIN_OPTIONS.keys()), index=3)
    selected_glossary_files = GLOSSARY_DOMAIN_OPTIONS[selected_domain]
    st.caption("Glossary files đang dùng:")
    for glossary_file in selected_glossary_files:
        st.code(glossary_file, language="text")
    quick_replacement_text = st.text_area(
        "Quick replacements thêm cho video này",
        value="",
        height=120,
        placeholder="Ví dụ:\n13.000 => mười ba nghìn\nMiddle East => Trung Đông\nairlines => các hãng hàng không",
    )

# Video source
st.subheader("0.5. Chọn nguồn video")
source_mode = st.radio("Nguồn video", options=["Upload file local", "YouTube URL"], horizontal=True)
input_path: Path | None = None

if source_mode == "Upload file local":
    uploaded_file = st.file_uploader("Upload 1 video ngắn", type=["mp4", "mov", "mkv", "avi", "webm"])
    if uploaded_file is not None:
        input_dir = PROJECT_ROOT / "data" / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        input_path = input_dir / uploaded_file.name
        with input_path.open("wb") as f:
            f.write(uploaded_file.read())
        st.success(f"Đã nhận video local: {input_path.name}")
else:
    youtube_url = st.text_input("Nhập YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    max_height_label = st.selectbox("Giới hạn độ phân giải tải về", options=["480p - nhẹ, nhanh", "720p - cân bằng", "1080p - nặng hơn"], index=1)
    max_height_map = {"480p - nhẹ, nhanh": 480, "720p - cân bằng": 720, "1080p - nặng hơn": 1080}
    if st.button("Tải video YouTube về data/input", type="primary", use_container_width=True):
        with st.spinner("Đang tải video từ YouTube bằng yt-dlp..."):
            try:
                download_result = download_youtube_to_input(url=youtube_url, max_height=max_height_map[max_height_label])
                st.session_state["youtube_download_result"] = download_result
                st.success("Đã tải video YouTube thành công.")
                st.rerun()
            except Exception as e:
                st.error("Lỗi khi tải video YouTube.")
                st.exception(e)
    if "youtube_download_result" in st.session_state:
        download_result = st.session_state["youtube_download_result"]
        input_path = Path(download_result["video_path"])
        st.markdown("**Video YouTube đã tải:**")
        st.json({"title": download_result.get("title"), "duration": download_result.get("duration"), "video_path": download_result.get("video_path"), "max_height": download_result.get("max_height")})

# Prepare
if input_path is not None and input_path.exists():
    st.subheader("1. Video đầu vào")
    st.video(str(input_path))
    col_prepare, col_reset = st.columns([2, 1])
    with col_prepare:
        prepare_button = st.button("Bước 1: Tạo transcript + bản dịch", type="primary", use_container_width=True)
    with col_reset:
        reset_button = st.button("Reset phiên làm việc", use_container_width=True)
    if reset_button:
        reset_working_state()
        st.success("Đã reset trạng thái giao diện. File trong thư mục data vẫn được giữ nguyên.")
        st.rerun()
    if prepare_button:
        run_prepare_translation(input_path, selected_domain, selected_glossary_files, quick_replacement_text)

# Editor
if "prepare_result" in st.session_state and "editor_df" in st.session_state:
    prepare_result = st.session_state["prepare_result"]
    bilingual_path = prepare_result["bilingual_transcript_path"]
    video_stem = prepare_result["video_stem"]
    st.divider()
    st.subheader("2. Kiểm tra và chỉnh sửa bản dịch")

    editor_df = normalize_editor_dataframe(st.session_state["editor_df"])
    st.session_state["editor_df"] = editor_df
    quality_df = quality_check_dataframe(editor_df)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Raw", prepare_result.get("raw_segments_count", "-"))
    col2.metric("Segments", len(editor_df))
    col3.metric("Đã chỉnh", get_changed_count(editor_df))
    col4.metric("Hơi dài", int((editor_df["tts_risk"] == "Hơi dài").sum()))
    col5.metric("Quá dài", int((editor_df["tts_risk"] == "Quá dài").sum()))
    col6.metric("Cảnh báo", len(quality_df))

    st.caption(f"Glossary domain: **{st.session_state.get('selected_domain', 'N/A')}** · Replacements loaded: **{st.session_state.get('runtime_replacements_count', 0)}**")

    edited_candidate = PROJECT_ROOT / "data" / "transcripts" / f"{video_stem}_bilingual_edited.json"
    if edited_candidate.exists():
        with st.expander("💾 Có bản chỉnh sửa cũ", expanded=False):
            st.write(f"Tìm thấy file: `{edited_candidate}`")
            if st.button("Tải lại bản chỉnh sửa cũ", use_container_width=True):
                edited_data = load_json(edited_candidate)
                st.session_state["editor_df"] = build_editor_dataframe(edited_data)
                st.session_state["edited_bilingual_path"] = str(edited_candidate)
                st.success("Đã tải lại bản chỉnh sửa cũ.")
                st.rerun()

    tab_table, tab_detail, tab_quality, tab_render = st.tabs(["📋 Bảng dịch", "✂️ Sửa / tách / phân vai", "✅ Kiểm tra chất lượng", "🎙️ Tạo lồng tiếng/video"])

    with tab_table:
        st.markdown("#### Bảng song ngữ")
        search_keyword = st.text_input("Tìm kiếm theo ID, tiếng Anh, tiếng Việt, trạng thái, voice hoặc speaker", value="", placeholder="Ví dụ: flights, Trung Đông, 13, quá dài, Speaker B...")
        filtered_df = filter_editor_dataframe(editor_df, search_keyword)
        edited_table_df = st.data_editor(
            filtered_df,
            use_container_width=True,
            height=560,
            hide_index=True,
            disabled=["segment_id", "start", "end", "duration", "tts_risk", "chars_per_sec", "source_text", "raw_vi_text", "voice", "rate"],
            column_order=["segment_id", "start", "end", "duration", "tts_risk", "chars_per_sec", "speaker_role", "voice_label", "rate_label", "source_text", "raw_vi_text", "vi_text"],
            column_config={
                "segment_id": st.column_config.NumberColumn("ID", width="small"),
                "start": st.column_config.NumberColumn("Start", width="small"),
                "end": st.column_config.NumberColumn("End", width="small"),
                "duration": st.column_config.NumberColumn("Duration", width="small"),
                "tts_risk": st.column_config.TextColumn("TTS risk", width="small"),
                "chars_per_sec": st.column_config.NumberColumn("Chars/s", width="small"),
                "speaker_role": st.column_config.SelectboxColumn("Speaker", options=SPEAKER_ROLE_OPTIONS, required=True, width="medium"),
                "voice_label": st.column_config.SelectboxColumn("Voice", options=list(VOICE_OPTIONS.keys()), required=True, width="medium"),
                "rate_label": st.column_config.SelectboxColumn("Rate", options=list(RATE_OPTIONS.keys()), required=True, width="medium"),
                "source_text": st.column_config.TextColumn("Source text", width="large"),
                "raw_vi_text": st.column_config.TextColumn("Raw VI", width="large"),
                "vi_text": st.column_config.TextColumn("Final VI - có thể sửa", width="large"),
            },
            key=f"editor_table_{st.session_state.get('editor_version', 0)}",
        )
        if not filtered_df.empty:
            current = st.session_state["editor_df"]
            for _, row in edited_table_df.iterrows():
                sid = int(row["segment_id"])
                mask = current["segment_id"] == sid
                if mask.any():
                    old_voice = str(current.loc[mask, "voice_label"].iloc[0])
                    old_rate = str(current.loc[mask, "rate_label"].iloc[0])
                    new_voice = canonical_voice_label(row.get("voice_label"))
                    new_rate_label = canonical_rate_label(row.get("rate_label"))
                    st.session_state["editor_df"].loc[mask, "vi_text"] = str(row.get("vi_text", "")).strip()
                    st.session_state["editor_df"].loc[mask, "speaker_role"] = str(row.get("speaker_role", "Mặc định"))
                    st.session_state["editor_df"].loc[mask, "voice_label"] = new_voice
                    st.session_state["editor_df"].loc[mask, "voice"] = get_voice_value(new_voice)
                    st.session_state["editor_df"].loc[mask, "rate_label"] = new_rate_label
                    st.session_state["editor_df"].loc[mask, "rate"] = get_rate_value(new_rate_label)
                    if new_voice != old_voice:
                        st.session_state["editor_df"].loc[mask, "voice_manual"] = True
                    if new_rate_label != old_rate:
                        st.session_state["editor_df"].loc[mask, "rate_manual"] = True
            st.session_state["editor_df"] = normalize_editor_dataframe(st.session_state["editor_df"])

    with tab_detail:
        st.markdown("#### Công cụ chỉnh segment")
        role_voice_map_ui = build_role_voice_map(
            st.selectbox("Voice cho Speaker A / Người dẫn", options=list(VOICE_OPTIONS.keys()), index=0, key="role_a_voice"),
            st.selectbox("Voice cho Speaker B / Khách mời", options=list(VOICE_OPTIONS.keys()), index=1, key="role_b_voice"),
            st.selectbox("Voice cho Speaker C", options=list(VOICE_OPTIONS.keys()), index=0, key="role_c_voice"),
        )

        with st.expander("🎭 Gán speaker role / voice hàng loạt", expanded=True):
            col_role, col_voice = st.columns(2)
            with col_role:
                with st.form("bulk_role_form"):
                    role_range_text = st.text_input("Khoảng segment gán role", placeholder="Ví dụ: 1-3,7")
                    bulk_role = st.selectbox("Speaker role", options=[r for r in SPEAKER_ROLE_OPTIONS if r != "Mặc định"], index=0)
                    submitted_role = st.form_submit_button("Áp dụng speaker role", use_container_width=True)
                    if submitted_role:
                        try:
                            count = apply_role_to_segment_ranges(role_range_text, bulk_role, role_voice_map_ui)
                            st.success(f"Đã gán role cho {count} segment.")
                            st.rerun()
                        except Exception as e:
                            st.error("Khoảng segment/role không hợp lệ. Ví dụ đúng: 1-3,7")
                            st.exception(e)
            with col_voice:
                with st.form("bulk_voice_form"):
                    voice_range_text = st.text_input("Khoảng segment gán voice", placeholder="Ví dụ: 1-3,7")
                    bulk_voice_label = st.selectbox("Voice cần gán", options=list(VOICE_OPTIONS.keys()), index=1)
                    submitted_voice = st.form_submit_button("Áp dụng voice", use_container_width=True)
                    if submitted_voice:
                        try:
                            count = apply_voice_to_segment_ranges(voice_range_text, bulk_voice_label)
                            st.success(f"Đã gán voice cho {count} segment.")
                            st.rerun()
                        except Exception as e:
                            st.error("Khoảng segment không hợp lệ. Ví dụ đúng: 1-3,7")
                            st.exception(e)

        with st.expander("🔗 Gộp segment bị tách cụt", expanded=False):
            with st.form("merge_form"):
                merge_range_text = st.text_input("Nhập khoảng segment cần gộp", placeholder="Ví dụ: 8-9 hoặc 14-16")
                merge_voice_strategy = st.selectbox("Giọng sau khi gộp", options=["Giữ giọng segment đầu", "Giữ giọng segment cuối"], index=0)
                submitted_merge = st.form_submit_button("Gộp các segment này", use_container_width=True)
                if submitted_merge:
                    try:
                        count = merge_segment_ranges(merge_range_text, voice_strategy="last" if "cuối" in merge_voice_strategy else "first")
                        st.success(f"Đã gộp {count} segment.")
                        st.rerun()
                    except Exception as e:
                        st.error("Khoảng segment không hợp lệ hoặc chưa đủ segment để gộp.")
                        st.exception(e)

        st.markdown("#### Sửa từng segment + tách segment + nghe thử TTS")
        segment_ids = st.session_state["editor_df"]["segment_id"].astype(int).tolist()
        selected_segment_id = st.selectbox("Chọn segment cần sửa", options=segment_ids, index=0)
        selected_row = st.session_state["editor_df"][st.session_state["editor_df"]["segment_id"] == int(selected_segment_id)].iloc[0]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("ID", int(selected_row["segment_id"]))
        c2.metric("Start", f"{float(selected_row['start']):.2f}s")
        c3.metric("End", f"{float(selected_row['end']):.2f}s")
        c4.metric("Risk", str(selected_row["tts_risk"]))
        c5.metric("Chars/s", f"{float(selected_row['chars_per_sec']):.1f}")

        col_seg_voice, col_seg_rate = st.columns(2)
        with col_seg_voice:
            current_voice_label = canonical_voice_label(selected_row.get("voice_label"))
            voice_index = list(VOICE_OPTIONS.keys()).index(current_voice_label) if current_voice_label in VOICE_OPTIONS else 0
            segment_voice_label = st.selectbox("Voice của segment", options=list(VOICE_OPTIONS.keys()), index=voice_index, key=f"segment_voice_{selected_segment_id}_{st.session_state.get('editor_version', 0)}")
        with col_seg_rate:
            current_rate_label = canonical_rate_label(selected_row.get("rate_label"), selected_row.get("rate"))
            rate_index = list(RATE_OPTIONS.keys()).index(current_rate_label) if current_rate_label in RATE_OPTIONS else 1
            segment_rate_label = st.selectbox("Rate của segment", options=list(RATE_OPTIONS.keys()), index=rate_index, key=f"segment_rate_{selected_segment_id}_{st.session_state.get('editor_version', 0)}")

        st.markdown("**Source text**")
        source_area = st.text_area("Source", value=str(selected_row["source_text"]), height=100, label_visibility="collapsed")
        st.markdown("**Bản dịch thô / raw VI**")
        raw_area = st.text_area("Raw VI", value=str(selected_row["raw_vi_text"]), height=100, disabled=True, label_visibility="collapsed")
        st.markdown("**Bản cuối dùng để TTS / Final VI**")
        new_vi_text = st.text_area("Final VI", value=str(selected_row["vi_text"]), height=140, label_visibility="collapsed", key=f"segment_text_area_{selected_segment_id}_{st.session_state.get('editor_version', 0)}")

        col_update, col_restore = st.columns(2)
        if col_update.button("Cập nhật đoạn này", type="primary", use_container_width=True):
            update_segment_text(int(selected_segment_id), new_vi_text)
            update_segment_voice(int(selected_segment_id), segment_voice_label)
            update_segment_rate(int(selected_segment_id), segment_rate_label, manual=True)
            mask = st.session_state["editor_df"]["segment_id"] == int(selected_segment_id)
            st.session_state["editor_df"].loc[mask, "source_text"] = source_area.strip()
            st.success(f"Đã cập nhật segment {selected_segment_id}.")
            st.rerun()
        if col_restore.button("Khôi phục về raw VI", use_container_width=True):
            update_segment_text(int(selected_segment_id), str(selected_row["raw_vi_text"]))
            st.success(f"Đã khôi phục segment {selected_segment_id}.")
            st.rerun()

        with st.expander("✂️ Tách segment này", expanded=False):
            split_default = round((float(selected_row["start"]) + float(selected_row["end"])) / 2, 2)
            split_time = st.slider(
                "Thời điểm tách",
                min_value=float(selected_row["start"]) + 0.1,
                max_value=float(selected_row["end"]) - 0.1,
                value=split_default,
                step=0.1,
            )
            src1, src2 = split_text_suggestion(str(selected_row["source_text"]))
            vi1, vi2 = split_text_suggestion(str(selected_row["vi_text"]))
            split_src_1 = st.text_area("Source phần 1", value=src1, height=80)
            split_vi_1 = st.text_area("VI phần 1", value=vi1, height=90)
            split_src_2 = st.text_area("Source phần 2", value=src2, height=80)
            split_vi_2 = st.text_area("VI phần 2", value=vi2, height=90)
            if st.button("Tách segment này", use_container_width=True):
                try:
                    split_segment(int(selected_segment_id), split_time, split_src_1, split_src_2, split_vi_1, split_vi_2)
                    st.success("Đã tách segment.")
                    st.rerun()
                except Exception as e:
                    st.error("Không tách được segment.")
                    st.exception(e)

        st.markdown("#### Nghe thử TTS đoạn này")
        if st.button("Tạo và nghe thử audio đoạn này", use_container_width=True):
            with st.spinner("Đang tạo preview TTS cho đoạn này..."):
                try:
                    preview_audio_path = make_preview_audio(
                        segment_id=int(selected_segment_id),
                        text=new_vi_text,
                        voice=get_voice_value(segment_voice_label),
                        rate=get_rate_value(segment_rate_label),
                    )
                    st.session_state["preview_audio_path"] = preview_audio_path
                    st.success("Đã tạo preview audio.")
                except Exception as e:
                    st.error("Lỗi khi tạo preview TTS.")
                    st.exception(e)
        preview_audio_path = st.session_state.get("preview_audio_path")
        if preview_audio_path and Path(preview_audio_path).exists():
            st.audio(Path(preview_audio_path).read_bytes(), format="audio/mp3")

    with tab_quality:
        st.markdown("#### Kiểm tra chất lượng trước khi render")
        qdf = quality_check_dataframe(st.session_state["editor_df"])
        if qdf.empty:
            st.success("Không phát hiện cảnh báo đáng kể.")
        else:
            st.dataframe(qdf, use_container_width=True, height=420)
            st.caption("Các cảnh báo này không chặn render, nhưng nên xử lý các dòng 'Nặng' và 'Cảnh báo' trước khi demo.")

    with tab_render:
        st.markdown("#### Cấu hình lồng tiếng/video")
        col_voice, col_rate = st.columns(2)
        with col_voice:
            default_voice_label = st.selectbox("Giọng mặc định", options=list(VOICE_OPTIONS.keys()), index=0)
            voice_render_mode_label = st.selectbox("Cách dùng giọng khi render", options=list(VOICE_RENDER_MODE_OPTIONS.keys()), index=0)
        with col_rate:
            default_rate_label = st.selectbox("Tốc độ mặc định", options=list(RATE_OPTIONS.keys()), index=3, help="Video news thường nói nhanh, nên thử +15% hoặc +20%.")
            auto_rate_enabled = st.checkbox("Tự động chỉnh tốc độ theo từng segment", value=True, help="Segment dài sẽ tự dùng +10%, +15% hoặc +20% nếu chưa chỉnh rate thủ công.")

        st.markdown("#### Chế độ render video")
        col_audio_mode, col_original_volume = st.columns(2)
        with col_audio_mode:
            audio_mode_label = st.selectbox("Chế độ xử lý âm thanh", options=list(AUDIO_MODE_OPTIONS.keys()), index=0)
        selected_audio_mode = AUDIO_MODE_OPTIONS[audio_mode_label]
        with col_original_volume:
            original_volume_db = st.slider("Âm lượng audio gốc khi trộn", min_value=-35, max_value=-6, value=-18, step=1)

        if selected_audio_mode == "subtitle_only":
            burn_subtitle = True
            st.info("Mode này giữ nguyên audio gốc và chỉ chèn phụ đề tiếng Việt vào video. Hệ thống sẽ không tạo TTS.")
        else:
            burn_subtitle = st.checkbox("Chèn phụ đề tiếng Việt trực tiếp vào video", value=False)

        subtitle_style = None
        subtitle_max_chars_per_line = 42
        if burn_subtitle:
            with st.expander("🎨 Tùy chỉnh style phụ đề", expanded=True):
                col_font, col_size, col_outline, col_margin = st.columns(4)
                subtitle_font = col_font.selectbox("Font", options=FONT_OPTIONS, index=0)
                subtitle_font_size = col_size.slider("Cỡ chữ", min_value=16, max_value=48, value=26, step=1)
                subtitle_outline = col_outline.slider("Độ dày viền", min_value=0, max_value=5, value=2, step=1)
                subtitle_margin_v = col_margin.slider("Khoảng cách đáy", min_value=10, max_value=120, value=45, step=5)
                subtitle_max_chars_per_line = st.slider("Số ký tự tối đa mỗi dòng phụ đề", min_value=24, max_value=60, value=42, step=1)
                subtitle_style = {
                    "font_name": subtitle_font,
                    "font_size": int(subtitle_font_size),
                    "outline": int(subtitle_outline),
                    "shadow": 1,
                    "margin_v": int(subtitle_margin_v),
                    "alignment": 2,
                    "primary_colour": "&H00FFFFFF",
                    "outline_colour": "&H00000000",
                }

        clear_cache = st.checkbox("Xóa cache TTS và tạo lại toàn bộ giọng đọc", value=True)
        st.warning("Khi render, hệ thống sẽ dùng bản dịch/segment/voice/rate hiện tại. Xem tab Quality check trước nếu muốn demo mượt.")

        col_save, col_render = st.columns(2)
        if col_save.button("Lưu bản dịch đã chỉnh", use_container_width=True):
            try:
                render_df_for_save = build_render_dataframe(
                    st.session_state["editor_df"],
                    default_voice_label,
                    default_rate_label,
                    VOICE_RENDER_MODE_OPTIONS[voice_render_mode_label],
                    auto_rate_enabled,
                    build_role_voice_map(st.session_state.get("role_a_voice", "Nữ - HoaiMy"), st.session_state.get("role_b_voice", "Nam - NamMinh"), st.session_state.get("role_c_voice", "Nữ - HoaiMy")),
                )
                edited_path = save_edited_bilingual(bilingual_path, render_df_for_save)
                st.session_state["edited_bilingual_path"] = edited_path
                st.success(f"Đã lưu bản dịch đã chỉnh: {edited_path}")
            except Exception as e:
                st.error("Lỗi khi lưu bản dịch đã chỉnh.")
                st.exception(e)

        if col_render.button("Bước 2: Tạo video từ bản đã chỉnh", type="primary", use_container_width=True):
            try:
                role_voice_map_render = build_role_voice_map(st.session_state.get("role_a_voice", "Nữ - HoaiMy"), st.session_state.get("role_b_voice", "Nam - NamMinh"), st.session_state.get("role_c_voice", "Nữ - HoaiMy"))
                render_df = build_render_dataframe(
                    st.session_state["editor_df"],
                    default_voice_label,
                    default_rate_label,
                    VOICE_RENDER_MODE_OPTIONS[voice_render_mode_label],
                    auto_rate_enabled,
                    role_voice_map_render,
                )
                edited_path = save_edited_bilingual(bilingual_path, render_df)
                st.session_state["edited_bilingual_path"] = edited_path
                with st.spinner("Đang tạo video đầu ra..."):
                    pipeline = get_pipeline()
                    pipeline.tts_engine.voice = get_voice_value(default_voice_label)
                    pipeline.tts_engine.rate = get_rate_value(default_rate_label)
                    render_result = pipeline.render_from_bilingual(
                        edited_path,
                        clear_tts_cache=clear_cache,
                        audio_mode=selected_audio_mode,
                        burn_subtitle=burn_subtitle,
                        original_audio_volume_db=float(original_volume_db),
                        subtitle_style=subtitle_style,
                        subtitle_max_chars_per_line=subtitle_max_chars_per_line,
                    )
                    render_result["selected_voice_label"] = default_voice_label
                    render_result["selected_voice"] = get_voice_value(default_voice_label)
                    render_result["selected_rate"] = get_rate_value(default_rate_label)
                    render_result["voice_render_mode"] = VOICE_RENDER_MODE_OPTIONS[voice_render_mode_label]
                    render_result["auto_rate_enabled"] = auto_rate_enabled
                    render_result["selected_audio_mode"] = selected_audio_mode
                    render_result["selected_burn_subtitle"] = burn_subtitle
                    render_result["selected_original_volume_db"] = float(original_volume_db)
                    render_result["selected_subtitle_style"] = subtitle_style
                    render_result["selected_subtitle_max_chars_per_line"] = subtitle_max_chars_per_line
                    report_paths = create_demo_report(
                        render_result,
                        render_df,
                        quality_check_dataframe(render_df),
                        st.session_state.get("selected_domain", "N/A"),
                        st.session_state.get("runtime_replacements_count", 0),
                    )
                    render_result.update(report_paths)
                    st.session_state["render_result"] = render_result
                    st.session_state["editor_df"] = render_df
                st.success("Đã tạo xong video đầu ra và report demo.")
                st.rerun()
            except Exception as e:
                st.error("Lỗi ở bước tạo video.")
                st.exception(e)

# Output
if "render_result" in st.session_state:
    result = st.session_state["render_result"]
    st.divider()
    st.subheader("3. Kết quả đầu ra")
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    col_info1.metric("Segments", result.get("segments_count", "-"))
    col_info2.metric("Voice mặc định", result.get("selected_voice_label", get_voice_label(result.get("selected_voice"))))
    col_info3.metric("Rate", result.get("selected_rate", "-"))
    col_info4.metric("Audio mode", result.get("selected_audio_mode", result.get("audio_mode", "-")))

    st.markdown("### Video đầu ra")
    st.video(result["output_video_path"])

    st.markdown("### File đầu ra")
    col_video, col_srt, col_json, col_report_json, col_report_html = st.columns(5)
    with col_video:
        show_download_button(result.get("output_video_path"), "Tải video", "video/mp4")
    with col_srt:
        show_download_button(result.get("output_subtitle_path"), "Tải .srt", "text/plain")
    with col_json:
        show_download_button(result.get("bilingual_transcript_path"), "Tải transcript JSON", "application/json")
    with col_report_json:
        show_download_button(result.get("report_json_path"), "Tải report JSON", "application/json")
    with col_report_html:
        show_download_button(result.get("report_html_path"), "Tải report HTML", "text/html")

    with st.expander("Thông tin pipeline"):
        st.json(result)
