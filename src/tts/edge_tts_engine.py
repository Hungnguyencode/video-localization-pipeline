from __future__ import annotations

import asyncio
import hashlib
import re
import time
from pathlib import Path
from typing import Dict, List

import edge_tts
from edge_tts.exceptions import NoAudioReceived

from src.tts.fpt_tts_engine import FPTTTSEngine
from src.tts.vi_number_normalizer import normalize_vietnamese_numbers_for_tts
from src.utils.config import ensure_dir


class EdgeTTSEngine:
    """TTS engine chính của project.

    Tên cũ vẫn là EdgeTTSEngine để không phải sửa main_pipeline.py,
    nhưng bên trong đã hỗ trợ multi-provider:
    - Edge TTS: voice dạng vi-VN-HoaiMyNeural / vi-VN-NamMinhNeural
    - FPT.AI: voice dạng fpt:banmai / fpt:leminh / ...
    """

    def __init__(
        self,
        output_dir: str,
        voice: str = "vi-VN-HoaiMyNeural",
        rate: str = "+0%",
        volume: str = "+0%",
        max_retries: int = 3,
    ):
        self.output_dir = ensure_dir(output_dir)
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.max_retries = int(max_retries)
        self.fpt_engine = FPTTTSEngine()

    def _clean_text(self, text: str) -> str:
        text = (text or "").strip()
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Fix lỗi 13.000 / 850.000 bị đọc thành từng chữ số.
        text = normalize_vietnamese_numbers_for_tts(text)

        if len(text) > 450:
            text = text[:450].rsplit(" ", 1)[0].strip()

        return text

    def _safe_name(self, value: str) -> str:
        value = str(value or "").strip()
        value = re.sub(r"[^a-zA-Z0-9_+\-]+", "_", value)
        return value.strip("_") or "default"

    def _is_fpt_voice(self, voice: str) -> bool:
        return str(voice or "").strip().startswith("fpt:")

    def _fpt_voice_code(self, voice: str) -> str:
        voice = str(voice or "").strip()
        return voice.split(":", 1)[1] if voice.startswith("fpt:") else voice

    def _edge_rate(self, rate: str) -> str:
        rate = str(rate or self.rate or "+0%").strip()
        if rate in {"-10%", "+0%", "+10%", "+15%", "+20%"}:
            return rate
        return "+0%"

    def _fpt_speed(self, rate: str) -> str:
        """Map rate UI sang speed FPT.

        FPT dùng speed dạng số. Mức này chủ yếu để demo; có thể chỉnh lại nếu cần.
        """
        mapping = {
            "-10%": "-1",
            "+0%": "0",
            "+10%": "1",
            "+15%": "2",
            "+20%": "3",
        }
        return mapping.get(str(rate or "+0%").strip(), "0")

    def _make_cache_key(
        self,
        text: str,
        voice: str,
        rate: str,
        volume: str,
    ) -> str:
        provider = "fpt" if self._is_fpt_voice(voice) else "edge"
        raw = f"{provider}|{voice}|{rate}|{volume}|{text}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    def _build_segment_output_path(
        self,
        segment_dir: Path,
        idx: int,
        text: str,
        voice: str,
        rate: str,
        volume: str,
    ) -> Path:
        cache_key = self._make_cache_key(text=text, voice=voice, rate=rate, volume=volume)
        safe_voice = self._safe_name(voice)
        safe_rate = self._safe_name(rate)
        provider = "fpt" if self._is_fpt_voice(voice) else "edge"
        return segment_dir / f"segment_{idx:04d}_{provider}_{safe_voice}_{safe_rate}_{cache_key}.mp3"

    async def _synthesize_edge_async(
        self,
        text: str,
        output_path: Path,
        voice: str,
        rate: str,
        volume: str,
    ) -> None:
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=self._edge_rate(rate),
            volume=volume,
        )
        await communicate.save(str(output_path))

    def _synthesize_edge_one(
        self,
        text: str,
        output_path: Path,
        voice: str,
        rate: str,
        volume: str,
    ) -> str:
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                asyncio.run(
                    self._synthesize_edge_async(
                        text=text,
                        output_path=output_path,
                        voice=voice,
                        rate=rate,
                        volume=volume,
                    )
                )

                if output_path.exists() and output_path.stat().st_size > 0:
                    return str(output_path)

                raise RuntimeError(f"TTS output file is empty: {output_path}")

            except NoAudioReceived as e:
                last_error = e
                print(
                    f"[TTS][Edge] No audio received. Retry {attempt}/{self.max_retries}. "
                    f"Voice={voice}, Text={text[:100]}"
                )
                time.sleep(1.5)

            except Exception as e:  # noqa: BLE001
                last_error = e
                print(
                    f"[TTS][Edge] Error. Retry {attempt}/{self.max_retries}. "
                    f"Voice={voice}, Reason={e}"
                )
                time.sleep(1.5)

        raise RuntimeError(f"Edge TTS failed after {self.max_retries} retries: {last_error}")

    def synthesize_one(
        self,
        text: str,
        output_path: str | Path,
        voice: str | None = None,
        rate: str | None = None,
        volume: str | None = None,
        use_cache: bool = True,
    ) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        voice = voice or self.voice
        rate = rate or self.rate
        volume = volume or self.volume
        cleaned_text = self._clean_text(text)

        if not cleaned_text:
            raise ValueError("TTS text is empty after cleaning.")

        if use_cache and path.exists() and path.stat().st_size > 0:
            return str(path)

        if self._is_fpt_voice(voice):
            fpt_voice = self._fpt_voice_code(voice)
            print(f"[TTS][FPT] Creating audio | voice={fpt_voice} | speed={self._fpt_speed(rate)}")
            return self.fpt_engine.synthesize_one(
                text=cleaned_text,
                output_path=path,
                voice=fpt_voice,
                speed=self._fpt_speed(rate),
                audio_format="mp3",
            )

        print(f"[TTS][Edge] Creating audio | voice={voice} | rate={self._edge_rate(rate)}")
        return self._synthesize_edge_one(
            text=cleaned_text,
            output_path=path,
            voice=voice,
            rate=rate,
            volume=volume,
        )

    def synthesize_segments(
        self,
        segments: List[Dict],
        video_stem: str,
        use_smart_cache: bool = True,
    ) -> List[Dict]:
        segment_dir = self.output_dir / video_stem
        segment_dir.mkdir(parents=True, exist_ok=True)

        output_segments = []

        for idx, item in enumerate(segments, start=1):
            text = self._clean_text(item.get("vi_text") or "")

            if not text:
                print(f"[TTS] Skip empty segment {idx}")
                continue

            voice = item.get("voice") or item.get("tts_voice") or self.voice
            rate = item.get("rate") or item.get("tts_rate") or self.rate
            volume = item.get("volume") or item.get("tts_volume") or self.volume
            provider = "fpt" if self._is_fpt_voice(voice) else "edge"

            output_path = self._build_segment_output_path(
                segment_dir=segment_dir,
                idx=idx,
                text=text,
                voice=voice,
                rate=rate,
                volume=volume,
            )

            try:
                existed_before = output_path.exists() and output_path.stat().st_size > 0

                audio_path = self.synthesize_one(
                    text=text,
                    output_path=output_path,
                    voice=voice,
                    rate=rate,
                    volume=volume,
                    use_cache=use_smart_cache,
                )

                new_item = dict(item)
                new_item["vi_text"] = item.get("vi_text") or ""
                new_item["tts_input_text"] = text
                new_item["tts_audio_path"] = audio_path
                new_item["tts_voice"] = voice
                new_item["tts_provider"] = provider
                new_item["tts_rate"] = rate
                new_item["tts_volume"] = volume

                output_segments.append(new_item)

                cache_status = "cached" if existed_before and use_smart_cache else "created"
                print(
                    f"[TTS] Done segment {idx}: {output_path.name} "
                    f"| provider={new_item['tts_provider']} | voice={voice} | rate={rate} | {cache_status}"
                )

            except Exception as e:
                raise RuntimeError(
                    f"TTS failed at segment {idx}. "
                    f"Voice={voice}, provider={provider}, text={text[:120]}"
                ) from e

        if not output_segments:
            raise RuntimeError(
                "TTS failed for all segments. Check Edge voice, FPT API key, network, or translated text."
            )

        return output_segments
