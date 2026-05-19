from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import edge_tts
from edge_tts.exceptions import NoAudioReceived
from pydub import AudioSegment, silence

from src.tts.vi_number_normalizer import normalize_vietnamese_numbers_for_tts
from src.utils.config import ensure_dir


class EdgeTTSEngine:
    """
    TTS engine chính của project.

    Bản này sửa các lỗi:
    - Segment báo created nhưng audio thực tế quá ngắn / gần như câm.
    - Segment FPT tạo file lỗi nhưng vẫn được đưa vào render.
    - Chỉ dùng Edge nhưng vẫn bị lỗi do thiếu FPT key.
    - FPT fail thì fallback sang Edge cho riêng segment đó.
    - Cache audio cũ nếu lỗi sẽ bị xóa và tạo lại.
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

        self.max_retries = int(os.getenv("EDGE_MAX_RETRIES", max_retries))

        # Không khởi tạo FPT ngay tại đây.
        # Nếu chỉ dùng Edge mà thiếu FPT_AI_API_KEY thì app vẫn phải chạy được.
        self._fpt_engine = None

        self.fpt_fallback_to_edge = (
            str(os.getenv("FPT_FALLBACK_TO_EDGE", "1")).strip().lower()
            not in {"0", "false", "no", "off"}
        )
        self.fallback_edge_voice = os.getenv(
            "FPT_FALLBACK_EDGE_VOICE",
            "vi-VN-HoaiMyNeural",
        )

        # Ngưỡng kiểm tra audio.
        # Nếu segment 10 chỉ tạo ra 157ms thì chắc chắn bị bắt lỗi ở đây.
        self.min_audio_ms = int(os.getenv("TTS_MIN_AUDIO_MS", "450"))
        self.min_nonsilent_ms = int(os.getenv("TTS_MIN_NONSILENT_MS", "250"))
        self.silence_thresh_db = float(os.getenv("TTS_SILENCE_THRESH_DB", "-45"))

        # Kiểm tra audio có quá ngắn so với duration gốc của subtitle không.
        self.enable_target_check = (
            str(os.getenv("TTS_VALIDATE_TARGET_DURATION", "1")).strip().lower()
            not in {"0", "false", "no", "off"}
        )

        # Ví dụ target 4600ms mà audio chỉ 157ms thì fail.
        self.min_target_ratio = float(os.getenv("TTS_MIN_TARGET_RATIO", "0.25"))

    # =========================================================
    # Basic helpers
    # =========================================================

    def _get_fpt_engine(self):
        if self._fpt_engine is None:
            from src.tts.fpt_tts_engine import FPTTTSEngine

            self._fpt_engine = FPTTTSEngine()

        return self._fpt_engine

    def _clean_text(self, text: str) -> str:
        text = (text or "").strip()

        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
        text = text.replace("\ufeff", "")
        text = text.replace("\u200b", "")
        text = text.replace("\u200c", "")
        text = text.replace("\u200d", "")

        text = re.sub(r"\s+", " ", text).strip()

        # Sửa dấu câu kỳ như ",."
        text = text.replace(" ,", ",")
        text = text.replace(" .", ".")
        text = text.replace(" ?", "?")
        text = text.replace(" !", "!")
        text = re.sub(r",\s*([.!?])", r"\1", text)
        text = re.sub(r"\.\s*\.", ".", text)

        text = normalize_vietnamese_numbers_for_tts(text)

        return text.strip()

    def _safe_name(self, value: str) -> str:
        value = str(value or "").strip()
        value = re.sub(r"[^a-zA-Z0-9_+\-]+", "_", value)
        return value.strip("_") or "default"

    def _is_fpt_voice(self, voice: str) -> bool:
        return str(voice or "").strip().lower().startswith("fpt:")

    def _fpt_voice_code(self, voice: str) -> str:
        voice = str(voice or "").strip()
        return voice.split(":", 1)[1] if voice.lower().startswith("fpt:") else voice

    def _edge_rate(self, rate: str) -> str:
        rate = str(rate or self.rate or "+0%").strip()

        allowed = {
            "-50%", "-40%", "-30%", "-20%", "-10%",
            "+0%", "+5%", "+10%", "+15%", "+20%",
            "+25%", "+30%", "+40%", "+50%",
        }

        if rate in allowed:
            return rate

        return "+0%"

    def _fpt_speed(self, rate: str) -> str:
        mapping = {
            "-20%": "-2",
            "-10%": "-1",
            "+0%": "0",
            "+5%": "0",
            "+10%": "1",
            "+15%": "2",
            "+20%": "3",
            "+25%": "3",
            "+30%": "3",
        }
        return mapping.get(str(rate or "+0%").strip(), "0")

    def _make_cache_key(self, text: str, voice: str, rate: str, volume: str) -> str:
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
        provider_override: Optional[str] = None,
    ) -> Path:
        cache_key = self._make_cache_key(
            text=text,
            voice=voice,
            rate=rate,
            volume=volume,
        )
        safe_voice = self._safe_name(voice)
        safe_rate = self._safe_name(rate)

        provider = provider_override or (
            "fpt" if self._is_fpt_voice(voice) else "edge"
        )

        return segment_dir / (
            f"segment_{idx:04d}_{provider}_{safe_voice}_{safe_rate}_{cache_key}.mp3"
        )

    def _segment_target_ms(self, item: Dict) -> Optional[int]:
        """
        Lấy duration gốc của subtitle segment.

        Hỗ trợ cả:
        - start/end dạng giây: 139.2 -> 143.84
        - start/end dạng milliseconds: 139200 -> 143840
        """

        try:
            start = item.get("start", None)
            end = item.get("end", None)

            if start is None or end is None:
                return None

            start_f = float(start)
            end_f = float(end)

            if end_f <= start_f:
                return None

            diff = end_f - start_f

            # Nếu diff lớn hơn 1000 thì khả năng cao đã là ms.
            if diff > 1000:
                return int(diff)

            return int(diff * 1000)

        except Exception:
            return None

    # =========================================================
    # Audio validation
    # =========================================================

    def _audio_quality(
        self,
        path: str | Path,
        text: str = "",
        target_ms: Optional[int] = None,
    ) -> dict:
        path = Path(path)

        if not path.exists():
            return {
                "ok": False,
                "reason": "file không tồn tại",
                "duration_ms": 0,
                "nonsilent_ms": 0,
            }

        if path.stat().st_size <= 0:
            return {
                "ok": False,
                "reason": "file rỗng",
                "duration_ms": 0,
                "nonsilent_ms": 0,
            }

        if path.stat().st_size < 1500:
            return {
                "ok": False,
                "reason": f"file quá nhỏ: {path.stat().st_size} bytes",
                "duration_ms": 0,
                "nonsilent_ms": 0,
            }

        try:
            audio = AudioSegment.from_file(path)
        except Exception as exc:
            return {
                "ok": False,
                "reason": f"không đọc được audio: {exc}",
                "duration_ms": 0,
                "nonsilent_ms": 0,
            }

        duration_ms = len(audio)

        try:
            ranges = silence.detect_nonsilent(
                audio,
                min_silence_len=120,
                silence_thresh=self.silence_thresh_db,
            )
            nonsilent_ms = sum(max(0, end - start) for start, end in ranges)
        except Exception:
            nonsilent_ms = duration_ms

        text_len = len((text or "").strip())

        if duration_ms <= 0:
            return {
                "ok": False,
                "reason": "duration audio bằng 0",
                "duration_ms": duration_ms,
                "nonsilent_ms": nonsilent_ms,
            }

        if duration_ms < self.min_audio_ms:
            return {
                "ok": False,
                "reason": f"audio quá ngắn {duration_ms}ms",
                "duration_ms": duration_ms,
                "nonsilent_ms": nonsilent_ms,
            }

        if text_len >= 20 and duration_ms < 1000:
            return {
                "ok": False,
                "reason": f"text dài nhưng audio chỉ {duration_ms}ms",
                "duration_ms": duration_ms,
                "nonsilent_ms": nonsilent_ms,
            }

        if nonsilent_ms < self.min_nonsilent_ms:
            return {
                "ok": False,
                "reason": f"audio gần như câm, nonsilent={nonsilent_ms}ms",
                "duration_ms": duration_ms,
                "nonsilent_ms": nonsilent_ms,
            }

        if (
            self.enable_target_check
            and target_ms is not None
            and target_ms >= 2500
            and text_len >= 20
        ):
            min_allowed = max(900, int(target_ms * self.min_target_ratio))

            if duration_ms < min_allowed:
                return {
                    "ok": False,
                    "reason": (
                        f"audio quá ngắn so với segment: "
                        f"audio={duration_ms}ms, target={target_ms}ms, "
                        f"min_allowed={min_allowed}ms"
                    ),
                    "duration_ms": duration_ms,
                    "nonsilent_ms": nonsilent_ms,
                }

        return {
            "ok": True,
            "reason": "ok",
            "duration_ms": duration_ms,
            "nonsilent_ms": nonsilent_ms,
        }

    def _is_good_audio(
        self,
        path: str | Path,
        text: str = "",
        target_ms: Optional[int] = None,
    ) -> bool:
        quality = self._audio_quality(path, text=text, target_ms=target_ms)
        return bool(quality.get("ok"))

    def _delete_bad_audio(self, path: str | Path, reason: str) -> None:
        path = Path(path)

        try:
            if path.exists():
                print(f"[TTS][BAD AUDIO] Delete {path.name}: {reason}")
                path.unlink()
        except Exception:
            pass

    # =========================================================
    # Edge TTS
    # =========================================================

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

    def _run_edge_async(
        self,
        text: str,
        output_path: Path,
        voice: str,
        rate: str,
        volume: str,
    ) -> None:
        """
        Chạy edge_tts async từ code sync.

        Streamlit thường không có event loop đang chạy ở thread này,
        nên asyncio.run là đủ. Nếu môi trường có loop lạ thì tạo loop mới.
        """

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
        except RuntimeError:
            loop = asyncio.new_event_loop()

            try:
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    self._synthesize_edge_async(
                        text=text,
                        output_path=output_path,
                        voice=voice,
                        rate=rate,
                        volume=volume,
                    )
                )
            finally:
                try:
                    loop.close()
                except Exception:
                    pass

    def _synthesize_edge_one(
        self,
        text: str,
        output_path: Path,
        voice: str,
        rate: str,
        volume: str,
        target_ms: Optional[int] = None,
    ) -> str:
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                if output_path.exists():
                    output_path.unlink()

                self._run_edge_async(
                    text=text,
                    output_path=output_path,
                    voice=voice,
                    rate=rate,
                    volume=volume,
                )

                quality = self._audio_quality(
                    output_path,
                    text=text,
                    target_ms=target_ms,
                )

                if quality["ok"]:
                    return str(output_path)

                raise RuntimeError(f"Bad Edge audio: {quality['reason']}")

            except NoAudioReceived as exc:
                last_error = exc
                print(
                    f"[TTS][Edge] No audio. Retry {attempt}/{self.max_retries}. "
                    f"Voice={voice}"
                )
                self._delete_bad_audio(output_path, "NoAudioReceived")
                time.sleep(min(2.0 * attempt, 10.0))

            except Exception as exc:
                last_error = exc
                print(
                    f"[TTS][Edge] Error. Retry {attempt}/{self.max_retries}. "
                    f"Voice={voice}, Reason={exc}"
                )
                self._delete_bad_audio(output_path, str(exc))
                time.sleep(min(2.0 * attempt, 10.0))

        raise RuntimeError(
            f"Edge TTS failed after {self.max_retries} retries: {last_error}"
        )

    # =========================================================
    # Public synthesize one
    # =========================================================

    def synthesize_one(
        self,
        text: str,
        output_path: str | Path,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        volume: Optional[str] = None,
        use_cache: bool = True,
        target_ms: Optional[int] = None,
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
            quality = self._audio_quality(
                path,
                text=cleaned_text,
                target_ms=target_ms,
            )

            if quality["ok"]:
                return str(path)

            self._delete_bad_audio(path, quality["reason"])

        if self._is_fpt_voice(voice):
            fpt_voice = self._fpt_voice_code(voice)
            fpt_speed = self._fpt_speed(rate)

            print(
                f"[TTS][FPT] Creating audio | voice={fpt_voice} "
                f"| speed={fpt_speed} | chars={len(cleaned_text)}"
            )

            fpt_engine = self._get_fpt_engine()

            result = fpt_engine.synthesize_one(
                text=cleaned_text,
                output_path=path,
                voice=fpt_voice,
                speed=fpt_speed,
                audio_format="mp3",
                use_cache=use_cache,
            )

            quality = self._audio_quality(
                result,
                text=cleaned_text,
                target_ms=target_ms,
            )

            if not quality["ok"]:
                self._delete_bad_audio(result, quality["reason"])
                raise RuntimeError(f"Bad FPT audio: {quality['reason']}")

            return result

        print(
            f"[TTS][Edge] Creating audio | voice={voice} "
            f"| rate={self._edge_rate(rate)} | chars={len(cleaned_text)}"
        )

        return self._synthesize_edge_one(
            text=cleaned_text,
            output_path=path,
            voice=voice,
            rate=rate,
            volume=volume,
            target_ms=target_ms,
        )

    # =========================================================
    # Public synthesize segments
    # =========================================================

    def synthesize_segments(
        self,
        segments: List[Dict],
        video_stem: str,
        use_smart_cache: bool = True,
    ) -> List[Dict]:
        segment_dir = self.output_dir / video_stem
        segment_dir.mkdir(parents=True, exist_ok=True)

        output_segments: List[Dict] = []

        for idx, item in enumerate(segments, start=1):
            text = self._clean_text(item.get("vi_text") or "")

            if not text:
                print(f"[TTS] Skip empty segment {idx}")
                continue

            voice = item.get("voice") or item.get("tts_voice") or self.voice
            rate = item.get("rate") or item.get("tts_rate") or self.rate
            volume = item.get("volume") or item.get("tts_volume") or self.volume

            target_ms = self._segment_target_ms(item)

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
                existed_before = (
                    output_path.exists()
                    and output_path.stat().st_size > 0
                    and self._is_good_audio(
                        output_path,
                        text=text,
                        target_ms=target_ms,
                    )
                )

                audio_path = self.synthesize_one(
                    text=text,
                    output_path=output_path,
                    voice=voice,
                    rate=rate,
                    volume=volume,
                    use_cache=use_smart_cache,
                    target_ms=target_ms,
                )

                actual_provider = provider

            except Exception as exc:
                if provider == "fpt" and self.fpt_fallback_to_edge:
                    print(
                        f"[TTS][FPT] Segment {idx} failed, fallback to Edge. "
                        f"Reason={exc}"
                    )

                    fallback_path = self._build_segment_output_path(
                        segment_dir=segment_dir,
                        idx=idx,
                        text=text,
                        voice=self.fallback_edge_voice,
                        rate=rate,
                        volume=volume,
                        provider_override="edge_fallback",
                    )

                    existed_before = (
                        fallback_path.exists()
                        and fallback_path.stat().st_size > 0
                        and self._is_good_audio(
                            fallback_path,
                            text=text,
                            target_ms=target_ms,
                        )
                    )

                    audio_path = self.synthesize_one(
                        text=text,
                        output_path=fallback_path,
                        voice=self.fallback_edge_voice,
                        rate=rate,
                        volume=volume,
                        use_cache=use_smart_cache,
                        target_ms=target_ms,
                    )

                    actual_provider = "edge_fallback"
                    output_path = fallback_path
                    voice = self.fallback_edge_voice

                else:
                    raise RuntimeError(
                        f"TTS failed at segment {idx}. "
                        f"Voice={voice}, provider={provider}, "
                        f"target={target_ms}ms, text={text[:160]}"
                    ) from exc

            quality = self._audio_quality(
                audio_path,
                text=text,
                target_ms=target_ms,
            )

            if not quality["ok"]:
                raise RuntimeError(
                    f"TTS segment {idx} tạo xong nhưng audio lỗi: "
                    f"{quality['reason']} | {audio_path}"
                )

            new_item = dict(item)
            new_item["vi_text"] = item.get("vi_text") or ""
            new_item["tts_input_text"] = text
            new_item["tts_audio_path"] = str(audio_path)
            new_item["tts_voice"] = voice
            new_item["tts_provider"] = actual_provider
            new_item["tts_rate"] = rate
            new_item["tts_volume"] = volume
            new_item["tts_duration_ms"] = quality["duration_ms"]
            new_item["tts_nonsilent_ms"] = quality["nonsilent_ms"]

            if target_ms is not None:
                new_item["tts_target_ms"] = target_ms

            output_segments.append(new_item)

            cache_status = "cached" if existed_before and use_smart_cache else "created"

            print(
                f"[TTS] Done segment {idx}: {output_path.name} "
                f"| provider={actual_provider} "
                f"| voice={voice} "
                f"| rate={rate} "
                f"| {cache_status} "
                f"| audio={quality['duration_ms']}ms "
                f"| nonsilent={quality['nonsilent_ms']}ms "
                f"| target={target_ms}ms"
            )

        if not output_segments:
            raise RuntimeError(
                "TTS failed for all segments. Check voice/API/network/translated text."
            )

        return output_segments