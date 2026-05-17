from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import List

import requests
from dotenv import load_dotenv

FPT_TTS_URL = "https://api.fpt.ai/hmi/tts/v5"


class FPTTTSEngine:
    """
    FPT.AI Text-to-Speech client for this video localization pipeline.

    Bản sửa này tập trung xử lý lỗi hay gặp của FPT free package:
    - URL audio trả 404 trong lúc server FPT còn xử lý.
    - Segment dài làm FPT xử lý rất lâu hoặc không sinh audio.
    - Retry cùng một đoạn quá dài gây mất thời gian rồi fail cả video.

    Cách làm mới:
    - Chia câu nhỏ hơn trước khi gửi FPT.
    - Chờ lâu hơn cho gói free.
    - Nếu chunk vẫn fail, tự chia nhỏ chunk đó thêm 1 lần rồi thử lại.
    - Cache audio nếu file đã tồn tại.
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout_sec: int = 30,
        poll_interval_sec: float = 2.0,
        max_wait_sec: int | None = None,
        chunk_chars: int | None = None,
        min_chunk_chars: int = 45,
        max_retries: int = 2,
    ):
        load_dotenv(override=False)
        self.api_key = (api_key or os.getenv("FPT_AI_API_KEY") or "").strip()
        self.timeout_sec = int(os.getenv("FPT_TIMEOUT_SEC", timeout_sec))
        self.poll_interval_sec = float(os.getenv("FPT_POLL_INTERVAL_SEC", poll_interval_sec))
        self.max_wait_sec = int(os.getenv("FPT_MAX_WAIT_SEC", max_wait_sec or 240))
        self.chunk_chars = int(os.getenv("FPT_CHUNK_CHARS", chunk_chars or 120))
        self.min_chunk_chars = int(os.getenv("FPT_MIN_CHUNK_CHARS", min_chunk_chars))
        self.max_retries = int(os.getenv("FPT_MAX_RETRIES", max_retries))
        self.session = requests.Session()

        if not self.api_key:
            raise RuntimeError(
                "Thiếu FPT_AI_API_KEY trong file .env. Ví dụ: FPT_AI_API_KEY=your_key_here"
            )

    def synthesize_one(
        self,
        text: str,
        output_path: str | Path,
        voice: str = "banmai",
        speed: str = "0",
        audio_format: str = "mp3",
        use_cache: bool = True,
    ) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        text = self._clean_text(text)
        if len(text) < 2:
            raise ValueError("FPT TTS text is empty or too short.")

        if use_cache and output_path.exists() and output_path.stat().st_size > 0:
            print(f"[FPT TTS] Cache hit: {output_path.name}")
            return str(output_path)

        chunks = self._split_text(text, self.chunk_chars)
        if not chunks:
            raise ValueError("FPT TTS text has no valid chunks.")

        if len(chunks) == 1:
            return self._synthesize_chunk_or_split(
                text=chunks[0],
                output_path=output_path,
                voice=voice,
                speed=speed,
                audio_format=audio_format,
            )

        tmp_dir = output_path.parent / f".tmp_{output_path.stem}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        chunk_paths: List[Path] = []
        try:
            print(
                f"[FPT TTS] Long segment -> {len(chunks)} chunks | "
                f"chars={len(text)} | voice={voice} | speed={speed}"
            )
            for i, chunk in enumerate(chunks, start=1):
                chunk_path = tmp_dir / f"chunk_{i:03d}.{audio_format}"
                print(f"[FPT TTS] Chunk {i}/{len(chunks)} | chars={len(chunk)}")
                self._synthesize_chunk_or_split(
                    text=chunk,
                    output_path=chunk_path,
                    voice=voice,
                    speed=speed,
                    audio_format=audio_format,
                )
                chunk_paths.append(chunk_path)

            self._concat_audio_files(chunk_paths, output_path)
            if not output_path.exists() or output_path.stat().st_size <= 0:
                raise RuntimeError(f"FPT concat output is empty: {output_path}")
            return str(output_path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _synthesize_chunk_or_split(
        self,
        text: str,
        output_path: Path,
        voice: str,
        speed: str,
        audio_format: str,
    ) -> str:
        """Try a chunk. If FPT keeps returning 404/timeout, split it smaller once."""
        try:
            return self._synthesize_chunk_with_retry(text, output_path, voice, speed, audio_format)
        except Exception as first_error:
            if len(text) <= max(self.min_chunk_chars * 2, 70):
                raise

            smaller_size = max(self.min_chunk_chars, min(70, len(text) // 2))
            smaller_chunks = self._split_text(text, smaller_size)
            if len(smaller_chunks) <= 1:
                raise

            print(
                f"[FPT TTS] Chunk failed, split smaller -> {len(smaller_chunks)} parts. "
                f"Reason: {first_error}"
            )
            tmp_dir = output_path.parent / f".tmp_retry_{output_path.stem}"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir(parents=True, exist_ok=True)

            retry_paths: List[Path] = []
            try:
                for i, small in enumerate(smaller_chunks, start=1):
                    small_path = tmp_dir / f"small_{i:03d}.{audio_format}"
                    self._synthesize_chunk_with_retry(small, small_path, voice, speed, audio_format)
                    retry_paths.append(small_path)
                self._concat_audio_files(retry_paths, output_path)
                return str(output_path)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def _synthesize_chunk_with_retry(
        self,
        text: str,
        output_path: Path,
        voice: str,
        speed: str,
        audio_format: str,
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                async_url = self._request_async_url(text, voice, speed, audio_format)
                self._download_when_ready(async_url, output_path, text[:140])
                if output_path.exists() and output_path.stat().st_size > 0:
                    return str(output_path)
                raise RuntimeError(f"FPT output file is empty: {output_path}")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(f"[FPT TTS] attempt {attempt}/{self.max_retries} failed: {exc}")
                if attempt < self.max_retries:
                    time.sleep(2.0)

        raise RuntimeError(
            f"FPT TTS failed after {self.max_retries} tries. Text={text[:140]}"
        ) from last_error

    def _request_async_url(self, text: str, voice: str, speed: str, audio_format: str) -> str:
        headers = {
            "api-key": self.api_key,
            "api_key": self.api_key,
            "voice": str(voice or "banmai").replace("fpt:", ""),
            "speed": str(speed or "0"),
            "format": str(audio_format or "mp3"),
            "Cache-Control": "no-cache",
        }
        response = self.session.post(
            FPT_TTS_URL,
            headers=headers,
            data=text.encode("utf-8"),
            timeout=self.timeout_sec,
        )

        if response.status_code != 200:
            raise RuntimeError(f"FPT API HTTP {response.status_code}: {response.text[:400]}")

        try:
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"FPT response is not JSON: {response.text[:400]}") from exc

        error = data.get("error", 0)
        if error not in (0, "0", None):
            raise RuntimeError(f"FPT API error: {data}")

        async_url = data.get("async") or data.get("message") or data.get("url")
        if not async_url or not str(async_url).startswith("http"):
            raise RuntimeError(f"FPT did not return a valid async URL: {data}")
        return str(async_url)

    def _download_when_ready(self, async_url: str, output_path: Path, text_preview: str = "") -> None:
        start_time = time.time()
        last_status = None
        last_body = ""
        last_error: Exception | None = None

        while time.time() - start_time <= self.max_wait_sec:
            elapsed = time.time() - start_time
            try:
                response = self.session.get(async_url, timeout=self.timeout_sec)
                last_status = response.status_code
                content_type = response.headers.get("Content-Type", "").lower()
                content = response.content or b""

                if response.status_code == 200:
                    looks_like_audio = (
                        "audio" in content_type
                        or "mpeg" in content_type
                        or "octet-stream" in content_type
                        or content.startswith(b"ID3")
                        or content[:2] in (b"\xff\xfb", b"\xff\xf3")
                    )
                    looks_like_html = b"<html" in content[:200].lower() or "html" in content_type
                    if len(content) > 1024 and looks_like_audio and not looks_like_html:
                        output_path.write_bytes(content)
                        return
                    last_body = self._safe_response_text(response)
                elif response.status_code in (202, 404, 408, 409, 425, 429, 500, 502, 503, 504):
                    last_body = self._safe_response_text(response)
                else:
                    raise RuntimeError(
                        f"FPT download HTTP {response.status_code}: {self._safe_response_text(response)[:400]}"
                    )
            except requests.RequestException as exc:
                last_error = exc
                last_body = str(exc)[:250]

            print(
                f"[FPT TTS] Waiting audio... {elapsed:.1f}/{self.max_wait_sec}s "
                f"| status={last_status} | {last_body[:160]}"
            )
            time.sleep(self.poll_interval_sec)

        raise TimeoutError(
            f"FPT audio not ready after {self.max_wait_sec}s. "
            f"Last status={last_status}, last_error={last_error}. Text={text_preview}"
        )

    @staticmethod
    def _safe_response_text(response: requests.Response) -> str:
        try:
            return re.sub(r"\s+", " ", response.text).strip()[:300]
        except Exception:
            return ""

    def _clean_text(self, text: str) -> str:
        text = str(text or "").strip()
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
        text = text.replace("\ufeff", "").replace("\u200b", "")
        text = text.replace("\u200c", "").replace("\u200d", "")
        text = text.replace(" ,", ",").replace(" .", ".")
        text = re.sub(r",\s*([.!?])", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        if text and text[-1] not in ".!?…":
            text += "."
        return text

    def _split_text(self, text: str, max_chars: int) -> List[str]:
        text = self._clean_text(text)
        if not text:
            return []
        max_chars = max(int(max_chars), self.min_chunk_chars)
        if len(text) <= max_chars:
            return [text]

        parts = self._split_by_regex(text, r"(?<=[.!?…])\s+")
        chunks = self._pack_parts(parts, max_chars)

        final_chunks: List[str] = []
        for chunk in chunks:
            if len(chunk) <= max_chars:
                final_chunks.append(chunk)
                continue
            comma_parts = self._split_by_regex(chunk, r"(?<=[,;:])\s+")
            comma_chunks = self._pack_parts(comma_parts, max_chars)
            for comma_chunk in comma_chunks:
                if len(comma_chunk) <= max_chars:
                    final_chunks.append(comma_chunk)
                else:
                    final_chunks.extend(self._hard_split_by_space(comma_chunk, max_chars))

        return [self._clean_text(c) for c in final_chunks if c and c.strip()]

    @staticmethod
    def _split_by_regex(text: str, pattern: str) -> List[str]:
        return [p.strip() for p in re.split(pattern, text) if p and p.strip()]

    @staticmethod
    def _pack_parts(parts: List[str], max_chars: int) -> List[str]:
        chunks: List[str] = []
        current = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            candidate = f"{current} {part}".strip() if current else part
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = part
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _hard_split_by_space(text: str, max_chars: int) -> List[str]:
        words = text.split()
        chunks: List[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip() if current else word
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                chunks.append(current)
            if len(word) > max_chars:
                chunks.extend(word[i : i + max_chars] for i in range(0, len(word), max_chars))
                current = ""
            else:
                current = word
        if current:
            chunks.append(current)
        return chunks

    def _concat_audio_files(self, input_paths: List[Path], output_path: Path) -> None:
        input_paths = [Path(p) for p in input_paths if Path(p).exists() and Path(p).stat().st_size > 0]
        if not input_paths:
            raise RuntimeError("No FPT audio chunks to concat.")
        if len(input_paths) == 1:
            shutil.copyfile(input_paths[0], output_path)
            return

        if shutil.which("ffmpeg"):
            list_file = output_path.parent / f"{output_path.stem}_concat.txt"
            try:
                lines = []
                for path in input_paths:
                    safe_path = str(path.resolve()).replace("\\", "/").replace("'", "'\\''")
                    lines.append(f"file '{safe_path}'")
                list_file.write_text("\n".join(lines), encoding="utf-8")
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        str(list_file),
                        "-c",
                        "copy",
                        str(output_path),
                    ],
                    check=True,
                )
                if output_path.exists() and output_path.stat().st_size > 0:
                    return
            except Exception as exc:  # noqa: BLE001
                print(f"[FPT TTS] ffmpeg concat failed, fallback byte concat: {exc}")
            finally:
                try:
                    list_file.unlink(missing_ok=True)
                except Exception:
                    pass

        with output_path.open("wb") as out:
            for path in input_paths:
                out.write(path.read_bytes())
