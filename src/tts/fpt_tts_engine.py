from __future__ import annotations

import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv


FPT_TTS_URL = "https://api.fpt.ai/hmi/tts/v5"


class FPTTTSEngine:
    """FPT.AI Text-to-Speech client.

    API key được đọc từ biến môi trường FPT_AI_API_KEY.
    Không hard-code API key vào source code.
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout_sec: int = 30,
        poll_interval_sec: float = 2.0,
        max_wait_sec: int = 120,
    ):
        # Cho phép chạy cả khi module được gọi ngoài Streamlit.
        load_dotenv(override=False)
        self.api_key = api_key or os.getenv("FPT_AI_API_KEY")
        self.timeout_sec = int(timeout_sec)
        self.poll_interval_sec = float(poll_interval_sec)
        self.max_wait_sec = int(max_wait_sec)

    def synthesize_one(
        self,
        text: str,
        output_path: str | Path,
        voice: str,
        speed: str = "0",
        audio_format: str = "mp3",
    ) -> str:
        if not self.api_key:
            raise RuntimeError(
                "FPT_AI_API_KEY is missing. Create .env in project root with: "
                "FPT_AI_API_KEY=your_api_key_here"
            )

        text = str(text or "").strip()
        if len(text) < 3:
            raise ValueError("FPT TTS text must contain at least 3 characters.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        async_url = self._request_async_url(
            text=text,
            voice=voice,
            speed=speed,
            audio_format=audio_format,
        )
        self._download_when_ready(async_url=async_url, output_path=output_path)

        if not output_path.exists() or output_path.stat().st_size <= 0:
            raise RuntimeError(f"FPT TTS output is empty: {output_path}")

        return str(output_path)

    def _request_async_url(
        self,
        text: str,
        voice: str,
        speed: str,
        audio_format: str,
    ) -> str:
        # FPT sample dùng header "api-key". Một số tài liệu cũ có "api_key".
        # Gửi cả hai để tương thích, nhưng "api-key" là header chính.
        headers = {
            "api-key": self.api_key,
            "api_key": self.api_key,
            "voice": str(voice),
            "speed": str(speed),
            "format": str(audio_format),
            "Cache-Control": "no-cache",
        }

        response = requests.post(
            FPT_TTS_URL,
            headers=headers,
            data=text.encode("utf-8"),
            timeout=self.timeout_sec,
        )
        response.raise_for_status()

        data = response.json()
        error = data.get("error", 0)
        if error not in (0, "0", None):
            raise RuntimeError(f"FPT TTS error: {data}")

        async_url = data.get("async") or data.get("message")
        if not async_url or not str(async_url).startswith("http"):
            raise RuntimeError(f"FPT TTS did not return a valid async URL: {data}")

        return str(async_url)

    def _download_when_ready(self, async_url: str, output_path: Path) -> None:
        start_time = time.time()
        last_status = None
        last_error: Exception | None = None

        while time.time() - start_time <= self.max_wait_sec:
            try:
                response = requests.get(async_url, timeout=self.timeout_sec)
                last_status = response.status_code
                content_type = response.headers.get("Content-Type", "").lower()
                content = response.content or b""

                if response.status_code == 200 and len(content) > 1024:
                    if "html" not in content_type:
                        output_path.write_bytes(content)
                        return
            except Exception as e:  # noqa: BLE001
                last_error = e

            time.sleep(self.poll_interval_sec)

        raise TimeoutError(
            f"FPT TTS audio was not ready after {self.max_wait_sec}s. "
            f"Last status={last_status}, last_error={last_error}, url={async_url}"
        )
