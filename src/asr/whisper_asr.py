from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
from faster_whisper import WhisperModel


class FasterWhisperASR:
    def __init__(
        self,
        model_name: str = "small",
        device: str = "auto",
        compute_type: str = "int8_float16",
        language: str = "auto",
        fallback_to_cpu: bool = True,
    ):
        self.model_name = model_name
        self.requested_device = device
        self.compute_type = compute_type
        self.language = language
        self.fallback_to_cpu = fallback_to_cpu
        self.model = self._load_model()

    def _resolve_device(self) -> str:
        if self.requested_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.requested_device

    def _load_model(self) -> WhisperModel:
        device = self._resolve_device()

        try:
            return WhisperModel(
                self.model_name,
                device=device,
                compute_type=self.compute_type,
            )
        except Exception as e:
            if self.fallback_to_cpu and device == "cuda":
                print(f"[ASR] CUDA load failed, falling back to CPU. Reason: {e}")
                return WhisperModel(
                    self.model_name,
                    device="cpu",
                    compute_type="int8",
                )
            raise

    def transcribe(self, audio_path: str | Path, video_name: str) -> Dict[str, Any]:
        language = None if self.language == "auto" else self.language

        segments_iter, info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=1,
            vad_filter=True,
        )

        segments: List[Dict[str, Any]] = []

        for seg in segments_iter:
            text = (seg.text or "").strip()
            if not text:
                continue

            segments.append(
                {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": text,
                }
            )

        full_text = " ".join(item["text"] for item in segments).strip()

        return {
            "video_name": video_name,
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "model_name": self.model_name,
            "full_text": full_text,
            "segments": segments,
        }