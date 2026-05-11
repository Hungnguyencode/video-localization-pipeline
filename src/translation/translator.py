from __future__ import annotations

import time
from typing import Dict, List

from src.translation.vi_postprocess import VietnamesePostProcessor, build_vi_postprocessor


class BaseTranslator:
    def translate_segments(self, segments: List[Dict]) -> List[Dict]:
        raise NotImplementedError


class IdentityTranslator(BaseTranslator):
    """
    Dùng để test pipeline khi chưa muốn dịch.
    Nó không dịch, chỉ copy text gốc sang vi_text.
    """

    def __init__(self, postprocessor: VietnamesePostProcessor | None = None):
        self.postprocessor = postprocessor

    def translate_segments(self, segments: List[Dict]) -> List[Dict]:
        output = []

        for item in segments:
            new_item = dict(item)
            new_item["source_text"] = item.get("text", "")
            new_item["vi_text"] = item.get("text", "")
            output.append(new_item)

        if self.postprocessor:
            output = self.postprocessor.postprocess_segments(output)

        return output


class GoogleFreeTranslator(BaseTranslator):
    """
    Dịch bằng deep-translator / Google Translate wrapper.
    Không cần API key, nhưng cần Internet và có thể bị giới hạn nếu gọi quá nhiều.
    Phù hợp cho demo video ngắn 3-7 phút.
    """

    def __init__(
        self,
        source_language: str = "en",
        target_language: str = "vi",
        sleep_seconds: float = 0.15,
        postprocessor: VietnamesePostProcessor | None = None,
    ):
        from deep_translator import GoogleTranslator

        self.source_language = source_language
        self.target_language = target_language
        self.sleep_seconds = float(sleep_seconds)
        self.postprocessor = postprocessor

        self.translator = GoogleTranslator(
            source=source_language,
            target=target_language,
        )

    def translate_text(self, text: str) -> str:
        text = (text or "").strip()

        if not text:
            return ""

        try:
            translated = self.translator.translate(text)
            translated = (translated or "").strip()
            return translated if translated else text

        except Exception as e:
            print(f"[TRANSLATE] Failed, fallback to source. Reason: {e}")
            return text

    def translate_segments(self, segments: List[Dict]) -> List[Dict]:
        output = []

        for idx, item in enumerate(segments, start=1):
            source_text = (item.get("text") or "").strip()
            vi_text = self.translate_text(source_text)

            new_item = dict(item)
            new_item["source_text"] = source_text
            new_item["vi_text"] = vi_text
            output.append(new_item)

            print(f"[TRANSLATE] Raw segment {idx}: {vi_text[:90]}")

            if self.sleep_seconds > 0:
                time.sleep(self.sleep_seconds)

        if self.postprocessor:
            output = self.postprocessor.postprocess_segments(output)

            for idx, item in enumerate(output, start=1):
                print(f"[POSTPROCESS] Done segment {idx}: {item.get('vi_text', '')[:90]}")

        return output


class LocalHFTranslator(BaseTranslator):
    """
    Dịch local bằng HuggingFace model.
    Mặc định dùng Helsinki-NLP/opus-mt-en-vi cho English -> Vietnamese.
    Chất lượng không cao, chỉ nên dùng khi muốn chạy offline.
    """

    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-en-vi",
        postprocessor: VietnamesePostProcessor | None = None,
    ):
        from transformers import MarianMTModel, MarianTokenizer

        self.model_name = model_name
        self.postprocessor = postprocessor
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(
            model_name,
            use_safetensors=True,
        )

    def translate_text(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )

        generated = self.model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
        )

        translated = self.tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
        )

        return translated.strip()

    def translate_segments(self, segments: List[Dict]) -> List[Dict]:
        output = []

        for idx, item in enumerate(segments, start=1):
            source_text = (item.get("text") or "").strip()
            vi_text = self.translate_text(source_text)

            new_item = dict(item)
            new_item["source_text"] = source_text
            new_item["vi_text"] = vi_text
            output.append(new_item)

            print(f"[TRANSLATE] Raw segment {idx}: {vi_text[:90]}")

        if self.postprocessor:
            output = self.postprocessor.postprocess_segments(output)

            for idx, item in enumerate(output, start=1):
                print(f"[POSTPROCESS] Done segment {idx}: {item.get('vi_text', '')[:90]}")

        return output


def build_translator(config: Dict) -> BaseTranslator:
    provider = config.get("provider", "identity")
    postprocessor = build_vi_postprocessor(config)

    if provider == "identity":
        return IdentityTranslator(postprocessor=postprocessor)

    if provider == "google_free":
        return GoogleFreeTranslator(
            source_language=config.get("source_language", "en"),
            target_language=config.get("target_language", "vi"),
            sleep_seconds=config.get("sleep_seconds", 0.15),
            postprocessor=postprocessor,
        )

    if provider == "local_hf":
        return LocalHFTranslator(
            model_name=config.get("model_name", "Helsinki-NLP/opus-mt-en-vi"),
            postprocessor=postprocessor,
        )

    raise ValueError(f"Unsupported translation provider: {provider}")