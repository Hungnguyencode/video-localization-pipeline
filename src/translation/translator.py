from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

from src.translation.vi_postprocess import VietnamesePostProcessor, build_vi_postprocessor
PROMPT_VERSION = "prompt_v3_news_tts"


class BaseTranslator:
    def translate_segments(self, segments: List[Dict]) -> List[Dict]:
        raise NotImplementedError


class IdentityTranslator(BaseTranslator):
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
        self.translator = GoogleTranslator(source=source_language, target=target_language)

    def translate_text(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        try:
            translated = self.translator.translate(text)
            return (translated or text).strip()
        except Exception as exc:
            print(f"[TRANSLATE][GoogleFree] failed, fallback source. Reason={exc}")
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
            print(f"[TRANSLATE][GoogleFree] segment {idx}: {vi_text[:90]}")
            if self.sleep_seconds > 0:
                time.sleep(self.sleep_seconds)
        if self.postprocessor:
            output = self.postprocessor.postprocess_segments(output)
        return output


class LocalHFTranslator(BaseTranslator):
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-vi", postprocessor: VietnamesePostProcessor | None = None):
        from transformers import MarianMTModel, MarianTokenizer

        self.model_name = model_name
        self.postprocessor = postprocessor
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name, use_safetensors=True)

    def translate_text(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        generated = self.model.generate(**inputs, max_length=256, num_beams=4)
        return self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()

    def translate_segments(self, segments: List[Dict]) -> List[Dict]:
        output = []
        for idx, item in enumerate(segments, start=1):
            source_text = (item.get("text") or "").strip()
            vi_text = self.translate_text(source_text)
            new_item = dict(item)
            new_item["source_text"] = source_text
            new_item["vi_text"] = vi_text
            output.append(new_item)
            print(f"[TRANSLATE][LocalHF] segment {idx}: {vi_text[:90]}")
        if self.postprocessor:
            output = self.postprocessor.postprocess_segments(output)
        return output


class GeminiTranslator(BaseTranslator):
    """
    Gemini translator dạng batch để thay google_free.
    Có cache theo nội dung text để chạy lại nhanh và ít tốn API hơn.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_key_env: str = "GEMINI_API_KEY",
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        max_output_tokens: int = 8192,
        batch_size: int = 12,
        max_batch_chars: int = 6000,
        request_sleep_seconds: float = 0.15,
        max_retries: int = 4,
        retry_base_seconds: float = 2.0,
        cache_enabled: bool = True,
        cache_path: str = "data/cache/gemini_translation_cache_v2.json",
        fallback_provider: str = "google_free",
        source_language: str = "en",
        target_language: str = "vi",
        content_domain: str = "general",
        pronoun_style: str = "auto",
        context_enabled: bool = True,
        context_window: int = 1,
        context_max_chars: int = 320,
        postprocessor: VietnamesePostProcessor | None = None,
    ):
        load_dotenv()
        self.api_key = (api_key or os.getenv(api_key_env) or "").strip()
        self.model_name = model_name
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.batch_size = int(batch_size)
        self.max_batch_chars = int(max_batch_chars)
        self.request_sleep_seconds = float(request_sleep_seconds)
        self.max_retries = int(max_retries)
        self.retry_base_seconds = float(retry_base_seconds)
        self.cache_enabled = bool(cache_enabled)
        self.cache_path = Path(cache_path)
        self.fallback_provider = fallback_provider
        self.source_language = source_language
        self.target_language = target_language
        self.content_domain = content_domain
        self.pronoun_style = pronoun_style

        # Level 2: context-aware translation
        self.context_enabled = bool(context_enabled)
        self.context_window = max(int(context_window), 0)
        self.context_max_chars = max(int(context_max_chars), 80)

        self.postprocessor = postprocessor
        self.cache: Dict[str, str] = self._load_cache()

        if not self.api_key:
            raise RuntimeError(f"Thiếu {api_key_env} trong file .env")

        self._client_kind = "none"
        self._model = None
        self._client = None
        self._init_gemini_client()

    def _init_gemini_client(self) -> None:
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
            self._client_kind = "google-generativeai"
            return
        except Exception:
            pass

        try:
            from google import genai as new_genai

            self._client = new_genai.Client(api_key=self.api_key)
            self._client_kind = "google-genai"
            return
        except Exception as exc:
            raise RuntimeError(
                "Chưa cài thư viện Gemini. Hãy chạy: pip install google-generativeai"
            ) from exc

    def _get_segment_source_text(self, segment: Dict) -> str:
        """
        Lấy text nguồn của segment theo nhiều key để tương thích pipeline.
        """
        return str(
            segment.get("text")
            or segment.get("source_text")
            or segment.get("original_text")
            or ""
        ).strip()

    def _clip_context_text(self, text: str, max_chars: int | None = None) -> str:
        """
        Cắt context để prompt không quá dài.
        """
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        limit = max_chars or self.context_max_chars

        if len(text) <= limit:
            return text

        # Giữ phần cuối vì thường sát với câu hiện tại hơn
        return "..." + text[-limit:].strip()

    def _build_context_for_index(self, segments: List[Dict], index: int) -> tuple[str, str]:
        """
        Lấy context trước/sau cho segment hiện tại.
        """
        if not self.context_enabled or self.context_window <= 0:
            return "", ""

        before_parts: List[str] = []
        after_parts: List[str] = []

        start_idx = max(0, index - self.context_window)
        end_idx = min(len(segments), index + self.context_window + 1)

        for i in range(start_idx, index):
            text = self._get_segment_source_text(segments[i])
            if text:
                before_parts.append(text)

        for i in range(index + 1, end_idx):
            text = self._get_segment_source_text(segments[i])
            if text:
                after_parts.append(text)

        context_before = self._clip_context_text(" ".join(before_parts))
        context_after = self._clip_context_text(" ".join(after_parts))

        return context_before, context_after

    def _attach_context_to_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Tạo bản copy của segments và thêm context_before/context_after.
        """
        output: List[Dict] = []

        for idx, segment in enumerate(segments):
            new_segment = dict(segment)
            context_before, context_after = self._build_context_for_index(segments, idx)
            new_segment["context_before"] = context_before
            new_segment["context_after"] = context_after
            output.append(new_segment)

        return output

    def _load_cache(self) -> Dict[str, str]:
        if not self.cache_enabled or not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_cache(self) -> None:
        if not self.cache_enabled:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")

    def _cache_key(self, text: str, context_before: str = "", context_after: str = "") -> str:
        """
        Cache key có tính cả context.
        Cùng một câu nhưng ngữ cảnh khác thì bản dịch có thể khác.
        """
        raw = "|".join(
            [
                str(self.model_name),
                str(self.content_domain),
                str(self.pronoun_style),
                PROMPT_VERSION,
                "context_v1",
                str(text or ""),
                str(context_before or ""),
                str(context_after or ""),
            ]
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _domain_instruction(self) -> str:
        mapping = {
            "general": "Dịch tự nhiên, dễ nghe, phù hợp video phổ thông.",
            "education": "Dịch theo phong cách giáo dục, rõ ý, dễ hiểu. Có thể xưng hô thầy/cô - em/các em nếu hợp ngữ cảnh.",
            "news": "Dịch theo phong cách bản tin, trang trọng vừa phải. Có thể xưng hô tôi/chúng tôi - quý vị.",
            "cooking": "Dịch thân thiện, hợp video hướng dẫn. Có thể xưng hô mình - bạn.",
            "technology": "Dịch rõ ràng, thuật ngữ công nghệ chính xác, không quá văn vẻ.",
        }
        return mapping.get(self.content_domain, mapping["general"])

    def _make_prompt(self, batch: List[Dict]) -> str:
        items = []

        for i, item in enumerate(batch, start=1):
            text = self._get_segment_source_text(item)
            items.append(
                {
                    "id": i,
                    "context_before": item.get("context_before", ""),
                    "text": text,
                    "context_after": item.get("context_after", ""),
                }
            )

        return (
            "Bạn là hệ thống dịch phụ đề/lồng tiếng video sang tiếng Việt.\n"
            "Nhiệm vụ:\n"
            "- Dịch CHỈ trường `text` sang tiếng Việt.\n"
            "- `context_before` và `context_after` chỉ dùng để hiểu ngữ cảnh, KHÔNG được dịch gộp vào kết quả.\n"
            "- Không thêm giải thích ngoài bản dịch.\n"
            "- Giữ nguyên số thứ tự id.\n"
            "- Trả về DUY NHẤT JSON array, mỗi phần tử có dạng {\"id\": 1, \"vi_text\": \"...\"}.\n"
            "- Dịch tự nhiên như lời nói, ngắn gọn để dễ lồng tiếng.\n"
            "- Nếu là bản tin/news, dùng giọng trang trọng, tự nhiên, không dùng 'anh/chị' trừ khi source thật sự là hội thoại trực tiếp.\n"
            "- Không đoán mò tên riêng. Nếu không chắc tên người/công ty, hãy giữ gần nguyên dạng tiếng Anh.\n"
            "- Tránh dịch quá dài; ưu tiên câu ngắn, dễ đọc, dễ lồng tiếng.\n"
            "- Không để câu kết thúc bằng dấu phẩy hoặc cụm từ cụt như 'của', 'vì', 'trong', 'và'.\n"
            "- Giữ thống nhất thuật ngữ, tên riêng, đại từ xưng hô giữa các segment.\n"
            "- Nếu gặp đại từ như it/they/this/that/he/she, hãy dựa vào context để dịch đúng đối tượng.\n"
            f"- Phong cách: {self._domain_instruction()}\n"
            f"- Pronoun style: {self.pronoun_style}.\n\n"
            "Dữ liệu cần dịch là JSON array sau:\n"
            f"{json.dumps(items, ensure_ascii=False, indent=2)}\n\n"
            "Output bắt buộc:\n"
            "[\n"
            "  {\"id\": 1, \"vi_text\": \"Bản dịch tiếng Việt của text\"}\n"
            "]"
        )

    def _call_gemini(self, prompt: str) -> str:
        for attempt in range(1, self.max_retries + 1):
            try:
                if self._client_kind == "google-generativeai":
                    response = self._model.generate_content(
                        prompt,
                        generation_config={
                            "temperature": self.temperature,
                            "max_output_tokens": self.max_output_tokens,
                        },
                    )
                    return (getattr(response, "text", "") or "").strip()

                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                return (getattr(response, "text", "") or "").strip()
            except Exception as exc:
                wait = self.retry_base_seconds * attempt
                print(f"[TRANSLATE][Gemini] attempt {attempt}/{self.max_retries} failed: {exc}")
                time.sleep(wait)
        raise RuntimeError("Gemini translation failed after retries.")

    @staticmethod
    def _extract_json_array(text: str) -> List[Dict]:
        text = (text or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json\n", "", 1).replace("JSON\n", "", 1).strip()
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Gemini response is not JSON array: {text[:300]}")
        return json.loads(text[start : end + 1])

    def _batch_segments(self, segments: List[Dict]) -> List[List[Dict]]:
        batches: List[List[Dict]] = []
        current: List[Dict] = []
        current_chars = 0

        for item in segments:
            text = self._get_segment_source_text(item)
            context_before = item.get("context_before", "")
            context_after = item.get("context_after", "")

            # Tính cả context để tránh prompt quá dài
            item_chars = len(text) + len(context_before) + len(context_after)
            candidate_chars = current_chars + item_chars

            if current and (len(current) >= self.batch_size or candidate_chars > self.max_batch_chars):
                batches.append(current)
                current = []
                current_chars = 0

            current.append(item)
            current_chars += item_chars

        if current:
            batches.append(current)

        return batches

    def translate_segments(self, segments: List[Dict]) -> List[Dict]:
        # Level 2: gắn context trước/sau vào từng segment
        output: List[Dict] = self._attach_context_to_segments(segments)
        to_translate: List[tuple[int, Dict]] = []

        for idx, item in enumerate(output):
            source_text = self._get_segment_source_text(item)
            item["source_text"] = source_text

            context_before = item.get("context_before", "")
            context_after = item.get("context_after", "")

            key = self._cache_key(source_text, context_before, context_after)

            if self.cache_enabled and key in self.cache:
                item["vi_text"] = self.cache[key]
            else:
                to_translate.append((idx, item))

        batches = self._batch_segments([item for _, item in to_translate])
        cursor = 0

        for batch_no, batch in enumerate(batches, start=1):
            print(
                f"[TRANSLATE][Gemini][Context] batch {batch_no}/{len(batches)} "
                f"| segments={len(batch)} "
                f"| context_enabled={self.context_enabled}"
            )

            prompt = self._make_prompt(batch)
            raw = self._call_gemini(prompt)
            parsed = self._extract_json_array(raw)
            by_id = {int(x.get("id")): (x.get("vi_text") or "").strip() for x in parsed}

            for local_id, item in enumerate(batch, start=1):
                global_idx = to_translate[cursor][0]

                source_text = item["source_text"]
                context_before = item.get("context_before", "")
                context_after = item.get("context_after", "")

                vi_text = by_id.get(local_id) or source_text

                output[global_idx]["vi_text"] = vi_text

                cache_key = self._cache_key(source_text, context_before, context_after)
                self.cache[cache_key] = vi_text

                cursor += 1

            self._save_cache()

            if self.request_sleep_seconds > 0:
                time.sleep(self.request_sleep_seconds)

        # Không cần lưu context_before/context_after ra transcript cuối nếu không muốn
        for item in output:
            item.pop("context_before", None)
            item.pop("context_after", None)

        if self.postprocessor:
            output = self.postprocessor.postprocess_segments(output)

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

    if provider == "gemini":
        gemini_cfg = config.get("gemini", {}) or {}
        return GeminiTranslator(
            api_key=gemini_cfg.get("api_key"),
            api_key_env=gemini_cfg.get("api_key_env", "GEMINI_API_KEY"),
            model_name=gemini_cfg.get("model_name", "gemini-2.5-flash"),
            temperature=gemini_cfg.get("temperature", 0.2),
            max_output_tokens=gemini_cfg.get("max_output_tokens", 8192),
            batch_size=gemini_cfg.get("batch_size", 12),
            max_batch_chars=gemini_cfg.get("max_batch_chars", 6000),
            request_sleep_seconds=gemini_cfg.get("request_sleep_seconds", 0.15),
            max_retries=gemini_cfg.get("max_retries", 4),
            retry_base_seconds=gemini_cfg.get("retry_base_seconds", 2.0),
            cache_enabled=gemini_cfg.get("cache_enabled", True),
            cache_path=gemini_cfg.get("cache_path", "data/cache/gemini_translation_cache_v2.json"),
            fallback_provider=gemini_cfg.get("fallback_provider", "google_free"),
            source_language=config.get("source_language", "en"),
            target_language=config.get("target_language", "vi"),
            content_domain=gemini_cfg.get("content_domain", "general"),
            pronoun_style=gemini_cfg.get("pronoun_style", "auto"),

            # Level 2: context-aware translation
            # Đọc được cả khi bạn để trong translation: hoặc translation.gemini:
            context_enabled=gemini_cfg.get("context_enabled", config.get("context_enabled", True)),
            context_window=gemini_cfg.get("context_window", config.get("context_window", 1)),
            context_max_chars=gemini_cfg.get("context_max_chars", config.get("context_max_chars", 320)),

            postprocessor=postprocessor,
        )

    raise ValueError(f"Unsupported translation provider: {provider}")
