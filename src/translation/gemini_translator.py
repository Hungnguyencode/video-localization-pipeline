from __future__ import annotations

import re
import difflib
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from google import genai
from google.genai import types

from src.translation.vi_postprocess import VietnamesePostProcessor


class GeminiTranslator:
    """
    Translator dùng Gemini API cho pipeline việt hóa video.

    Tối ưu request/token:
    - Batch nhiều segment trong 1 request.
    - Cache bản dịch để chạy lại không tốn token.
    - Deduplicate câu giống nhau trước khi gửi Gemini.
    - Bắt Gemini trả JSON để giữ đúng id/segment.
    - Retry/backoff khi lỗi tạm thời hoặc rate limit.
    - Điều chỉnh xưng hô theo domain: cooking, education, news, technology, general.
    """

    def __init__(
        self,
        source_language: str = "en",
        target_language: str = "vi",
        model_name: str = "gemini-2.5-flash",
        api_key_env: str = "GEMINI_API_KEY",
        temperature: float = 0.2,
        max_output_tokens: int = 8192,
        batch_size: int = 12,
        max_batch_chars: int = 6000,
        request_sleep_seconds: float = 0.15,
        max_retries: int = 4,
        retry_base_seconds: float = 2.0,
        cache_enabled: bool = True,
        cache_path: str = "data/cache/gemini_translation_cache.json",
        fallback_to_source: bool = True,
        fallback_provider: str = "google_free",
        fallback_sleep_seconds: float = 0.15,
        glossary_prompt_max_items: int = 80,
        content_domain: str = "general",
        pronoun_style: str = "auto",
        postprocessor: VietnamesePostProcessor | None = None,
    ):
        self.source_language = source_language
        self.target_language = target_language
        self.model_name = model_name
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)

        self.batch_size = max(1, int(batch_size))
        self.max_batch_chars = max(500, int(max_batch_chars))
        self.request_sleep_seconds = max(0.0, float(request_sleep_seconds))

        self.max_retries = max(1, int(max_retries))
        self.retry_base_seconds = max(0.5, float(retry_base_seconds))

        self.cache_enabled = bool(cache_enabled)
        self.cache_path = Path(cache_path)

        self.fallback_to_source = bool(fallback_to_source)
        self.glossary_prompt_max_items = max(0, int(glossary_prompt_max_items))
        self.fallback_provider = str(fallback_provider or "none").strip().lower()
        self.fallback_sleep_seconds = max(0.0, float(fallback_sleep_seconds))
        self._google_fallback_translator = None

        self.content_domain = str(content_domain or "general").strip().lower()
        self.pronoun_style = str(pronoun_style or "auto").strip().lower()

        self.postprocessor = postprocessor

        api_key = (
            os.getenv(api_key_env)
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )

        if not api_key:
            raise ValueError(
                f"Không tìm thấy Gemini API key. "
                f"Hãy thêm {api_key_env}=YOUR_KEY vào file .env "
                f"hoặc set biến môi trường GEMINI_API_KEY."
            )

        self.client = genai.Client(api_key=api_key)
        self.cache: Dict[str, str] = self._load_cache()

    # =========================
    # Cache helpers
    # =========================

    def _cache_key(self, text: str) -> str:
        raw = f"{self.model_name}|{self.source_language}|{self.target_language}|{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load_cache(self) -> Dict[str, str]:
        if not self.cache_enabled:
            return {}

        try:
            if self.cache_path.exists():
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
        except Exception as e:
            print(f"[GEMINI] Không đọc được cache, bỏ qua. Reason: {e}")

        return {}

    def _save_cache(self) -> None:
        if not self.cache_enabled:
            return

        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self.cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"[GEMINI] Không ghi được cache, bỏ qua. Reason: {e}")

    # =========================
    # Fallback translator helpers
    # =========================

    def _get_google_fallback_translator(self):
        """
        Lazy init GoogleTranslator.
        Chỉ tạo khi Gemini thật sự lỗi/hết quota.
        """
        if self._google_fallback_translator is not None:
            return self._google_fallback_translator

        from deep_translator import GoogleTranslator

        self._google_fallback_translator = GoogleTranslator(
            source=self.source_language,
            target=self.target_language,
        )

        return self._google_fallback_translator


    def _translate_text_with_google_free(self, text: str) -> str:
        text = str(text or "").strip()

        if not text:
            return ""

        translator = self._get_google_fallback_translator()
        translated = translator.translate(text)
        translated = str(translated or "").strip()

        if self.fallback_sleep_seconds > 0:
            time.sleep(self.fallback_sleep_seconds)

        return translated if translated else text


    def _fallback_translate_batch(
        self,
        batch: List[Dict[str, str]],
        reason: Exception | str,
    ) -> Dict[str, str]:
        """
        Fallback sang google_free khi Gemini lỗi/hết quota/429/resource exhausted.
        Trả về dict: id -> vi_text.
        """
        if self.fallback_provider != "google_free":
            print(
                f"[GEMINI] Không bật google_free fallback "
                f"(fallback_provider={self.fallback_provider}). Reason: {reason}"
            )
            return {}

        print(
            f"[GEMINI] Gemini lỗi/hết quota, fallback sang google_free cho "
            f"{len(batch)} đoạn. Reason: {reason}"
        )

        result: Dict[str, str] = {}

        for item in batch:
            item_id = str(item.get("id", "")).strip()
            source_text = str(item.get("text", "")).strip()

            if not item_id:
                continue

            try:
                vi_text = self._translate_text_with_google_free(source_text)
                result[item_id] = vi_text

                print(f"[GOOGLE_FREE FALLBACK] {item_id}: {vi_text[:90]}")

            except Exception as e:
                print(f"[GOOGLE_FREE FALLBACK] Lỗi segment {item_id}: {e}")

                if self.fallback_to_source:
                    result[item_id] = source_text
                else:
                    result[item_id] = ""

        return result

    # =========================
    # Prompt helpers
    # =========================

    def _glossary_hint(self) -> str:
        """
        Đưa một phần glossary vào prompt.
        Có giới hạn số item để tránh prompt quá dài, tốn token.
        """
        replacements = (
            getattr(self.postprocessor, "replacements", None)
            if self.postprocessor
            else None
        )

        if not replacements:
            return ""

        rows: List[str] = []

        for pair in list(replacements)[: self.glossary_prompt_max_items]:
            try:
                source, target = pair[0], pair[1]
                rows.append(f"- {source} => {target}")
            except Exception:
                continue

        if not rows:
            return ""

        return (
            "\nƯu tiên áp dụng glossary/thuật ngữ sau khi phù hợp ngữ cảnh:\n"
            + "\n".join(rows)
        )

    def _build_pronoun_rule(self) -> str:
        """
        Tạo quy tắc xưng hô cho Gemini.

        pronoun_style:
        - auto: tự chọn theo content_domain
        - minh_ban
        - thay_co_em
        - toi_quy_vi
        - toi_ban
        - chung_ta
        """
        style = self.pronoun_style
        domain = self.content_domain

        if style == "auto":
            if domain in {"cooking", "food", "recipe", "lifestyle"}:
                style = "minh_ban"
            elif domain in {"education", "learning", "teaching", "school"}:
                style = "thay_co_em"
            elif domain in {"news", "report", "current_affairs"}:
                style = "toi_quy_vi"
            elif domain in {"technology", "tech", "ai"}:
                style = "toi_ban"
            else:
                style = "minh_ban"

        if style == "minh_ban":
            return (
                'Dùng nhất quán đại từ "mình" cho người nói và "bạn" cho người xem. '
                'Không đổi qua lại giữa "tôi" và "mình". '
                "Phong cách thân thiện, tự nhiên, phù hợp video YouTube/hướng dẫn."
            )

        if style == "thay_co_em":
            return (
                'Dùng xưng hô giáo dục: người giảng xưng "thầy/cô", '
                'người học là "em" hoặc "các em". '
                'Không dùng "mình - bạn" trong nội dung bài giảng. '
                'Nếu câu mang tính hướng dẫn chung, có thể dùng "chúng ta" để nghe tự nhiên. '
                'Vì chưa có speaker diarization để biết người nói là thầy hay cô, '
                'hãy dùng dạng trung tính "thầy/cô" khi cần tự xưng.'
            )

        if style == "toi_quy_vi":
            return (
                'Dùng phong cách tin tức/truyền hình: người nói xưng "tôi" hoặc "chúng tôi", '
                'người nghe là "quý vị". '
                'Không dùng "mình - bạn". '
                "Giọng văn trang trọng, rõ ràng, phù hợp bản tin hoặc phóng sự."
            )

        if style == "toi_ban":
            return (
                'Dùng nhất quán đại từ "tôi" cho người nói và "bạn" hoặc "các bạn" cho người xem. '
                "Phong cách rõ ràng, trung tính, phù hợp video công nghệ/giải thích."
            )

        if style == "chung_ta":
            return (
                'Ưu tiên dùng "chúng ta" khi hướng dẫn cùng làm. '
                'Khi người nói tự kể trải nghiệm cá nhân thì dùng "mình". '
                'Người xem có thể được gọi là "bạn".'
            )

        return (
            "Dùng đại từ nhất quán, tự nhiên theo ngữ cảnh. "
            "Tránh đổi qua lại giữa nhiều kiểu xưng hô trong cùng một video."
        )

    def _build_prompt(self, batch: List[Dict[str, str]]) -> str:
        payload = [
            {
                "id": item["id"],
                "text": item["text"],
            }
            for item in batch
        ]

        glossary_hint = self._glossary_hint()
        pronoun_rule = self._build_pronoun_rule()

        return f"""
Bạn là biên dịch viên chuyên việt hóa phụ đề và lồng tiếng video.

Nhiệm vụ:
Dịch các đoạn từ {self.source_language} sang {self.target_language}.

Domain nội dung:
{self.content_domain}

Yêu cầu dịch:
- Dịch tự nhiên, rõ nghĩa, phù hợp tiếng Việt nói.
- Ưu tiên câu ngắn gọn, dễ đọc bằng TTS.
- Không thêm thông tin ngoài nội dung gốc.
- Không bỏ sót số, tên riêng, thuật ngữ kỹ thuật.
- Nếu đoạn gốc bị cắt ngắn do ASR, hãy dịch mượt nhưng không tự bịa thêm ngữ cảnh.
- Giữ phong cách phù hợp với domain nội dung.
- {pronoun_rule}
- Không trả markdown, không giải thích.

Bắt buộc trả về JSON hợp lệ theo schema:
{{
  "translations": [
    {{
      "id": "id của input",
      "vi_text": "bản dịch tiếng Việt"
    }}
  ]
}}

Quy định output:
- Số lượng phần tử trong translations phải bằng số lượng input.
- Mỗi phần tử phải giữ đúng id.
- Không được đổi id.
- Không được trả thêm field không cần thiết.
{glossary_hint}

Dữ liệu cần dịch:
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    # =========================
    # Response parsing
    # =========================

    def _extract_json_object(self, text: str) -> str:
        """
        Trích JSON object hoàn chỉnh từ response nếu model lỡ trả thêm chữ ngoài JSON.
        Cách này an toàn hơn regex khi JSON có object lồng nhau.
        """
        text = (text or "").strip()

        start = text.find("{")
        if start < 0:
            raise ValueError("Không tìm thấy dấu mở JSON object '{' trong response.")

        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(text)):
            ch = text[i]

            if escape:
                escape = False
                continue

            if ch == "\\":
                escape = True
                continue

            if ch == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1

                if depth == 0:
                    return text[start : i + 1]

        raise ValueError("Không tìm thấy JSON object hoàn chỉnh trong response.")

    def _parse_response(self, response_text: str) -> Dict[str, str]:
        text = (response_text or "").strip()

        # Phòng khi model vẫn bọc ```json
        if text.startswith("```"):
            text = text.strip()
            if text.lower().startswith("```json"):
                text = text[7:].strip()
            elif text.startswith("```"):
                text = text[3:].strip()

            if text.endswith("```"):
                text = text[:-3].strip()

        try:
            data = json.loads(text)
        except Exception:
            json_text = self._extract_json_object(text)
            data = json.loads(json_text)

        translations = data.get("translations")

        if not isinstance(translations, list):
            raise ValueError("JSON Gemini không có field 'translations' dạng list.")

        result: Dict[str, str] = {}

        for item in translations:
            if not isinstance(item, dict):
                continue

            item_id = str(item.get("id", "")).strip()
            vi_text = str(item.get("vi_text", "")).strip()

            if item_id:
                result[item_id] = vi_text

        return result

    # =========================
    # Gemini call
    # =========================

    def _call_gemini(self, batch: List[Dict[str, str]]) -> Dict[str, str]:
        prompt = self._build_prompt(batch)
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                        response_mime_type="application/json",
                    ),
                )

                return self._parse_response(response.text)

            except Exception as e:
                last_error = e
                wait = self.retry_base_seconds * (2 ** (attempt - 1))
                print(
                    f"[GEMINI] Lỗi attempt {attempt}/{self.max_retries}: {e}. "
                    f"Retry sau {wait:.1f}s"
                )
                time.sleep(wait)

        raise RuntimeError(f"Gemini translation failed after retries: {last_error}")

    # =========================
    # Batch helpers
    # =========================

    def _make_batches(self, items: List[Dict[str, str]]) -> List[List[Dict[str, str]]]:
        batches: List[List[Dict[str, str]]] = []
        current: List[Dict[str, str]] = []
        current_chars = 0

        for item in items:
            text_len = len(item["text"])

            should_flush = (
                current
                and (
                    len(current) >= self.batch_size
                    or current_chars + text_len > self.max_batch_chars
                )
            )

            if should_flush:
                batches.append(current)
                current = []
                current_chars = 0

            current.append(item)
            current_chars += text_len

        if current:
            batches.append(current)

        return batches

    def _has_vietnamese_chars(self, text: str) -> bool:
        text = str(text or "").lower()
        return bool(
            re.search(
                r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệ"
                r"íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
                text,
            )
        )


    def _looks_untranslated(self, source_text: str, vi_text: str) -> bool:
        """
        Kiểm tra bản dịch có vẻ vẫn là tiếng Anh / chưa được dịch không.

        Dùng để xử lý case Gemini không lỗi API nhưng trả nguyên source cho 1 segment.
        """
        source_text = str(source_text or "").strip()
        vi_text = str(vi_text or "").strip()

        if not source_text:
            return False

        if not vi_text:
            return True

        source_lower = source_text.lower()
        vi_lower = vi_text.lower()

        # Nếu gần như giống hệt source thì coi như chưa dịch.
        similarity = difflib.SequenceMatcher(None, source_lower, vi_lower).ratio()
        if similarity >= 0.82:
            return True

        # Nếu output không có dấu/char tiếng Việt và chứa nhiều từ tiếng Anh thông dụng.
        has_vi_chars = self._has_vietnamese_chars(vi_text)

        english_words = re.findall(r"[a-zA-Z']+", vi_lower)
        if len(english_words) >= 6 and not has_vi_chars:
            common_en_words = {
                "if", "it", "is", "it's", "too", "high", "the", "will", "up",
                "fast", "but", "also", "just", "as", "quickly", "once", "you",
                "take", "them", "out", "of", "pan", "pancakes", "deflate",
            }

            hit_count = sum(1 for word in english_words if word in common_en_words)

            if hit_count >= 3:
                return True

        return False
    
    # =========================
    # Public API
    # =========================

    def translate_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []

        # source_text -> vi_text
        text_to_translation: Dict[str, str] = {}

        # Dùng dict để deduplicate nhưng vẫn giữ thứ tự insert.
        pending_unique: Dict[str, str] = {}

        # 1. Chuẩn hóa input và lấy cache nếu có
        for item in segments:
            source_text = (
                item.get("text")
                or item.get("source_text")
                or ""
            ).strip()

            new_item = dict(item)
            new_item["source_text"] = source_text
            output.append(new_item)

            if not source_text:
                text_to_translation[source_text] = ""
                continue

            cache_key = self._cache_key(source_text)

            if self.cache_enabled and cache_key in self.cache:
                text_to_translation[source_text] = self.cache[cache_key]
            else:
                pending_unique[source_text] = source_text

        # 2. Chuyển phần chưa có cache thành pending items
        pending_items = [
            {
                "id": str(i + 1),
                "text": text,
            }
            for i, text in enumerate(pending_unique.keys())
        ]

        id_to_text = {
            item["id"]: item["text"]
            for item in pending_items
        }

        # 3. Dịch theo batch
        batches = self._make_batches(pending_items)

        for batch_idx, batch in enumerate(batches, start=1):
            try:
                translated_by_id = self._call_gemini(batch)

                for item in batch:
                    source_text = id_to_text[item["id"]]
                    vi_text = translated_by_id.get(item["id"], "").strip()

                    # Gemini không lỗi API nhưng trả nguyên tiếng Anh/chưa dịch
                    # thì fallback riêng segment đó sang google_free.
                    if self._looks_untranslated(source_text, vi_text):
                        print(
                            f"[GEMINI] Segment {item['id']} có vẻ chưa được dịch, "
                            f"fallback sang google_free."
                        )

                        fallback_by_id = self._fallback_translate_batch(
                            [item],
                            reason="Gemini output looks untranslated",
                        )

                        vi_text = str(fallback_by_id.get(item["id"], "")).strip()

                    if not vi_text and self.fallback_to_source:
                        vi_text = source_text

                    text_to_translation[source_text] = vi_text

                    # Cache cả kết quả Gemini hoặc fallback nếu có kết quả.
                    if vi_text:
                        self.cache[self._cache_key(source_text)] = vi_text


                self._save_cache()

                print(
                    f"[GEMINI] Dịch xong batch {batch_idx}/{len(batches)}: "
                    f"{len(batch)} đoạn"
                )

                if self.request_sleep_seconds > 0:
                    time.sleep(self.request_sleep_seconds)

            except Exception as e:
                print(f"[GEMINI] Batch {batch_idx} lỗi. Reason: {e}")

                # Gemini lỗi/hết API/quota thì fallback sang google_free.
                fallback_by_id = self._fallback_translate_batch(batch, reason=e)

                for item in batch:
                    source_text = id_to_text[item["id"]]
                    vi_text = str(fallback_by_id.get(item["id"], "")).strip()

                    if not vi_text:
                        vi_text = source_text if self.fallback_to_source else ""

                    text_to_translation[source_text] = vi_text

                    # Cache cả kết quả fallback nếu đã dịch được sang tiếng Việt.
                    # Lần chạy sau sẽ không tốn Gemini/Google request nữa.
                    if vi_text and vi_text != source_text:
                        self.cache[self._cache_key(source_text)] = vi_text

                self._save_cache()

        # 4. Gán kết quả về đúng segment ban đầu
        for idx, item in enumerate(output, start=1):
            source_text = item.get("source_text", "")
            item["vi_text"] = text_to_translation.get(source_text, source_text)

            print(f"[GEMINI] Segment {idx}: {item.get('vi_text', '')[:90]}")

        # 5. Postprocess tiếng Việt như pipeline cũ
        if self.postprocessor:
            output = self.postprocessor.postprocess_segments(output)

            for idx, item in enumerate(output, start=1):
                print(f"[POSTPROCESS] Done segment {idx}: {item.get('vi_text', '')[:90]}")

        return output