import re


BROKEN_ENDINGS = {
    "vì",
    "trong",
    "của",
    "là",
    "một",
    "để",
    "và",
    "rằng",
    "với",
    "từ",
    "cho",
    "như",
    "khi",
    "nếu",
    "nhưng",
    "hoặc",
}


def _last_word(text: str) -> str:
    words = re.findall(r"[A-Za-zÀ-ỹ0-9]+", text or "")
    if not words:
        return ""
    return words[-1].lower()


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-zÀ-ỹ0-9]+", text or ""))


def clean_text_for_fpt_tts(text: str) -> str:
    """
    Dọn text riêng cho FPT.AI TTS.
    Mục tiêu:
    - Không đổi nghĩa.
    - Giảm ngắt quá gắt ở câu cụt.
    - Tránh FPT đọc kiểu: 'Vì vậy, có một.' rồi nghỉ lâu.
    """

    text = str(text or "").strip()
    if not text:
        return text

    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text)

    # Xóa khoảng trắng thừa trước dấu câu
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    # Dọn lỗi dấu câu lặp
    text = re.sub(r",\s*\.", ".", text)
    text = re.sub(r"\.\s*,", ".", text)
    text = re.sub(r"([,.!?])\1+", r"\1", text)

    words = _word_count(text)
    last = _last_word(text)

    # Nếu segment quá ngắn mà kết thúc bằng dấu chấm,
    # FPT thường ngắt rất gắt -> bỏ dấu chấm.
    if words <= 5 and re.search(r"[.!?…]$", text):
        text = re.sub(r"[.!?…]+$", "", text).strip()

    # Nếu câu kết thúc bằng từ nối/cụm đang dang dở,
    # đổi dấu chấm thành dấu phẩy để FPT đọc mềm hơn.
    if last in BROKEN_ENDINGS:
        text = re.sub(r"[.!?…]+$", "", text).strip()
        if not text.endswith(","):
            text += ","

    # Giảm bớt ngắt do quá nhiều dấu phẩy sát nhau
    text = re.sub(r",\s*,+", ",", text)

    return text.strip()


def fpt_speed_from_rate(rate: str | None) -> str:
    """
    FPT.AI dùng speed dạng số, thường quanh -3..3.
    Project của mình lại đang dùng rate kiểu Edge: '+15%', '+20%', '0%', '-10%'.

    Nguyên tắc:
    - Không cho FPT ăn theo +15% của Edge.
    - FPT mặc định speed = 0.
    - Nếu user chọn chậm hơn thì mới cho FPT chậm nhẹ.
    """

    raw = str(rate or "").strip()

    match = re.search(r"[-+]?\d+", raw)
    if not match:
        return "0"

    value = int(match.group(0))

    # FPT vốn đã đọc nhanh, nên mọi rate dương của Edge đưa về 0
    if value > 0:
        return "0"

    # Cho phép chậm nhẹ nếu user chọn rate âm
    if value <= -20:
        return "-2"

    if value <= -5:
        return "-1"

    return "0"