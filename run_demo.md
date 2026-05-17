# RUN_DEMO.md

# Hướng dẫn chạy demo Video Localization Pipeline

## 1. Kích hoạt virtual environment
```bash
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
```

## 2. Chạy Streamlit App
```bash
streamlit run app/streamlit_app.py
```

## 3. Giao diện web
- Chọn **glossary** theo lĩnh vực.
- Upload video hoặc nhập **YouTube URL**.
- Nhấn **Bước 1: Tạo transcript + bản dịch**.

## 4. Kiểm tra và chỉnh sửa transcript
- **Tab Bảng dịch**: xem toàn bộ segment, chỉnh sửa text/voice/rate.
- **Tab Sửa / tách / phân vai**: gán speaker, merge/tách segment.
- **Tab Kiểm tra chất lượng**: xem cảnh báo về TTS và độ dài segment.

## 5. Xuất video và SRT
- **Tab Tạo lồng tiếng/video**: chọn giọng mặc định, tốc độ, audio mode.
- Nhấn **Bước 2: Tạo video từ bản đã chỉnh**.
- File đầu ra:
  - Video
  - File SRT tự động căn chỉnh: `data/transcripts/{video_stem}_auto_fitted.srt`
  - JSON transcript
  - Report HTML/JSON

## 6. Lưu ý
- Không push video/audio lớn lên GitHub.
- `.env` chứa API key riêng nên không push.
- Có thể chỉnh `chars_per_sec` trong code nếu muốn tốc độ TTS nhanh/chậm.
- Tab Quality check nên kiểm tra trước khi render video để tránh TTS quá nhanh hoặc segment quá dài.