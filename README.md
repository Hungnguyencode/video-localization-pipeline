# README.md

# Video Localization Pipeline

Project này giúp **việt hóa video ngắn**: tự động tạo transcript từ video, dịch sang tiếng Việt, chuẩn hóa segment, lồng TTS bằng Edge/FPT.AI, và xuất SRT + video đầu ra.

## Cấu trúc project
```
video-localization-pipeline/
├── app/
│   └── streamlit_app.py
├── src/
│   ├── tts/
│   ├── translation/
│   ├── subtitle/
│   └── utils/
├── data/
│   ├── input/
│   ├── transcripts/
│   └── tts_segments/
├── configs/
│   └── glossary files (.yaml)
├── .env.example
├── requirements.txt
└── README.md
```

## Cài đặt
1. Clone repo về máy:
```bash
git clone https://github.com/Hungnguyencode/video-localization-pipeline.git
cd video-localization-pipeline
```

2. Tạo virtual environment và cài dependencies:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

3. Tạo file `.env` dựa trên `.env.example` nếu muốn sử dụng giọng FPT.AI:
```
FPT_AI_API_KEY=your_fpt_ai_api_key_here
```

## Tính năng
- Upload video hoặc tải từ YouTube
- Tạo transcript tự động bằng ASR
- Dịch và chuẩn hóa segment tiếng Việt
- Tùy chỉnh voice/role và tốc độ đọc từng segment
- Xuất video đầu ra và file SRT đã căn chỉnh timestamp
- Kiểm tra chất lượng segment (Cảnh báo dài, TTS nhanh, segment cắt cụt...)

---

# RUN_DEMO.md

# Hướng dẫn chạy demo Streamlit

1. Kích hoạt virtual environment
```bash
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
```

2. Chạy Streamlit app
```bash
streamlit run app/streamlit_app.py
```

3. Trình duyệt sẽ mở giao diện web:
- Chọn **glossary** theo lĩnh vực
- Upload video hoặc nhập **YouTube URL**
- Nhấn **Bước 1: Tạo transcript + bản dịch**

4. Kiểm tra và chỉnh sửa:
- Tab **Bảng dịch**: xem toàn bộ segment, chỉnh sửa text/voice/rate
- Tab **Sửa / tách / phân vai**: gán speaker, merge/tách segment
- Tab **Kiểm tra chất lượng**: xem cảnh báo về TTS

5. Tạo video/SRT:
- Tab **Tạo lồng tiếng/video**
- Chọn giọng mặc định, tốc độ, audio mode
- Nhấn **Bước 2: Tạo video từ bản đã chỉnh**
- File đầu ra gồm video, SRT tự động căn chỉnh, JSON transcript, report HTML/JSON

6. File SRT tự động xuất ra: `data/transcripts/{video_stem}_auto_fitted.srt`

## Lưu ý
- Không push video/audio lớn lên GitHub
- `.env` chứa API key riêng nên không push
- Tùy chỉnh `chars_per_sec` trong code nếu muốn tốc độ TTS nhanh/chậm

