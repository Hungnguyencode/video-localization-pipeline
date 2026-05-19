# README.md

# Video Localization Pipeline

# 🎬 Video Localization Pipeline

## Mục đích
Project này là một pipeline hoàn chỉnh để **việt hóa video ngắn**, bao gồm các bước:
1. Tự động tạo transcript từ audio/video bằng **ASR (Whisper/FasterWhisper)**.
2. Dịch sang tiếng Việt với **Gemini Translator**, có fallback sang Google Free hoặc Local HF nếu cần.
3. Hậu xử lý tiếng Việt: chuẩn hóa, tách câu, chuyển số thành chữ.
4. Gán voice/TTS (Edge hoặc FPT.AI) và tạo audio.
5. Lồng tiếng, render video, chèn phụ đề và xuất báo cáo demo.

Mục tiêu: giảm thời gian thủ công, giữ chất lượng audio/TTS và đảm bảo đồng bộ phụ đề.

---

## Chức năng chính

### 1. Transcript & dịch
- ASR segment tự động, merge các segment bị cắt lẻ.
- Tích hợp **glossary theo domain** (Cooking, Education, News, Technology) để dịch chuẩn thuật ngữ.
- Hỗ trợ quick replacements từ UI.

### 2. Editor & postprocess
- Xem, sửa, gán speaker/voice, rate cho từng segment.
- Split/merge segment linh hoạt.
- Kiểm tra chất lượng TTS:
  - Segment dài/short.
  - Chữ nhiều quá so với thời lượng.
  - Kiểm tra dấu câu, chữ thường đầu câu, số tách sai.
  
### 3. TTS & Audio
- Tích hợp **Edge TTS** và **FPT.AI TTS**.
- Chạy đồng thời voice/rate cache, fallback khi lỗi FPT.
- Chuẩn hóa audio, trim silence, speedup segment dài.

### 4. Render Video & Subtitle
- Render video với audio lồng tiếng.
- Chèn phụ đề tiếng Việt (burn-in) hoặc xuất SRT.
- Tùy chỉnh font, size, outline, margin.
- Output đầy đủ: video, SRT, JSON, report HTML/JSON.

### 5. Báo cáo & kiểm tra
- Báo cáo demo chi tiết:
  - Số segment, số đã chỉnh, cảnh báo chất lượng.
  - Thống kê voice, rate, speaker.
  - Link download video/SRT/JSON/report.

---

## Cấu trúc project
```
video-localization-pipeline/
├── app/
│   └── streamlit_app.py
├── src/
│   ├── tts/
│   ├── alignment/
│   ├── asr/
│   ├── audio/
│   ├── ingest/
│   ├── subtitle/
│   ├── translation/
│   ├── video_render/
│   └── utils/
├── data/
│   ├── input/
│   ├── output/
│   ├── audio/
│   ├── cache/
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

## Điểm mạnh
- **Full pipeline:** ASR → dịch → TTS → render video.
- **Multi-TTS provider:** Edge + FPT.AI, có fallback.
- **Human-in-the-loop:** sửa segment, gán voice, rate.
- **Chất lượng cao:** kiểm tra TTS, kiểm tra dấu câu, cảnh báo segment dài.
- **Flexible:** merge/split segment, glossary domain, quick replacements.
- **Export đa dạng:** video, SRT, JSON, report HTML/JSON.

## Điểm hạn chế
- **Tốc độ FPT chậm**, cần cache audio.
- **Gemini API có hạn mức**, fallback sang Google/Local.
- **Segment merge chưa tự động hoàn toàn**, vẫn cần kiểm tra manual.
- **Yêu cầu môi trường:** FFmpeg, Python >=3.10, GPU nếu dùng model Whisper lớn.

---

## License
MIT License

---

