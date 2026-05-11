# Video Localization Pipeline

Đề tài: Nghiên cứu xây dựng pipeline dữ liệu Việt hóa video giáo dục bằng dịch phụ đề và lồng tiếng tự động.

## Chức năng bản core

- Nhận 1 video đầu vào.
- Tách âm thanh bằng FFmpeg.
- Nhận dạng giọng nói bằng faster-whisper.
- Tạo transcript và phụ đề gốc.
- Dịch phụ đề sang tiếng Việt.
- Tổng hợp giọng nói tiếng Việt bằng edge-tts.
- Căn chỉnh audio tiếng Việt theo timestamp.
- Ghép giọng tiếng Việt vào video.
- Xuất video đã Việt hóa và file phụ đề `.srt`.

## Phạm vi

- Xử lý từng video đơn lẻ.
- Video ngắn khoảng 3–7 phút.
- Phù hợp với video học tập, giáo dục, tin tức hoặc hướng dẫn ngắn.
- Chưa xử lý batch.
- Chưa thực hiện khớp khẩu hình.
- Chưa clone giọng gốc ở bản core.
- Chưa tự động phân tách nhiều người nói ở bản core.

## Công nghệ sử dụng

- Python
- Streamlit
- FFmpeg
- faster-whisper
- HuggingFace Transformers
- edge-tts
- pydub
- srt