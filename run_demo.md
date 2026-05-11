# Hướng dẫn chạy demo

## 1. Cài FFmpeg

Cần cài FFmpeg và thêm vào PATH.

Kiểm tra:

```powershell
ffmpeg -version
ffprobe -version
2. Tạo môi trường ảo
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
3. Cài PyTorch GPU

Với NVIDIA GPU, có thể cài bản CUDA 12.1:

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

Nếu lỗi GPU, có thể cài CPU hoặc để faster-whisper fallback CPU.

4. Cài thư viện
pip install -r requirements.txt
5. Chạy bằng command line
python main_pipeline.py --video "data/input/demo.mp4"
6. Chạy giao diện Streamlit
streamlit run app/streamlit_app.py
7. Kết quả đầu ra

Sau khi chạy, kết quả nằm trong:

data/output/

Bao gồm:

video đã lồng tiếng tiếng Việt;
phụ đề tiếng Việt .srt;
transcript song ngữ;
audio lồng tiếng.