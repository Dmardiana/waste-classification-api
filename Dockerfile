FROM python:3.10-slim

WORKDIR /app

# 1. Install dependensi sistem yang diperlukan oleh Pillow & TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy dan install requirements terlebih dahulu (Optimasi caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy seluruh kode program dan folder model setelah library terinstall
COPY . .

# 4. Expose port wajib Hugging Face
EXPOSE 7860

# 5. Jalankan Uvicorn pada port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
