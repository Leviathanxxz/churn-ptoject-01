# 1. Gunakan base image Python yang ringan
FROM python:3.11-slim

# 2. Set working directory di dalam container
WORKDIR /app

# 3. Copy file requirements dan install library
# Ini diletakkan di awal agar proses build lebih cepat (caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy seluruh struktur project kamu ke dalam folder /app di container
COPY . .

# 5. Ekspos port yang digunakan oleh FastAPI
EXPOSE 8000

# 6. Jalankan server uvicorn saat container dimulai
# Kita pakai host 0.0.0.0 agar bisa diakses dari luar container
CMD ["uvicorn", "src.g_app:app", "--host", "0.0.0.0", "--port", "8000"]