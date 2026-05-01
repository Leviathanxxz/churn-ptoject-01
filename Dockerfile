# 1. Gunakan slim image agar ringan
FROM python:3.13-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements dan install langsung
# Kita skip apt-get install untuk menghindari error 100
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy folder models dan src ke dalam container
COPY models/ ./models/
COPY src/ ./src/

# 5. Tambahkan environment variable agar Python bisa menemukan folder 'src'
ENV PYTHONPATH="/app"

# 6. Expose port Streamlit
# EXPOSE 8501 

# 7. Jalankan Streamlit dengan path yang benar
CMD ["sh", "-c", "streamlit run src/h_app_streamlit.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]