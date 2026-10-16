# Gives us Python + ffmpeg in one image. Works on Render, Railway, Fly, HF Spaces, any VPS.
FROM python:3.11-slim

# ffmpeg is required for decoding, mixing and key changes; it is not a pip package.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so Docker caches this layer when only the app code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code.
COPY server.py karaoke.html ./

# Model weights download on first run; keep them inside the image dir so they persist with the disk.
ENV HF_HOME=/app/cache/hf \
    TORCH_HOME=/app/cache/torch \
    XDG_CACHE_HOME=/app/cache/xdg \
    PYTHONUNBUFFERED=1

# Hosts inject PORT; server.py reads it and binds 0.0.0.0 automatically.
EXPOSE 8000
CMD ["python3", "server.py"]
