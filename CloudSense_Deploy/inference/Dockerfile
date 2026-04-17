# ── CloudSense Inference Server ──────────────────────────────────────────────
# Base image: slim Python (no GPU needed for inference on EC2 free tier)
FROM python:3.11-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

# Model artifacts will be mounted at /app/model_export (via volume or COPY)
# Uncomment below if you want to bake the model into the image:
# COPY model_export/ ./model_export/

EXPOSE 8000

ENV MODEL_DIR=/app/model_export

# Start server with 2 workers (tune for t2.micro: 1 vCPU, 1 GB RAM)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
