# FastAPI + TensorFlow model inference container
# Using the official TensorFlow image avoids manylinux / system-lib headaches.
FROM tensorflow/tensorflow:2.20.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# (Optional but helps Pillow/OpenCV-style deps on slim images; safe here too)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

# Keep the same relative layout expected by the code: /app/api and /app/models
COPY api /app/api
COPY models /app/models

# Most platforms expose port 8080; we also respect $PORT if provided.
EXPOSE 8080

CMD ["bash", "-lc", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
