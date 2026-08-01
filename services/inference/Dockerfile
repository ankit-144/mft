# syntax=docker/dockerfile:1

# Python inference node (Service 2). TabFM is loaded at startup; this image
# keeps the PyTorch runtime available for GPU passthrough when configured.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY services/inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/inference/app/ app/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
