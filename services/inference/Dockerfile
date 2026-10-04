# syntax=docker/dockerfile:1
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY services/inference/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY services/inference/app/ app/
COPY services/inference/model/ model/
COPY configs/config.example.yaml configs/config.yaml
ENV PYTHONPATH=/app MFT_CONFIG=/app/configs/config.yaml MFT_INFERENCE_MODEL=heuristic
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
