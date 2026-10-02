# syntax=docker/dockerfile:1

# Python inference node (Service 2). TabFM is loaded at startup; this image
# keeps the PyTorch runtime available for GPU passthrough when configured.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY services/inference/requirements.txt .
# CPU-only torch: the default PyPI wheel is the CUDA build, which drags in
# ~3.2GB of NVIDIA libraries for no benefit here. See services/inference/venv.sh.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Both packages, and they must sit side by side: app/main.py does
# `from model import ...`, so copying app/ alone gives ModuleNotFoundError at
# startup. The venv on the host sets PYTHONPATH for exactly this reason.
COPY services/inference/app/ app/
COPY services/inference/model/ model/

ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
