# Yoruba ASR web app (CPU). The trained models are NOT baked into the image: they are mounted from
# ./models at run time (see docker-compose.yml), so the image stays small and models can be swapped.
#   docker compose up -d --build      then open http://localhost:7870
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=20 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7870 \
    MPLBACKEND=Agg

# ffmpeg + libsndfile: decoding of mp3/m4a/ogg uploads and microphone recordings; curl: health check.
# Retries/timeouts make the one-time download survive slow or unstable internet connections.
RUN printf 'Acquire::Retries "10";\nAcquire::http::Timeout "300";\nAcquire::https::Timeout "300";\n' \
      > /etc/apt/apt.conf.d/80-retries \
 && apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first (much smaller than the default CUDA build), then the app requirements
COPY app/requirements.txt /tmp/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 \
 && pip install -r /tmp/requirements.txt

COPY src/ src/
COPY app/ app/
RUN mkdir -p models results

EXPOSE 7870
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fs http://127.0.0.1:7870/ > /dev/null || exit 1

CMD ["python", "app/app.py"]
