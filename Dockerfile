# syntax=docker/dockerfile:1
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg — кадры видео/кружков, звук для распознавания речи, темп и паузы голоса
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg tzdata ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# Необязательный CA-сертификат для сервера за TLS-прокси: docker build --secret id=pip_ca,src=ca.crt .
RUN --mount=type=secret,id=pip_ca,required=false \
    if [ -f /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi; \
    pip install -r requirements.txt

COPY columbina ./columbina
COPY data ./data
COPY scripts ./scripts
COPY evals ./evals

RUN useradd --create-home --uid 1000 columbina \
 && mkdir -p /app/var/tmp \
 && chown -R columbina /app/var
USER columbina

ENV MEDIA_TMP_DIR=/app/var/tmp
CMD ["python", "-m", "columbina.main"]
