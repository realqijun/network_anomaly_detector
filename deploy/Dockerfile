FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y \
    libpcap-dev \
    tcpdump \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY deploy/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY deploy/ ./
COPY detector_runtime.py ./detector_runtime.py
RUN test -f model_bundle/manifest.json && test -f model_bundle/model.pt

ENV MODEL_BUNDLE_DIR=/app/model_bundle

EXPOSE 8000

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8000", "--capture-output", "--log-level", "info", "app:app"]
