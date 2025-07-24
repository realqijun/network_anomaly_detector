FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y \
    libpcap-dev \
    tcpdump \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/working
COPY working/anomaly_detector_model.pth /app/working/
COPY working/min_max_scaler.pkl /app/working/
COPY working/optimal_threshold.npy /app/working/

EXPOSE 8000

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8000", "--capture-output", "--log-level", "info", "app:app"]