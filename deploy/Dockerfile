FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y \
    libpcap0.8 \
    tcpdump \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -m unittest discover -s tests -v

EXPOSE 8080

CMD ["sh", "-c", "exec gunicorn -w 1 -b 0.0.0.0:${PORT:-8080} --timeout 300 --capture-output --log-level info app:app"]
