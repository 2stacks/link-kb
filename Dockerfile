FROM python:3.12-slim

# System deps for ChromaDB sqlite-vec backend
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libsqlite3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY templates/ ./templates/

RUN mkdir -p /data

ENV PYTHONUNBUFFERED=1
ENV ANONYMIZED_TELEMETRY=False
ENV PORT=5000
ENV DB_PATH=/data/link-kb

EXPOSE 5000

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "300", "--graceful-timeout", "300", "app.server:app"]
