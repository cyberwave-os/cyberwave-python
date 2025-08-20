FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==1.7.1

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DATABASE_URL=sqlite+aiosqlite:///app/cyberwave.db

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

RUN mkdir -p /app/data && \
    touch /app/cyberwave.db && \
    chown -R appuser:appuser /app/data /app/cyberwave.db

# Use the PORT environment variable provided by Cloud Run
ENV PORT=8080
EXPOSE ${PORT}

CMD ["sh", "-c", "alembic upgrade head && python -m uvicorn src.main:app --host 0.0.0.0 --port ${PORT}"] 