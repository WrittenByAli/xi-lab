FROM python:3.11-slim AS base

WORKDIR /app

COPY pyproject.toml .
COPY ml/ ./ml/
COPY backend/ ./backend/
COPY params.yaml .

RUN pip install --no-cache-dir . uvicorn[standard] && \
    rm -rf /root/.cache/pip

COPY data/reference/ ./data/reference/

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
