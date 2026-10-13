FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e "." && \
    pip install --no-cache-dir uvicorn[standard]

COPY ml/ ./ml/
COPY backend/ ./backend/
COPY data/reference/ ./data/reference/
COPY params.yaml .

ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/models
ENV DATA_DIR=/app/data

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
