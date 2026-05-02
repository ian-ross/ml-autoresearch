FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml ./
COPY docs ./docs
COPY src ./src
RUN pip install --no-cache-dir .

ENTRYPOINT ["python", "-m", "ml_autoresearch.container_smoke"]
