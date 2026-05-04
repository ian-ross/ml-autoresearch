FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml ./
COPY docs ./docs
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

ENTRYPOINT ["python", "-m", "ml_autoresearch.container_smoke"]
