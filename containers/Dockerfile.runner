FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/local/bin/pip3.12 /usr/local/bin/pip \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.5.1+cu121"

WORKDIR /app
COPY pyproject.toml ./
COPY docs ./docs
COPY src ./src
RUN python -m pip install --no-cache-dir \
    "numpy>=2,<3" \
    "pillow>=10,<12" \
    "pydantic>=2,<3" \
    "PyYAML>=6,<7" \
    "typer>=0.12,<1"
RUN python -m pip install --no-cache-dir --no-deps .

ENTRYPOINT ["python", "-m", "ml_autoresearch.container_smoke"]
