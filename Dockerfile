# Legacy alias: same as Dockerfile.train.cpu (use that for clarity).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY libs ./libs
COPY projects ./projects
RUN uv sync --no-dev --package train

WORKDIR /app/projects/train
ENTRYPOINT ["uv", "run", "--package", "train", "torchrun"]