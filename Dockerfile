# Cadence — production image (API + UI, CPU tiers)
# Build:  docker build -t cadence .
# Run:    see docker-compose.yml

FROM python:3.11-slim AS base

# uv: fast, lockfile-exact installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    CADENCE_UPLOAD_DIR=/data/uploads \
    CADENCE_MAX_UPLOAD_MB=200

WORKDIR /app

# dependency layers first (cached until the lockfile changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-default-groups --group ui

# app source
COPY cadence/ cadence/
RUN uv sync --frozen --no-default-groups --group ui

# uploads volume + non-root runtime user
RUN mkdir -p /data/uploads && useradd -m -u 10001 cadence && chown -R cadence /data /app
USER cadence

EXPOSE 8000 8501
