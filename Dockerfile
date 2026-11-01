# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.13.6 AS uv

FROM python:3.12.16-slim-bookworm AS base
COPY --from=uv /uv /uvx /usr/local/bin/
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/uv-cache \
    PATH="/app/.venv/bin:$PATH" \
    DATA_DIR=/app/data \
    RUNTIME_DIR=/app/runtime \
    HF_HOME=/app/runtime/huggingface \
    HOME=/app/runtime/home \
    XDG_CACHE_HOME=/app/runtime/cache
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/tmp/uv-cache \
    uv sync --locked --no-dev --no-install-project --no-editable
COPY src ./src
COPY app.py ./
COPY .streamlit ./.streamlit
COPY data ./data
RUN --mount=type=cache,target=/tmp/uv-cache \
    uv sync --locked --no-dev --no-editable
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app \
    && mkdir -p /app/runtime/home /app/runtime/cache \
    && chown -R app:app /app/runtime

FROM base AS production
USER app
EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2)"
ENTRYPOINT ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]

FROM base AS test
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/playwright \
    PYTEST_ADDOPTS="-o cache_dir=/tmp/pytest_cache"
COPY tests ./tests
COPY scripts ./scripts
RUN --mount=type=cache,target=/tmp/uv-cache \
    uv sync --locked --no-editable \
    && uv run --no-sync playwright install --with-deps chromium \
    && chmod -R a+rX /opt/playwright \
    && mkdir -p /app/test-results \
    && chown app:app /app/test-results
USER app
CMD ["uv", "run", "--no-sync", "pytest", "-m", "not live"]
