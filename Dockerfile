# Stage 1: resolve and install dependencies with uv into a relocatable virtualenv.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# Stage 2: slim runtime, non-root user, atlas prebuilt so the first request is fast.
FROM python:3.13-slim-bookworm AS runtime
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app benchmarks/threshold.json benchmarks/calibration.json /app/benchmarks/
ENV PATH="/app/.venv/bin:$PATH" \
    LEDGERLENS_DATA_DIR=/app/data \
    LEDGERLENS_THRESHOLD_FILE=/app/benchmarks/threshold.json \
    LEDGERLENS_CALIBRATION_FILE=/app/benchmarks/calibration.json \
    PYTHONUNBUFFERED=1
USER app
RUN mkdir -p /app/data && ledgerlens atlas
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
CMD ["ledgerlens", "serve", "--host", "0.0.0.0", "--port", "8000"]
