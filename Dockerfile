FROM python:3.12-slim AS builder

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv "${VIRTUAL_ENV}"

WORKDIR /build
COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip setuptools wheel && pip install .

FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="Fast Local GraphRAG" \
      org.opencontainers.image.description="Fast local GraphRAG using Ollama, LangGraph, Neo4j Community, PyMuPDF, and FastEmbed" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/opt/models/fastembed

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data /app/artifacts /opt/models \
    && chown -R appuser:appuser /app /opt/models
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "enterprise_graphrag.main:app", "--host", "0.0.0.0", "--port", "8000"]
