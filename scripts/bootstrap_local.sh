#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

for command in python docker ollama; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Missing required command: ${command}" >&2
    exit 1
  fi
done

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

GENERATION_MODEL="$(grep -E '^GENERATION_MODEL=' .env | cut -d= -f2- || true)"
REASONING_MODEL="$(grep -E '^REASONING_MODEL=' .env | cut -d= -f2- || true)"
VISION_MODEL="$(grep -E '^VISION_MODEL=' .env | cut -d= -f2- || true)"
ENABLE_VISION="$(grep -E '^ENABLE_VISION=' .env | cut -d= -f2- || true)"

GENERATION_MODEL="${GENERATION_MODEL:-qwen2.5:7b-instruct}"
REASONING_MODEL="${REASONING_MODEL:-${GENERATION_MODEL}}"
VISION_MODEL="${VISION_MODEL:-qwen2.5vl:3b}"
ENABLE_VISION="${ENABLE_VISION:-true}"

ollama pull "${GENERATION_MODEL}"
if [[ "${REASONING_MODEL}" != "${GENERATION_MODEL}" ]]; then
  ollama pull "${REASONING_MODEL}"
fi
if [[ "${ENABLE_VISION,,}" == "true" ]]; then
  ollama pull "${VISION_MODEL}"
fi

docker compose up -d neo4j

if [[ ! -d .venv ]]; then
  python -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo
echo "Bootstrap complete. Start the API with:"
echo "  source .venv/bin/activate"
echo "  uvicorn enterprise_graphrag.main:app --host 0.0.0.0 --port 8000"
