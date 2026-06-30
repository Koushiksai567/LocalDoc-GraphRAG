#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/download_immigration_sources.py" \
  --project-root "$PROJECT_ROOT" --force
rm -f "$PROJECT_ROOT/.cache/streamlit_startup_cache.pkl"
echo
echo "Sources refreshed. Start the application with:"
echo "  ./.venv/bin/python scripts/run_ui.py"
