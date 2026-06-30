#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
INGEST_PATH="${INGEST_PATH:-sample_data}"
API_HEADER=()
if [[ -n "${API_KEY:-}" ]]; then
  API_HEADER=(-H "X-API-Key: ${API_KEY}")
fi

printf '\n== Health before ingestion ==\n'
curl -fsS "${BASE_URL}/health" | python -m json.tool

printf '\n== Ingest sample data ==\n'
python - "${BASE_URL}" "${INGEST_PATH}" "${API_KEY:-}" <<'PY'
import json
import sys
import urllib.request

base_url, path, api_key = sys.argv[1:]
payload = json.dumps({"path": path, "recursive": True, "replace_existing": True}).encode()
request = urllib.request.Request(
    f"{base_url}/v1/ingest",
    data=payload,
    headers={"Content-Type": "application/json", **({"X-API-Key": api_key} if api_key else {})},
    method="POST",
)
with urllib.request.urlopen(request, timeout=3600) as response:
    print(json.dumps(json.load(response), indent=2))
PY

printf '\n== Health after ingestion ==\n'
curl -fsS "${BASE_URL}/health" | python -m json.tool

printf '\n== Ask a multi-hop question ==\n'
python - "${BASE_URL}" "${API_KEY:-}" <<'PY'
import json
import sys
import urllib.request

base_url, api_key = sys.argv[1:]
payload = json.dumps({
    "query": "Why can checkout return HTTP 503 even when the inventory reservation row exists, and how should an operator verify and mitigate it?",
    "include_contexts": True,
}).encode()
request = urllib.request.Request(
    f"{base_url}/v1/query",
    data=payload,
    headers={"Content-Type": "application/json", **({"X-API-Key": api_key} if api_key else {})},
    method="POST",
)
with urllib.request.urlopen(request, timeout=3600) as response:
    result = json.load(response)
print(json.dumps(result, indent=2))
if not result.get("citations"):
    raise SystemExit("Verification failed: no citations returned")
if not result.get("contexts"):
    raise SystemExit("Verification failed: no contexts returned")
PY

printf '\nVerification completed successfully.\n'
