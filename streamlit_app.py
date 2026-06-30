from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import httpx  # noqa: E402
import streamlit as st  # noqa: E402

from enterprise_graphrag.ui_cache import save_ui_cache  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
CACHE_PATH = PROJECT_ROOT / ".cache" / "streamlit_startup_cache.pkl"
API_URL = os.getenv("GRAPHRAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("GRAPHRAG_UI_TIMEOUT_SECONDS", "3600"))
BACKEND_STARTUP_TIMEOUT = float(os.getenv("GRAPHRAG_UI_STARTUP_TIMEOUT_SECONDS", "180"))
SUPPORTED_EXTENSIONS = {
    ".md",
    ".mmd",
    ".txt",
    ".csv",
    ".sql",
    ".json",
    ".yaml",
    ".yml",
    ".py",
    ".java",
    ".js",
    ".ts",
    ".html",
    ".pdf",
    ".docx",
    ".pptx",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


def _read_env_value(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value.strip().strip('"').strip("'")
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return None
    for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        if key.strip().casefold() == name.casefold():
            parsed = raw_value.strip().strip('"').strip("'")
            return parsed or None
    return None


def _headers() -> dict[str, str]:
    api_key = _read_env_value("GRAPHRAG_API_KEY") or _read_env_value("API_KEY")
    return {"x-api-key": api_key} if api_key else {}


def _data_fingerprint() -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for path in sorted(item for item in DATA_DIR.rglob("*") if item.is_file()):
        stat = path.stat()
        digest.update(str(path.relative_to(DATA_DIR)).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _local_documents() -> list[dict[str, Any]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    documents: list[dict[str, Any]] = []
    for path in sorted(DATA_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        documents.append(
            {
                "document_id": "",
                "title": path.stem,
                "source_path": str(path.resolve()),
                "source_kind": "document",
            }
        )
    return documents


def _paths(documents: list[dict[str, Any]]) -> set[str]:
    return {
        str(Path(str(item["source_path"])).expanduser().resolve())
        for item in documents
        if item.get("source_path")
    }


def _friendly_connection_error(exc: Exception) -> RuntimeError:
    if isinstance(exc, httpx.ReadTimeout):
        return RuntimeError(
            "The request took too long. Try one small document first or ask a narrower question. "
            "The backend remains safe to restart."
        )
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return RuntimeError(
            "The GraphRAG backend is not reachable on port 8000. Restart with "
            "`./.venv/bin/python scripts/run_ui.py`."
        )
    if isinstance(exc, httpx.RemoteProtocolError):
        return RuntimeError(
            "The backend disconnected while processing the request. It may have run out of memory or been stopped. "
            "Restart the app and test with one small document."
        )
    return RuntimeError(str(exc))


def _request(method: str, path: str, *, timeout: float, json: dict[str, Any] | None = None) -> httpx.Response:
    try:
        return httpx.request(
            method,
            f"{API_URL}{path}",
            headers=_headers(),
            json=json,
            timeout=httpx.Timeout(connect=10.0, read=timeout, write=30.0, pool=30.0),
        )
    except httpx.HTTPError as exc:
        raise _friendly_connection_error(exc) from exc


def _wait_for_backend() -> dict[str, Any]:
    deadline = time.monotonic() + BACKEND_STARTUP_TIMEOUT
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = _request("GET", "/health", timeout=5.0)
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            last_error = exc
            time.sleep(0.75)
    raise RuntimeError(
        "The backend did not become ready. Confirm that Docker, Neo4j, and Ollama are running, "
        "then restart with `./.venv/bin/python scripts/run_ui.py`."
    ) from last_error


def _error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or f"HTTP {response.status_code}"
    if isinstance(payload, dict):
        return str(payload.get("detail") or payload.get("error") or payload)
    return str(payload)


def _list_documents() -> list[dict[str, Any]]:
    response = _request("GET", "/v1/documents", timeout=30.0)
    if response.is_error:
        raise RuntimeError(f"Could not list indexed documents: {_error_detail(response)}")
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError("The backend returned an invalid document list.")
    return [item for item in payload if isinstance(item, dict) and item.get("source_path")]


def _delete_indexed_document(source_path: str) -> None:
    response = _request(
        "DELETE",
        "/v1/documents",
        timeout=60.0,
        json={"source_path": source_path},
    )
    if response.is_error:
        raise RuntimeError(f"Could not remove stale document {Path(source_path).name}: {_error_detail(response)}")


def _remove_stale_documents(
    local_documents: list[dict[str, Any]],
    backend_documents: list[dict[str, Any]],
) -> int:
    local_paths = _paths(local_documents)
    data_root = DATA_DIR.resolve()
    stale: list[str] = []
    for item in backend_documents:
        raw_path = item.get("source_path")
        if not raw_path:
            continue
        path = Path(str(raw_path)).expanduser().resolve()
        if path.is_relative_to(data_root) and str(path) not in local_paths:
            stale.append(str(path))
    for source_path in stale:
        _delete_indexed_document(source_path)
    return len(stale)


def _ingest_document(source_path: str) -> dict[str, Any]:
    response = _request(
        "POST",
        "/v1/ingest",
        timeout=REQUEST_TIMEOUT,
        json={
            "path": source_path,
            "recursive": False,
            "replace_existing": True,
        },
    )
    if response.is_error:
        raise RuntimeError(f"Could not index {Path(source_path).name}: {_error_detail(response)}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"The backend returned an invalid ingestion response for {Path(source_path).name}.")
    if int(payload.get("failed", 0)) > 0:
        errors = payload.get("errors", [])
        detail = "; ".join(str(item) for item in errors[:3]) if isinstance(errors, list) else str(errors)
        raise RuntimeError(f"Could not index {Path(source_path).name}: {detail}")
    return payload


def _prepare_selected_documents(
    local_documents: list[dict[str, Any]],
    selected_source_paths: list[str],
) -> tuple[list[dict[str, Any]], int, int, int]:
    _wait_for_backend()
    backend_documents = _list_documents()
    deleted = _remove_stale_documents(local_documents, backend_documents)

    processed = 0
    skipped = 0
    for source_path in selected_source_paths:
        payload = _ingest_document(str(Path(source_path).expanduser().resolve()))
        processed += int(payload.get("processed", 0))
        skipped += int(payload.get("skipped", 0))

    backend_documents = _list_documents()
    save_ui_cache(CACHE_PATH, _data_fingerprint(), backend_documents)
    return backend_documents, processed, skipped, deleted


def _ask_backend(query: str, source_paths: list[str]) -> dict[str, Any]:
    response = _request(
        "POST",
        "/v1/query",
        timeout=REQUEST_TIMEOUT,
        json={
            "query": query,
            "source_paths": source_paths,
            "include_contexts": False,
        },
    )
    if response.is_error:
        detail = _error_detail(response)
        if response.status_code == 503 and "Ollama" in detail:
            raise RuntimeError(
                f"The local Ollama model was unavailable: {detail} "
                "Confirm `ollama list` works and retry with one small document."
            )
        raise RuntimeError(f"Query failed: {detail}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("The backend returned an invalid answer response.")
    return payload


st.set_page_config(page_title="Local GraphRAG", page_icon="🔎", layout="centered")
st.markdown(
    """
    <style>
      .block-container {max-width: 900px; padding-top: 2.5rem;}
      [data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); padding: 1rem; border-radius: .75rem;}
      .stButton > button {width: 100%;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Local GraphRAG Assistant")
st.caption("Documents are indexed only when selected. Unchanged files are skipped automatically.")

local_documents = _local_documents()
if not local_documents:
    st.warning("Add one or more supported files to the `data` folder, then refresh this page.")
    st.stop()

labels = {
    str(item["source_path"]): f"{item.get('title') or Path(str(item['source_path'])).stem} — {Path(str(item['source_path'])).name}"
    for item in local_documents
}
options = list(labels)
previous = [path for path in st.session_state.get("selected_source_paths", []) if path in labels]
default_selection = previous or (options if len(options) == 1 else [])
selected_source_paths = st.multiselect(
    "Select document(s)",
    options=options,
    default=default_selection,
    format_func=lambda value: labels[value],
    help="Only selected documents are indexed and searched, which keeps the app faster.",
)
st.session_state.selected_source_paths = selected_source_paths
st.caption(f"{len(local_documents)} file(s) found in `data/`. {len(selected_source_paths)} selected.")

with st.form("query_form", clear_on_submit=False):
    query = st.text_area(
        "Enter your question",
        height=120,
        placeholder="Ask a question about the selected document(s)...",
    )
    submitted = st.form_submit_button("Ask")

if submitted:
    clean_query = query.strip()
    if not selected_source_paths:
        st.warning("Select at least one document before asking a question.")
    elif len(clean_query) < 3:
        st.warning("Please enter a complete question.")
    else:
        try:
            with st.status("Preparing selected documents...", expanded=True) as status:
                status.write("Checking the backend and removing stale indexed files...")
                backend_documents, processed, skipped, deleted = _prepare_selected_documents(
                    local_documents,
                    selected_source_paths,
                )
                status.write(
                    f"Indexing complete: {processed} new/changed, {skipped} unchanged, {deleted} stale removed."
                )
                backend_paths = _paths(backend_documents)
                filtered_selection = [
                    str(Path(path).expanduser().resolve())
                    for path in selected_source_paths
                    if str(Path(path).expanduser().resolve()) in backend_paths
                ]
                if not filtered_selection:
                    raise RuntimeError(
                        "None of the selected documents could be found in the indexed knowledge base."
                    )
                status.update(label="Generating answer...", state="running")
                result = _ask_backend(clean_query, filtered_selection)
                status.update(label="Answer ready", state="complete", expanded=False)
            st.session_state.last_result = result
        except Exception as exc:
            st.error(str(exc))

result = st.session_state.get("last_result")
if isinstance(result, dict):
    answer = str(result.get("answer", "No answer was returned."))
    raw_accuracy = result.get("accuracy_score", result.get("relevance_score", 0.0))
    try:
        accuracy = max(0.0, min(1.0, float(raw_accuracy or 0.0)))
    except (TypeError, ValueError):
        accuracy = 0.0

    st.subheader("Response")
    st.markdown(answer)
    st.metric("Estimated answer confidence", f"{accuracy * 100:.0f}%")
    st.progress(accuracy)
