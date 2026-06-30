from __future__ import annotations

import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any

CACHE_SCHEMA_VERSION = 1


class _RestrictedUnpickler(pickle.Unpickler):
    """Load only primitive pickle data created by this application."""

    def find_class(self, module: str, name: str) -> Any:
        del module, name
        raise pickle.UnpicklingError("Global objects are not allowed in the UI cache")


def load_ui_cache(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            payload = _RestrictedUnpickler(handle).load()
    except (OSError, EOFError, pickle.PickleError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        return None
    fingerprint = payload.get("fingerprint")
    documents = payload.get("documents")
    if not isinstance(fingerprint, str) or not isinstance(documents, list):
        return None
    cleaned_documents: list[dict[str, str]] = []
    for item in documents:
        if not isinstance(item, dict):
            continue
        source_path = item.get("source_path")
        if not isinstance(source_path, str) or not source_path:
            continue
        cleaned_documents.append(
            {
                "document_id": str(item.get("document_id") or ""),
                "title": str(item.get("title") or Path(source_path).stem),
                "source_path": source_path,
                "source_kind": str(item.get("source_kind") or "document"),
            }
        )
    if not cleaned_documents:
        return None
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "documents": cleaned_documents,
        "saved_at": float(payload.get("saved_at") or 0.0),
    }


def save_ui_cache(path: Path, fingerprint: str, documents: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "documents": [
            {
                "document_id": str(item.get("document_id") or ""),
                "title": str(item.get("title") or Path(str(item.get("source_path") or "")).stem),
                "source_path": str(item.get("source_path") or ""),
                "source_kind": str(item.get("source_kind") or "document"),
            }
            for item in documents
            if item.get("source_path")
        ],
        "saved_at": time.time(),
    }
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(path)
    except Exception:
        try:
            Path(temporary_name).unlink(missing_ok=True)
        finally:
            raise
