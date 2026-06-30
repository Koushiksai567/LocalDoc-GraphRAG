from __future__ import annotations

import shutil
from pathlib import Path

FAST_VALUES = {
    "FAST_MODE": "true",
    "PRELOAD_MODELS": "true",
    "REQUEST_TIMEOUT_SECONDS": "180",
    "INDEXING_VERSION": "fast-v3.8-on-demand",
    "GENERATION_MODEL": "qwen3:1.7b",
    "REASONING_MODEL": "qwen3:1.7b",
    "OLLAMA_NUM_CTX": "4096",
    "OLLAMA_KEEP_ALIVE": "60m",
    "MAX_OUTPUT_TOKENS": "450",
    "LLM_CONCURRENCY": "1",
    "ENABLE_VISION": "false",
    "DISABLE_MODEL_THINKING": "true",
    "EMBEDDING_BACKEND": "fastembed",
    "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
    "EMBEDDING_DIMENSIONS": "384",
    "EMBEDDING_BATCH_SIZE": "128",
    "CROSS_ENCODER_MODEL": "Xenova/ms-marco-MiniLM-L-6-v2",
    "CROSS_ENCODER_BACKEND": "fastembed",
    "ENABLE_NEURAL_RERANKER": "true",
    "PDF_PARSER": "pymupdf",
    "PDF_MIN_TEXT_CHARS": "80",
    "ENABLE_DOCLING_FALLBACK": "false",
    "ENABLE_PARENT_LLM_SUMMARIES": "false",
    "ENABLE_LLM_GRAPH_EXTRACTION": "false",
    "CHUNK_SIZE_TOKENS": "450",
    "CHUNK_OVERLAP_TOKENS": "40",
    "PARENT_CHILDREN": "4",
    "INGESTION_CONCURRENCY": "2",
    "QUERY_EXPANSION_LIMIT": "12",
    "VECTOR_TOP_K": "8",
    "KEYWORD_TOP_K": "10",
    "GRAPH_SEED_K": "4",
    "GRAPH_EXPAND_K": "6",
    "RERANK_TOP_K": "6",
    "RERANK_CANDIDATE_K": "24",
    "ANSWER_CONTEXT_CHARS": "10000",
}


def main() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        example = Path(".env.example")
        if not example.exists():
            raise SystemExit("Run this script from the project root; .env.example was not found.")
        shutil.copy2(example, env_path)
        print("Created .env from .env.example")
    else:
        backup = Path(".env.before-fast-v3.8")
        shutil.copy2(env_path, backup)
        print(f"Backed up the current environment file to {backup}")

    lines = env_path.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    updated: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            updated.append(line)
            continue
        key, _value = line.split("=", 1)
        normalized = key.strip()
        if normalized in FAST_VALUES:
            updated.append(f"{normalized}={FAST_VALUES[normalized]}")
            seen.add(normalized)
        else:
            updated.append(line)

    missing = [key for key in FAST_VALUES if key not in seen]
    if missing:
        updated.extend(["", "# Fast v3.8 settings added automatically"])
        updated.extend(f"{key}={FAST_VALUES[key]}" for key in missing)

    env_path.write_text("\n".join(updated).rstrip() + "\n", encoding="utf-8")
    print("Fast mode enabled. Your Neo4j password and other unrelated settings were preserved.")


if __name__ == "__main__":
    main()
