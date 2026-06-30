from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Fast Local GraphRAG"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: SecretStr | None = None
    request_timeout_seconds: float = Field(default=180.0, gt=0.0, le=3_600.0)
    max_query_chars: int = Field(default=12_000, ge=3, le=100_000)

    # Fast mode removes all LLM calls from ingestion and uses exactly one LLM call per query.
    fast_mode: bool = True
    preload_models: bool = True
    indexing_version: str = "fast-v3.8-on-demand"

    # Local Ollama service. No cloud key is required.
    ollama_base_url: str = "http://localhost:11434"
    generation_model: str = "qwen3:1.7b"
    reasoning_model: str = "qwen3:1.7b"
    vision_model: str = "qwen2.5vl:3b"
    ollama_num_ctx: int = Field(default=4_096, ge=2_048, le=262_144)
    ollama_keep_alive: str = "60m"
    max_output_tokens: int = Field(default=450, gt=0, le=32_000)
    llm_concurrency: int = Field(default=1, gt=0, le=32)
    enable_vision: bool = False
    disable_model_thinking: bool = True

    # FastEmbed uses quantized ONNX models and does not require PyTorch.
    embedding_backend: Literal["fastembed", "sentence_transformers"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = Field(default=384, gt=0, le=100_000)
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=128, gt=0, le=1_024)
    embedding_threads: int | None = Field(default=None, gt=0, le=128)
    normalize_embeddings: bool = True

    # Neo4j Community Edition.
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("change-me-now")
    neo4j_database: str = "neo4j"
    vector_index_name: str = "chunk_embedding_idx"
    fulltext_index_name: str = "chunk_fulltext_idx"

    # Fast parsing and deterministic ingestion.
    pdf_parser: Literal["pymupdf", "docling"] = "pymupdf"
    pdf_min_text_chars: int = Field(default=80, ge=0, le=100_000)
    enable_docling_fallback: bool = False
    enable_parent_llm_summaries: bool = False
    enable_llm_graph_extraction: bool = False
    chunk_size_tokens: int = Field(default=450, gt=0, le=16_000)
    chunk_overlap_tokens: int = Field(default=40, ge=0, le=8_000)
    parent_children: int = Field(default=4, gt=0, le=100)
    ingestion_concurrency: int = Field(default=2, gt=0, le=32)
    figure_caption_concurrency: int = Field(default=1, gt=0, le=16)
    supported_extensions: str = (
        ".md,.mmd,.txt,.csv,.sql,.json,.yaml,.yml,.py,.java,.js,.ts,.html,"
        ".pdf,.docx,.pptx,.png,.jpg,.jpeg,.webp"
    )
    allowed_ingestion_roots: str = "data,sample_data"
    max_file_size_mb: int = Field(default=200, gt=0, le=10_000)

    # Fast hybrid retrieval: no HyDE or LLM query planner in fast mode.
    enable_hyde: bool = False
    enable_neural_reranker: bool = True
    query_expansion_limit: int = Field(default=12, ge=1, le=24)
    vector_top_k: int = Field(default=8, gt=0, le=1_000)
    keyword_top_k: int = Field(default=10, gt=0, le=1_000)
    graph_seed_k: int = Field(default=4, gt=0, le=1_000)
    graph_expand_k: int = Field(default=6, gt=0, le=5_000)
    rerank_top_k: int = Field(default=6, gt=0, le=500)
    rerank_candidate_k: int = Field(default=24, gt=0, le=1_000)
    rrf_k: int = Field(default=60, gt=0, le=10_000)
    cross_encoder_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    cross_encoder_backend: Literal["fastembed", "torch", "onnx", "openvino"] = "fastembed"
    cross_encoder_device: str = "cpu"
    answer_context_chars: int = Field(default=10_000, ge=2_000, le=100_000)
    source_filter_candidate_k: int = Field(default=500, gt=0, le=10_000)

    # Full Self-RAG remains available by setting FAST_MODE=false.
    max_agent_iterations: int = Field(default=2, ge=1, le=10)
    min_context_relevance: float = Field(default=0.55, ge=0.0, le=1.0)
    min_groundedness: float = Field(default=0.82, ge=0.0, le=1.0)

    data_dir: Path = Field(default=Path("data"))
    artifacts_dir: Path = Field(default=Path("artifacts"))
    model_cache_dir: Path = Field(default=Path(".model_cache"))
    eval_dataset_path: Path = Field(default=Path("eval/evalset.jsonl"))

    @field_validator("api_key", mode="before")
    @classmethod
    def normalize_blank_api_key(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, SecretStr):
            return value if value.get_secret_value().strip() else None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("ollama_base_url")
    @classmethod
    def normalize_ollama_url(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def validate_overlap(cls, value: int, info):  # type: ignore[no-untyped-def]
        chunk_size = info.data.get("chunk_size_tokens")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        return value

    @property
    def extension_set(self) -> set[str]:
        return {item.strip().lower() for item in self.supported_extensions.split(",") if item.strip()}

    @property
    def ingestion_roots(self) -> list[Path]:
        return [
            Path(item.strip()).expanduser().resolve()
            for item in self.allowed_ingestion_roots.split(",")
            if item.strip()
        ]

    @property
    def required_ollama_models(self) -> set[str]:
        models = {self.generation_model, self.reasoning_model}
        if self.enable_vision:
            models.add(self.vision_model)
        return models


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    return settings
