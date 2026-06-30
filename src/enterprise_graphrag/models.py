from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SourceKind(StrEnum):
    TEXT = "text"
    CODE = "code"
    SCHEMA = "schema"
    DOCUMENT = "document"
    IMAGE = "image"
    DIAGRAM = "diagram"


class EntityType(StrEnum):
    SERVICE = "service"
    DATABASE = "database"
    TABLE = "table"
    COLUMN = "column"
    API = "api"
    QUEUE = "queue"
    TOPIC = "topic"
    HOST = "host"
    CLOUD_RESOURCE = "cloud_resource"
    TEAM = "team"
    PERSON = "person"
    INCIDENT = "incident"
    METRIC = "metric"
    TECHNOLOGY = "technology"
    OTHER = "other"


class ExtractedEntity(StrictModel):
    name: str = Field(min_length=1, max_length=300)
    entity_type: EntityType = EntityType.OTHER
    description: str = Field(default="", max_length=1_000)
    aliases: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("aliases")
    @classmethod
    def deduplicate_aliases(cls, aliases: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for alias in aliases:
            normalized = alias.casefold()
            if normalized not in seen:
                seen.add(normalized)
                result.append(alias)
        return result


class ExtractedRelation(StrictModel):
    source: str = Field(min_length=1, max_length=300)
    target: str = Field(min_length=1, max_length=300)
    relation_type: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=1_000)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class ExtractionResult(StrictModel):
    entities: list[ExtractedEntity] = Field(default_factory=list, max_length=100)
    relations: list[ExtractedRelation] = Field(default_factory=list, max_length=200)


class ContextGrade(StrictModel):
    relevant: bool
    score: float = Field(ge=0.0, le=1.0)
    missing_information: list[str] = Field(default_factory=list, max_length=10)
    rationale: str = Field(max_length=1_500)
    rewritten_query: str | None = Field(default=None, max_length=2_000)


class GroundednessGrade(StrictModel):
    grounded: bool
    score: float = Field(ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list, max_length=20)
    corrected_answer: str | None = None


class QueryPlan(StrictModel):
    normalized_query: str = Field(min_length=1, max_length=4_000)
    subqueries: list[str] = Field(min_length=1, max_length=8)
    likely_entities: list[str] = Field(default_factory=list, max_length=30)
    graph_hops: int = Field(default=2, ge=1, le=4)


class RawDocument(StrictModel):
    path: Path
    title: str
    source_kind: SourceKind
    content: str
    content_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    figure_descriptions: list[str] = Field(default_factory=list)


class ParentChunk(StrictModel):
    id: str
    document_id: str
    ordinal: int
    summary: str
    title: str


class Chunk(StrictModel):
    id: str
    document_id: str
    parent_id: str
    ordinal: int
    title: str
    text: str
    context: str
    source_path: str
    source_kind: SourceKind
    content_hash: str
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class EntityRecord(StrictModel):
    id: str
    name: str
    normalized_name: str
    entity_type: EntityType
    description: str
    aliases: list[str]
    chunk_ids: list[str]


class RelationRecord(StrictModel):
    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    description: str
    confidence: float
    evidence_chunk_id: str


class IngestionBundle(StrictModel):
    document_id: str
    title: str
    source_path: str
    source_kind: SourceKind
    content_hash: str
    metadata: dict[str, Any]
    parents: list[ParentChunk]
    chunks: list[Chunk]
    entities: list[EntityRecord]
    relations: list[RelationRecord]


class RetrievedItem(StrictModel):
    chunk_id: str
    document_id: str
    title: str
    text: str
    context: str
    source_path: str
    source_kind: SourceKind
    vector_score: float = 0.0
    keyword_score: float = 0.0
    graph_score: float = 0.0
    fusion_score: float = 0.0
    rerank_score: float = 0.0
    entities: list[str] = Field(default_factory=list)
    retrieval_paths: list[str] = Field(default_factory=list)


class Citation(StrictModel):
    number: int
    chunk_id: str
    source_path: str
    title: str


class AnswerResult(StrictModel):
    query: str
    answer: str
    citations: list[Citation]
    contexts: list[RetrievedItem]
    iterations: int
    relevance_score: float
    groundedness_score: float
    groundedness_checked: bool = True
    accuracy_score: float = Field(default=0.0, ge=0.0, le=1.0)
    mode: str = "full"
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    trace_id: str


class QueryRequest(StrictModel):
    query: str = Field(min_length=3, max_length=12_000)
    conversation_id: str | None = Field(default=None, max_length=200)
    source_paths: list[str] = Field(default_factory=list, max_length=100)
    include_contexts: bool = True

    @field_validator("source_paths")
    @classmethod
    def deduplicate_source_paths(cls, values: list[str]) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []
        for value in values:
            clean = value.strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            output.append(clean)
        return output


class DocumentInfo(StrictModel):
    document_id: str
    title: str
    source_path: str
    source_kind: SourceKind


class IngestRequest(StrictModel):
    path: str = Field(min_length=1)
    recursive: bool = True
    replace_existing: bool = True


class IngestResponse(StrictModel):
    processed: int
    skipped: int
    failed: int
    document_ids: list[str]
    errors: list[str]


class DeleteDocumentRequest(StrictModel):
    source_path: str = Field(min_length=1, max_length=4_096)


class DeleteDocumentResponse(StrictModel):
    deleted: bool
    source_path: str
