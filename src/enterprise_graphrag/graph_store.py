from __future__ import annotations

import json
import logging
import re
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncManagedTransaction, RoutingControl
from neo4j.exceptions import DriverError, Neo4jError

from .config import Settings
from .models import DocumentInfo, IngestionBundle, RetrievedItem, SourceKind

logger = logging.getLogger(__name__)
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NEO4J_ERRORS = (Neo4jError, DriverError)

_DELETE_DOCUMENT_QUERY = """
MATCH (d:Document {source_path: $source_path})
OPTIONAL MATCH (d)-[:HAS_PARENT]->(p:ParentChunk)-[:HAS_CHUNK]->(c:Chunk)
OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
WITH d, collect(DISTINCT p) AS parents, collect(DISTINCT c) AS chunks,
     collect(DISTINCT c.id) AS chunk_ids, collect(DISTINCT e.id) AS entity_ids
CALL (chunk_ids) {
    MATCH ()-[r:RELATED_TO]->()
    WHERE r.evidence_chunk_id IN chunk_ids
    DELETE r
}
CALL (chunks) {
    UNWIND chunks AS chunk
    DETACH DELETE chunk
}
CALL (parents) {
    UNWIND parents AS parent
    DETACH DELETE parent
}
DETACH DELETE d
WITH entity_ids
UNWIND entity_ids AS entity_id
MATCH (e:Entity {id: entity_id})
WHERE NOT (e)<-[:MENTIONS]-(:Chunk)
DETACH DELETE e
"""

_UPSERT_BUNDLE_QUERY = """
MERGE (d:Document {id: $document.id})
SET d.title = $document.title,
    d.source_path = $document.source_path,
    d.source_kind = $document.source_kind,
    d.content_hash = $document.content_hash,
    d.metadata_json = $document.metadata_json,
    d.updated_at = datetime()
WITH d
CALL (d) {
    UNWIND $parents AS row
    MERGE (p:ParentChunk {id: row.id})
    SET p.document_id = row.document_id, p.ordinal = row.ordinal,
        p.summary = row.summary, p.title = row.title
    MERGE (d)-[:HAS_PARENT]->(p)
}
CALL (d) {
    UNWIND $chunks AS row
    MATCH (p:ParentChunk {id: row.parent_id})
    MERGE (c:Chunk {id: row.id})
    SET c.document_id = row.document_id,
        c.parent_id = row.parent_id,
        c.ordinal = row.ordinal,
        c.title = row.title,
        c.text = row.text,
        c.context = row.context,
        c.source_path = row.source_path,
        c.source_kind = row.source_kind,
        c.content_hash = row.content_hash,
        c.embedding = row.embedding,
        c.metadata_json = row.metadata_json,
        c.updated_at = datetime()
    MERGE (p)-[:HAS_CHUNK]->(c)
}
CALL () {
    UNWIND $entities AS row
    MERGE (e:Entity {id: row.id})
    ON CREATE SET e.created_at = datetime()
    SET e.name = row.name,
        e.normalized_name = row.normalized_name,
        e.entity_type = row.entity_type,
        e.description = CASE WHEN size(row.description) > size(coalesce(e.description, ''))
                             THEN row.description ELSE e.description END,
        e.aliases = reduce(existing = coalesce(e.aliases, []), alias IN row.aliases |
                           CASE WHEN alias IN existing THEN existing ELSE existing + alias END),
        e.updated_at = datetime()
    WITH e, row
    UNWIND row.chunk_ids AS chunk_id
    MATCH (c:Chunk {id: chunk_id})
    MERGE (c)-[:MENTIONS]->(e)
}
CALL () {
    UNWIND $relations AS row
    MATCH (source:Entity {id: row.source_entity_id})
    MATCH (target:Entity {id: row.target_entity_id})
    MATCH (evidence:Chunk {id: row.evidence_chunk_id})
    MERGE (source)-[r:RELATED_TO {id: row.id}]->(target)
    SET r.relation_type = row.relation_type,
        r.description = row.description,
        r.confidence = row.confidence,
        r.evidence_chunk_id = row.evidence_chunk_id,
        r.updated_at = datetime()
    MERGE (evidence)-[:EVIDENCE_FOR]->(source)
    MERGE (evidence)-[:EVIDENCE_FOR]->(target)
}
RETURN d.id AS document_id
"""


class GraphStoreError(RuntimeError):
    pass


def _safe_identifier(value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Unsafe Neo4j identifier: {value!r}")
    return value


class Neo4jGraphStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.driver: AsyncDriver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
            max_connection_pool_size=50,
            connection_acquisition_timeout=30,
        )

    async def close(self) -> None:
        await self.driver.close()

    async def verify_connectivity(self) -> None:
        try:
            await self.driver.verify_connectivity()
        except _NEO4J_ERRORS as exc:
            raise GraphStoreError(f"Neo4j connectivity failed: {exc}") from exc

    async def ensure_schema(self) -> None:
        vector_index = _safe_identifier(self.settings.vector_index_name)
        fulltext_index = _safe_identifier(self.settings.fulltext_index_name)
        dimensions = self.settings.embedding_dimensions
        statements = [
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            (
                "CREATE CONSTRAINT document_source_path_unique IF NOT EXISTS "
                "FOR (d:Document) REQUIRE d.source_path IS UNIQUE"
            ),
            "CREATE CONSTRAINT parent_id_unique IF NOT EXISTS FOR (p:ParentChunk) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE INDEX chunk_document_idx IF NOT EXISTS FOR (c:Chunk) ON (c.document_id)",
            "CREATE INDEX chunk_source_path_idx IF NOT EXISTS FOR (c:Chunk) ON (c.source_path)",
            "CREATE INDEX entity_normalized_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.normalized_name)",
            (
                f"CREATE VECTOR INDEX {vector_index} IF NOT EXISTS FOR (c:Chunk) ON c.embedding "
                f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dimensions}, "
                "`vector.similarity_function`: 'cosine'}}"
            ),
            (
                f"CREATE FULLTEXT INDEX {fulltext_index} IF NOT EXISTS "
                "FOR (c:Chunk) ON EACH [c.title, c.text, c.context]"
            ),
        ]
        for statement in statements:
            try:
                await self.driver.execute_query(
                    statement,
                    database_=self.settings.neo4j_database,
                    routing_=RoutingControl.WRITE,
                )
            except _NEO4J_ERRORS as exc:
                raise GraphStoreError(f"Schema creation failed for {statement!r}: {exc}") from exc

    async def wait_for_indexes(self, timeout_seconds: int = 120) -> None:
        try:
            await self.driver.execute_query(
                "CALL db.awaitIndexes($timeout)",
                timeout=timeout_seconds,
                database_=self.settings.neo4j_database,
                routing_=RoutingControl.READ,
            )
        except _NEO4J_ERRORS as exc:
            raise GraphStoreError(f"Neo4j indexes did not become ready: {exc}") from exc

    async def document_hash(self, source_path: str) -> str | None:
        try:
            records, _, _ = await self.driver.execute_query(
                "MATCH (d:Document {source_path: $source_path}) RETURN d.content_hash AS content_hash LIMIT 1",
                source_path=source_path,
                database_=self.settings.neo4j_database,
                routing_=RoutingControl.READ,
            )
        except _NEO4J_ERRORS as exc:
            raise GraphStoreError(f"Failed to read document hash for {source_path}: {exc}") from exc
        return str(records[0]["content_hash"]) if records else None

    async def delete_document(self, source_path: str) -> None:
        try:
            await self.driver.execute_query(
                _DELETE_DOCUMENT_QUERY,
                source_path=source_path,
                database_=self.settings.neo4j_database,
                routing_=RoutingControl.WRITE,
            )
        except _NEO4J_ERRORS as exc:
            raise GraphStoreError(f"Failed to delete existing document {source_path}: {exc}") from exc

    async def upsert_bundle(self, bundle: IngestionBundle) -> None:
        parameters = _bundle_parameters(bundle)
        try:
            await self.driver.execute_query(
                _UPSERT_BUNDLE_QUERY,
                **parameters,
                database_=self.settings.neo4j_database,
                routing_=RoutingControl.WRITE,
            )
        except _NEO4J_ERRORS as exc:
            logger.exception("Failed to upsert document %s", bundle.source_path)
            raise GraphStoreError(f"Bundle upsert failed: {exc}") from exc

    async def replace_bundle(self, bundle: IngestionBundle) -> None:
        """Atomically remove the old source graph and insert its rebuilt bundle."""
        parameters = _bundle_parameters(bundle)

        async def replace_in_transaction(tx: AsyncManagedTransaction) -> None:
            delete_result = await tx.run(_DELETE_DOCUMENT_QUERY, source_path=bundle.source_path)
            await delete_result.consume()
            upsert_result = await tx.run(_UPSERT_BUNDLE_QUERY, **parameters)
            await upsert_result.consume()

        try:
            async with self.driver.session(database=self.settings.neo4j_database) as session:
                await session.execute_write(replace_in_transaction)
        except _NEO4J_ERRORS as exc:
            logger.exception("Failed to atomically replace document %s", bundle.source_path)
            raise GraphStoreError(f"Atomic bundle replacement failed: {exc}") from exc

    async def vector_search(
        self,
        embedding: list[float],
        limit: int,
        source_paths: list[str] | None = None,
    ) -> list[RetrievedItem]:
        filters = _normalize_source_paths(source_paths)
        candidate_limit = _candidate_limit(limit, filters, self.settings.source_filter_candidate_k)
        query = """
        CALL db.index.vector.queryNodes($index_name, $candidate_limit, $embedding)
        YIELD node, score
        WHERE size($source_paths) = 0 OR node.source_path IN $source_paths
        OPTIONAL MATCH (node)-[:MENTIONS]->(e:Entity)
        WITH node, score, collect(DISTINCT e.name)[0..20] AS entities
        RETURN node.id AS chunk_id, node.document_id AS document_id,
               node.title AS title, node.text AS text, node.context AS context,
               node.source_path AS source_path, node.source_kind AS source_kind,
               score AS vector_score, entities
        ORDER BY vector_score DESC
        LIMIT $limit
        """
        try:
            records, _, _ = await self.driver.execute_query(
                query,
                index_name=self.settings.vector_index_name,
                candidate_limit=candidate_limit,
                limit=limit,
                embedding=embedding,
                source_paths=filters,
                database_=self.settings.neo4j_database,
                routing_=RoutingControl.READ,
            )
        except _NEO4J_ERRORS as exc:
            raise GraphStoreError(f"Vector search failed: {exc}") from exc
        return [_record_to_item(record, path="vector") for record in records]

    async def keyword_search(
        self,
        query_text: str,
        limit: int,
        source_paths: list[str] | None = None,
    ) -> list[RetrievedItem]:
        sanitized = _lucene_escape(query_text)
        filters = _normalize_source_paths(source_paths)
        candidate_limit = _candidate_limit(limit, filters, self.settings.source_filter_candidate_k)
        query = """
        CALL db.index.fulltext.queryNodes($index_name, $query_text, {limit: $candidate_limit})
        YIELD node, score
        WHERE size($source_paths) = 0 OR node.source_path IN $source_paths
        OPTIONAL MATCH (node)-[:MENTIONS]->(e:Entity)
        WITH node, score, collect(DISTINCT e.name)[0..20] AS entities
        RETURN node.id AS chunk_id, node.document_id AS document_id,
               node.title AS title, node.text AS text, node.context AS context,
               node.source_path AS source_path, node.source_kind AS source_kind,
               score AS keyword_score, entities
        ORDER BY keyword_score DESC
        LIMIT $limit
        """
        try:
            records, _, _ = await self.driver.execute_query(
                query,
                index_name=self.settings.fulltext_index_name,
                query_text=sanitized,
                candidate_limit=candidate_limit,
                limit=limit,
                source_paths=filters,
                database_=self.settings.neo4j_database,
                routing_=RoutingControl.READ,
            )
        except _NEO4J_ERRORS as exc:
            raise GraphStoreError(f"Full-text search failed: {exc}") from exc
        return [_record_to_item(record, path="keyword") for record in records]

    async def exact_phrase_search(
        self,
        phrases: list[str],
        limit: int,
        source_paths: list[str] | None = None,
    ) -> list[RetrievedItem]:
        """Search the full-text index for one or more exact policy phrases."""
        normalized = [phrase.strip() for phrase in phrases if phrase.strip()]
        normalized = list(dict.fromkeys(normalized))
        if not normalized:
            return []
        filters = _normalize_source_paths(source_paths)
        candidate_limit = _candidate_limit(limit, filters, self.settings.source_filter_candidate_k)
        query_text = " OR ".join(_lucene_phrase(phrase) for phrase in normalized)
        query = """
        CALL db.index.fulltext.queryNodes($index_name, $query_text, {limit: $candidate_limit})
        YIELD node, score
        WHERE size($source_paths) = 0 OR node.source_path IN $source_paths
        OPTIONAL MATCH (node)-[:MENTIONS]->(e:Entity)
        WITH node, score, collect(DISTINCT e.name)[0..20] AS entities
        RETURN node.id AS chunk_id, node.document_id AS document_id,
               node.title AS title, node.text AS text, node.context AS context,
               node.source_path AS source_path, node.source_kind AS source_kind,
               score AS keyword_score, entities
        ORDER BY keyword_score DESC
        LIMIT $limit
        """
        try:
            records, _, _ = await self.driver.execute_query(
                query,
                index_name=self.settings.fulltext_index_name,
                query_text=query_text,
                candidate_limit=candidate_limit,
                limit=limit,
                source_paths=filters,
                database_=self.settings.neo4j_database,
                routing_=RoutingControl.READ,
            )
        except _NEO4J_ERRORS as exc:
            raise GraphStoreError(f"Exact phrase search failed: {exc}") from exc
        return [_record_to_item(record, path="exact_phrase") for record in records]

    async def entity_search(
        self,
        entity_names: list[str],
        limit: int,
        source_paths: list[str] | None = None,
    ) -> list[RetrievedItem]:
        normalized_names = sorted({_normalize_name(name) for name in entity_names if name.strip()})
        if not normalized_names:
            return []
        filters = _normalize_source_paths(source_paths)
        query = """
        UNWIND $normalized_names AS requested
        MATCH (matched:Entity)
        WHERE matched.normalized_name = requested
           OR any(alias IN coalesce(matched.aliases, []) WHERE toLower(alias) = requested)
        MATCH (candidate:Chunk)-[:MENTIONS]->(matched)
        WHERE size($source_paths) = 0 OR candidate.source_path IN $source_paths
        WITH candidate, collect(DISTINCT matched.name) AS matched_entities,
             count(DISTINCT requested) AS match_count
        OPTIONAL MATCH (candidate)-[:MENTIONS]->(all_entity:Entity)
        RETURN candidate.id AS chunk_id, candidate.document_id AS document_id,
               candidate.title AS title, candidate.text AS text, candidate.context AS context,
               candidate.source_path AS source_path, candidate.source_kind AS source_kind,
               toFloat(match_count) AS graph_score,
               collect(DISTINCT all_entity.name)[0..20] AS entities,
               matched_entities
        ORDER BY graph_score DESC
        LIMIT $limit
        """
        try:
            records, _, _ = await self.driver.execute_query(
                query,
                normalized_names=normalized_names,
                source_paths=filters,
                limit=limit,
                database_=self.settings.neo4j_database,
                routing_=RoutingControl.READ,
            )
        except _NEO4J_ERRORS as exc:
            raise GraphStoreError(f"Entity lookup failed: {exc}") from exc
        items: list[RetrievedItem] = []
        for record in records:
            item = _record_to_item(record, path="entity")
            item.retrieval_paths.extend(
                f"entity:{name}" for name in record.get("matched_entities", []) if name
            )
            items.append(item)
        return items

    async def graph_expand(
        self,
        seed_chunk_ids: list[str],
        limit: int,
        max_hops: int = 2,
        source_paths: list[str] | None = None,
    ) -> list[RetrievedItem]:
        del max_hops  # Parent-neighbour expansion is intentionally one hop in Community Edition.
        if not seed_chunk_ids:
            return []
        filters = _normalize_source_paths(source_paths)
        query = """
        MATCH (seed:Chunk)
        WHERE seed.id IN $seed_ids
          AND (size($source_paths) = 0 OR seed.source_path IN $source_paths)
        MATCH (seed)<-[:HAS_CHUNK]-(parent:ParentChunk)-[:HAS_CHUNK]->(candidate:Chunk)
        WHERE NOT candidate.id IN $seed_ids
          AND (size($source_paths) = 0 OR candidate.source_path IN $source_paths)
        WITH candidate, count(DISTINCT seed) AS path_count
        OPTIONAL MATCH (candidate)-[:MENTIONS]->(e:Entity)
        RETURN candidate.id AS chunk_id, candidate.document_id AS document_id,
               candidate.title AS title, candidate.text AS text, candidate.context AS context,
               candidate.source_path AS source_path, candidate.source_kind AS source_kind,
               toFloat(path_count) AS graph_score,
               coalesce(candidate.ordinal, 0) AS ordinal,
               collect(DISTINCT e.name)[0..20] AS entities,
               [] AS graph_paths
        ORDER BY graph_score DESC, ordinal ASC
        LIMIT $limit
        """
        try:
            records, _, _ = await self.driver.execute_query(
                query,
                seed_ids=seed_chunk_ids,
                source_paths=filters,
                limit=limit,
                database_=self.settings.neo4j_database,
                routing_=RoutingControl.READ,
            )
        except _NEO4J_ERRORS as exc:
            raise GraphStoreError(f"Graph expansion failed: {exc}") from exc
        items: list[RetrievedItem] = []
        for record in records:
            item = _record_to_item(record, path="graph")
            item.retrieval_paths.extend([str(value) for value in record.get("graph_paths", []) if value])
            items.append(item)
        return items

    async def list_documents(self) -> list[DocumentInfo]:
        query = """
        MATCH (d:Document)
        RETURN d.id AS document_id, d.title AS title,
               d.source_path AS source_path, d.source_kind AS source_kind
        ORDER BY toLower(d.title), d.source_path
        """
        try:
            records, _, _ = await self.driver.execute_query(
                query,
                database_=self.settings.neo4j_database,
                routing_=RoutingControl.READ,
            )
        except _NEO4J_ERRORS as exc:
            raise GraphStoreError(f"Failed to list documents: {exc}") from exc
        output: list[DocumentInfo] = []
        for record in records:
            raw_kind = str(record.get("source_kind") or SourceKind.TEXT.value)
            try:
                kind = SourceKind(raw_kind)
            except ValueError:
                kind = SourceKind.TEXT
            output.append(
                DocumentInfo(
                    document_id=str(record.get("document_id") or ""),
                    title=str(record.get("title") or "Untitled"),
                    source_path=str(record.get("source_path") or ""),
                    source_kind=kind,
                )
            )
        return output

    async def health(self) -> dict[str, Any]:
        try:
            records, _, _ = await self.driver.execute_query(
                "MATCH (d:Document) WITH count(d) AS documents MATCH (c:Chunk) RETURN documents, count(c) AS chunks",
                database_=self.settings.neo4j_database,
                routing_=RoutingControl.READ,
            )
        except _NEO4J_ERRORS as exc:
            raise GraphStoreError(f"Neo4j health query failed: {exc}") from exc
        row = records[0] if records else {}
        return {"documents": int(row.get("documents", 0)), "chunks": int(row.get("chunks", 0))}


def _bundle_parameters(bundle: IngestionBundle) -> dict[str, Any]:
    document = {
        "id": bundle.document_id,
        "title": bundle.title,
        "source_path": bundle.source_path,
        "source_kind": bundle.source_kind.value,
        "content_hash": bundle.content_hash,
        "metadata_json": _json_dump(bundle.metadata),
    }
    parents = [parent.model_dump(mode="json") for parent in bundle.parents]
    chunks: list[dict[str, Any]] = []
    for chunk in bundle.chunks:
        item = chunk.model_dump(mode="json")
        item["source_kind"] = chunk.source_kind.value
        item["metadata_json"] = _json_dump(chunk.metadata)
        item.pop("metadata", None)
        chunks.append(item)
    entities: list[dict[str, Any]] = []
    for entity in bundle.entities:
        item = entity.model_dump(mode="json")
        item["entity_type"] = entity.entity_type.value
        entities.append(item)
    relations = [relation.model_dump(mode="json") for relation in bundle.relations]
    return {
        "document": document,
        "parents": parents,
        "chunks": chunks,
        "entities": entities,
        "relations": relations,
    }


def _normalize_source_paths(source_paths: list[str] | None) -> list[str]:
    if not source_paths:
        return []
    return list(dict.fromkeys(path.strip() for path in source_paths if path.strip()))


def _candidate_limit(limit: int, source_paths: list[str], minimum: int) -> int:
    if not source_paths:
        return limit
    return min(10_000, max(limit * 20, minimum))


def _record_to_item(record: Any, path: str) -> RetrievedItem:
    source_kind_raw = str(record.get("source_kind") or SourceKind.TEXT.value)
    try:
        source_kind = SourceKind(source_kind_raw)
    except ValueError:
        source_kind = SourceKind.TEXT
    return RetrievedItem(
        chunk_id=str(record["chunk_id"]),
        document_id=str(record["document_id"]),
        title=str(record.get("title") or "Untitled"),
        text=str(record.get("text") or ""),
        context=str(record.get("context") or ""),
        source_path=str(record.get("source_path") or ""),
        source_kind=source_kind,
        vector_score=float(record.get("vector_score") or 0.0),
        keyword_score=float(record.get("keyword_score") or 0.0),
        graph_score=float(record.get("graph_score") or 0.0),
        entities=[str(value) for value in record.get("entities", []) if value],
        retrieval_paths=[path],
    )


def _lucene_escape(text: str) -> str:
    reserved = r'+-&|!(){}[]^"~*?:\\/'
    escaped = "".join(f"\\{char}" if char in reserved else char for char in text)
    tokens = [token for token in escaped.split() if token]
    return " ".join(tokens) or "*"


def _lucene_phrase(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('\"', '\\\"')
    return f'"{escaped}"'


def _normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
