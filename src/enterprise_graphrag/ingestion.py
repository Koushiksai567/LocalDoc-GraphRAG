from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import shutil
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .graph_store import Neo4jGraphStore
from .llm import LocalModelProvider
from .models import (
    Chunk,
    EntityRecord,
    EntityType,
    ExtractionResult,
    IngestionBundle,
    IngestResponse,
    ParentChunk,
    RawDocument,
    RelationRecord,
    SourceKind,
)

logger = logging.getLogger(__name__)
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
_CODE_EXTENSIONS = {".py", ".java", ".js", ".ts"}
_SCHEMA_EXTENSIONS = {".csv", ".sql", ".json", ".yaml", ".yml"}


class IngestionError(RuntimeError):
    pass


@dataclass(slots=True)
class TextPiece:
    title: str
    text: str
    ordinal: int


class DocumentLoader:
    def __init__(self, settings: Settings, llm: LocalModelProvider) -> None:
        self.settings = settings
        self.llm = llm

    async def load(self, path: Path, *, content_hash: str | None = None) -> RawDocument:
        is_file, file_stat, resolved_path = await asyncio.gather(
            asyncio.to_thread(path.is_file),
            asyncio.to_thread(path.stat),
            asyncio.to_thread(path.resolve),
        )
        if not is_file:
            raise IngestionError(f"Not a file: {path}")
        extension = path.suffix.lower()
        if file_stat.st_size > self.settings.max_file_size_mb * 1024 * 1024:
            raise IngestionError(
                f"File exceeds {self.settings.max_file_size_mb} MB ingestion limit: {path}"
            )
        if extension not in self.settings.extension_set:
            raise IngestionError(f"Unsupported file extension: {extension}")
        if content_hash is None:
            content_hash = await asyncio.to_thread(
                _indexed_content_hash,
                path,
                self.settings.indexing_version,
            )
        if extension in _IMAGE_EXTENSIONS:
            description = await self.llm.describe_image(path)
            return RawDocument(
                path=resolved_path,
                title=path.stem,
                source_kind=SourceKind.DIAGRAM,
                content=f"# {path.stem}\n\nImage description:\n{description}",
                content_hash=content_hash,
                metadata={"extension": extension, "size_bytes": file_stat.st_size},
                figure_descriptions=[description],
            )
        if extension in {".md", ".mmd", ".txt", ".csv", ".sql", ".json", ".yaml", ".yml", ".py", ".java", ".js", ".ts"}:
            content = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")
            return RawDocument(
                path=resolved_path,
                title=path.stem,
                source_kind=_source_kind(extension),
                content=content,
                content_hash=content_hash,
                metadata={"extension": extension, "size_bytes": file_stat.st_size},
            )
        if extension == ".pdf" and self.settings.pdf_parser == "pymupdf":
            try:
                markdown, figures = await asyncio.to_thread(self._convert_pdf_fast, path)
            except Exception:
                if not self.settings.enable_docling_fallback:
                    raise
                logger.warning("Fast PDF parsing failed for %s; falling back to Docling", path)
                markdown, figures = await asyncio.to_thread(self._convert_with_docling, path)
        else:
            markdown, figures = await asyncio.to_thread(self._convert_with_docling, path)
        descriptions = await self._describe_figures(figures, markdown[:3_000]) if self.settings.enable_vision else []
        if descriptions:
            markdown += "\n\n# Extracted figure and table descriptions\n\n" + "\n\n".join(
                f"## Figure {index + 1}\n{description}" for index, description in enumerate(descriptions)
            )
        return RawDocument(
            path=resolved_path,
            title=path.stem,
            source_kind=SourceKind.DOCUMENT,
            content=markdown,
            content_hash=content_hash,
            metadata={
                "extension": extension,
                "size_bytes": file_stat.st_size,
                "extracted_figures": len(figures),
            },
            figure_descriptions=descriptions,
        )


    def _convert_pdf_fast(self, path: Path) -> tuple[str, list[Path]]:
        """Extract selectable PDF text and preserve page-aware section headings."""
        try:
            import pymupdf
        except ImportError as exc:
            raise IngestionError("PyMuPDF is required for fast PDF ingestion") from exc

        try:
            document = pymupdf.open(path)
        except Exception as exc:
            raise IngestionError(f"Could not open PDF {path}: {exc}") from exc

        sections: list[str] = []
        total_chars = 0
        active_section: str | None = None
        try:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text", sort=True).strip()
                if not text:
                    continue
                total_chars += len(text)
                page_sections, active_section = _pdf_page_sections(
                    page_number,
                    text,
                    active_section,
                )
                sections.extend(page_sections)
        finally:
            document.close()

        if total_chars < self.settings.pdf_min_text_chars:
            raise IngestionError(
                f"PDF {path.name} contains too little selectable text ({total_chars} characters). "
                "It is probably scanned. Set ENABLE_DOCLING_FALLBACK=true or convert it with OCR first."
            )
        return "\n\n".join(sections), []

    def _convert_with_docling(self, path: Path) -> tuple[str, list[Path]]:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            raise IngestionError("Docling is required for PDF/DOCX/PPTX/HTML ingestion") from exc

        artifact_dir = self.settings.artifacts_dir / _stable_id(str(path.resolve()), length=20)
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        if path.suffix.lower() == ".pdf":
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import PdfFormatOption

            options = PdfPipelineOptions()
            options.do_ocr = False
            options.images_scale = 1.0
            options.generate_picture_images = self.settings.enable_vision
            converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )
        else:
            converter = DocumentConverter()

        try:
            result = converter.convert(path)
            document = result.document
            markdown = document.export_to_markdown()
        except Exception as exc:
            raise IngestionError(f"Docling conversion failed for {path}: {exc}") from exc

        figures: list[Path] = []
        try:
            from docling_core.types.doc import PictureItem, TableItem

            picture_count = 0
            table_count = 0
            for element, _level in document.iterate_items():
                if isinstance(element, PictureItem):
                    picture_count += 1
                    image_path = artifact_dir / f"picture-{picture_count}.png"
                    element.get_image(document).save(image_path, "PNG")
                    figures.append(image_path)
                elif isinstance(element, TableItem):
                    table_count += 1
                    image_path = artifact_dir / f"table-{table_count}.png"
                    element.get_image(document).save(image_path, "PNG")
                    figures.append(image_path)
        except Exception as exc:
            logger.warning("Figure extraction failed for %s; continuing with document text: %s", path, exc)
        return markdown, figures

    async def _describe_figures(self, paths: list[Path], context_hint: str) -> list[str]:
        semaphore = asyncio.Semaphore(self.settings.figure_caption_concurrency)

        async def describe(path: Path) -> str:
            async with semaphore:
                return await self.llm.describe_image(path, context_hint=context_hint)

        descriptions: list[str] = []
        batch_size = self.settings.figure_caption_concurrency
        for start in range(0, len(paths), batch_size):
            figure_batch = paths[start : start + batch_size]
            results = await asyncio.gather(*(describe(path) for path in figure_batch), return_exceptions=True)
            for path, result in zip(figure_batch, results, strict=True):
                if isinstance(result, BaseException):
                    logger.warning("Skipping failed figure description for %s: %s", path, result)
                else:
                    descriptions.append(result)
        return descriptions


class ContextualChunker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.target_chars = settings.chunk_size_tokens * 4
        self.overlap_chars = settings.chunk_overlap_tokens * 4

    def split(self, document: RawDocument) -> list[TextPiece]:
        sections = self._sections(document.content, document.title)
        pieces: list[TextPiece] = []
        ordinal = 0
        for title, section_text in sections:
            for chunk in self._window(section_text):
                clean = chunk.strip()
                if clean:
                    pieces.append(TextPiece(title=title, text=clean, ordinal=ordinal))
                    ordinal += 1
        if not pieces and document.content.strip():
            pieces.append(TextPiece(title=document.title, text=document.content.strip(), ordinal=0))
        return pieces

    @staticmethod
    def _sections(text: str, fallback_title: str) -> list[tuple[str, str]]:
        heading_pattern = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")
        matches = list(heading_pattern.finditer(text))
        if not matches:
            return [(fallback_title, text)]
        sections: list[tuple[str, str]] = []
        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                sections.append((fallback_title, preamble))
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            title = match.group(2).strip()
            body = text[start:end].strip()
            if body:
                sections.append((title, body))
        return sections

    def _window(self, text: str) -> list[str]:
        if len(text) <= self.target_chars:
            return [text]
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        units: list[str] = []
        for paragraph in paragraphs:
            if len(paragraph) <= self.target_chars:
                units.append(paragraph)
            else:
                units.extend(_split_long_unit(paragraph, self.target_chars))
        chunks: list[str] = []
        current: list[str] = []
        current_length = 0
        for unit in units:
            addition = len(unit) + (2 if current else 0)
            if current and current_length + addition > self.target_chars:
                chunks.append("\n\n".join(current))
                overlap = _tail_text(chunks[-1], self.overlap_chars)
                current = [overlap, unit] if overlap else [unit]
                current_length = sum(len(item) for item in current) + 2 * (len(current) - 1)
            else:
                current.append(unit)
                current_length += addition
        if current:
            chunks.append("\n\n".join(current))
        return chunks


class IngestionService:
    def __init__(self, settings: Settings, llm: LocalModelProvider, store: Neo4jGraphStore) -> None:
        self.settings = settings
        self.llm = llm
        self.store = store
        self.loader = DocumentLoader(settings, llm)
        self.chunker = ContextualChunker(settings)
        self._llm_semaphore = asyncio.Semaphore(settings.llm_concurrency)

    async def ingest_path(self, path: Path, *, recursive: bool = True, replace_existing: bool = True) -> IngestResponse:
        files = await asyncio.to_thread(self._discover, path, recursive)
        semaphore = asyncio.Semaphore(self.settings.ingestion_concurrency)

        async def process(file_path: Path) -> tuple[str, str | None, str | None]:
            async with semaphore:
                try:
                    resolved_path = await asyncio.to_thread(file_path.resolve)
                    source_path = str(resolved_path)
                    content_hash = await asyncio.to_thread(
                        _indexed_content_hash,
                        resolved_path,
                        self.settings.indexing_version,
                    )
                    existing_hash = await self.store.document_hash(source_path)
                    if existing_hash == content_hash:
                        return "skipped", None, None
                    if existing_hash and not replace_existing:
                        return "skipped", None, None
                    document = await self.loader.load(
                        resolved_path,
                        content_hash=content_hash,
                    )
                    bundle = await self._build_bundle(document)
                    if existing_hash and replace_existing:
                        await self.store.replace_bundle(bundle)
                    else:
                        await self.store.upsert_bundle(bundle)
                    return "processed", bundle.document_id, None
                except Exception as exc:
                    logger.exception("Ingestion failed for %s", file_path)
                    return "failed", None, f"{file_path}: {exc}"

        results: list[tuple[str, str | None, str | None]] = []
        batch_size = self.settings.ingestion_concurrency
        for start in range(0, len(files), batch_size):
            file_batch = files[start : start + batch_size]
            results.extend(await asyncio.gather(*(process(file_path) for file_path in file_batch)))
        return IngestResponse(
            processed=sum(status == "processed" for status, _, _ in results),
            skipped=sum(status == "skipped" for status, _, _ in results),
            failed=sum(status == "failed" for status, _, _ in results),
            document_ids=[document_id for _, document_id, _ in results if document_id],
            errors=[error for _, _, error in results if error],
        )

    def _discover(self, path: Path, recursive: bool) -> list[Path]:
        path = path.expanduser().resolve()
        if not any(path == root or path.is_relative_to(root) for root in self.settings.ingestion_roots):
            allowed = ", ".join(str(root) for root in self.settings.ingestion_roots)
            raise IngestionError(f"Path {path} is outside allowed ingestion roots: {allowed}")
        if path.is_file():
            return [path]
        if not path.is_dir():
            raise IngestionError(f"Path does not exist: {path}")
        iterator: Iterable[Path] = path.rglob("*") if recursive else path.glob("*")
        files: list[Path] = []
        for item in iterator:
            if not item.is_file() or item.suffix.lower() not in self.settings.extension_set:
                continue
            resolved_item = item.resolve()
            if any(
                resolved_item == root or resolved_item.is_relative_to(root)
                for root in self.settings.ingestion_roots
            ):
                files.append(resolved_item)
            else:
                logger.warning("Skipping file that resolves outside allowed ingestion roots: %s", item)
        files.sort()
        if not files:
            raise IngestionError(f"No supported files found under {path}")
        return files

    async def _build_bundle(self, document: RawDocument) -> IngestionBundle:
        document_id = _stable_id(f"document:{document.path}")
        pieces = self.chunker.split(document)
        if not pieces:
            raise IngestionError(f"No extractable content found in {document.path}")
        groups = [
            pieces[index : index + self.settings.parent_children]
            for index in range(0, len(pieces), self.settings.parent_children)
        ]
        if self.settings.enable_parent_llm_summaries:
            summaries: list[str] = []
            for start in range(0, len(groups), self.settings.llm_concurrency):
                parent_batch = groups[start : start + self.settings.llm_concurrency]
                summaries.extend(
                    await asyncio.gather(*(self._summarize_parent(document, group) for group in parent_batch))
                )
        else:
            summaries = [self._extractive_parent_summary(document, group) for group in groups]
        parents: list[ParentChunk] = []
        chunk_specs: list[tuple[str, TextPiece, str]] = []
        for parent_ordinal, (group, parent_summary) in enumerate(zip(groups, summaries, strict=True)):
            parent_id = _stable_id(f"{document_id}:parent:{parent_ordinal}")
            parent_title = group[0].title if group else document.title
            parents.append(
                ParentChunk(
                    id=parent_id,
                    document_id=document_id,
                    ordinal=parent_ordinal,
                    summary=parent_summary,
                    title=parent_title,
                )
            )
            for piece in group:
                chunk_specs.append((parent_id, piece, parent_summary))

        contexts = [self._contextualize(document, piece, summary) for _, piece, summary in chunk_specs]
        embeddings = await self.llm.embed_batched(contexts)
        chunks: list[Chunk] = []
        for (parent_id, piece, summary), context, embedding in zip(chunk_specs, contexts, embeddings, strict=True):
            chunk_id = _stable_id(f"{document_id}:chunk:{piece.ordinal}:{hashlib.sha256(piece.text.encode()).hexdigest()}")
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    parent_id=parent_id,
                    ordinal=piece.ordinal,
                    title=piece.title,
                    text=piece.text,
                    context=context,
                    source_path=str(document.path),
                    source_kind=document.source_kind,
                    content_hash=hashlib.sha256(piece.text.encode()).hexdigest(),
                    embedding=embedding,
                    metadata={
                        "parent_summary": summary,
                        **_piece_metadata(piece),
                        **document.metadata,
                    },
                )
            )

        if self.settings.enable_llm_graph_extraction:
            extractions: list[ExtractionResult] = []
            for start in range(0, len(chunks), self.settings.llm_concurrency):
                chunk_batch = chunks[start : start + self.settings.llm_concurrency]
                extractions.extend(await asyncio.gather(*(self._extract(chunk) for chunk in chunk_batch)))
            entities, relations = self._consolidate_graph(chunks, extractions)
        else:
            entities, relations = [], []
        return IngestionBundle(
            document_id=document_id,
            title=document.title,
            source_path=str(document.path),
            source_kind=document.source_kind,
            content_hash=document.content_hash,
            metadata=document.metadata,
            parents=parents,
            chunks=chunks,
            entities=entities,
            relations=relations,
        )

    @staticmethod
    def _extractive_parent_summary(document: RawDocument, group: list[TextPiece]) -> str:
        """Create compact parent context without an LLM call."""
        lines = [f"Document: {document.title}"]
        remaining = 1_200
        for piece in group:
            snippet = re.sub(r"\s+", " ", piece.text).strip()
            snippet = snippet[: min(320, remaining)]
            if not snippet:
                continue
            lines.append(f"{piece.title}: {snippet}")
            remaining -= len(snippet)
            if remaining <= 0:
                break
        return "\n".join(lines)

    async def _summarize_parent(self, document: RawDocument, group: list[TextPiece]) -> str:
        text = "\n\n".join(f"[{piece.title}]\n{piece.text}" for piece in group)
        if len(text) < 600:
            return text[:1_500]
        prompt = (
            f"Document: {document.title}\nSource type: {document.source_kind.value}\n\n"
            f"Create a dense retrieval summary of this parent group. Preserve component names, dependencies, "
            f"interfaces, tables/columns, failure modes, and quantitative details. Do not add facts.\n\n{text[:18_000]}"
        )
        try:
            async with self._llm_semaphore:
                return await self.llm.generate(
                    prompt,
                    instructions="You create factual parent summaries for RAPTOR-style hierarchical retrieval.",
                    max_output_tokens=900,
                )
        except Exception as exc:
            logger.warning("Parent summarization failed; using extractive fallback: %s", exc)
            return text[:1_500]

    def _contextualize(self, document: RawDocument, piece: TextPiece, parent_summary: str) -> str:
        return (
            f"Document: {document.title}\n"
            f"Source: {document.path}\n"
            f"Source kind: {document.source_kind.value}\n"
            f"Section: {piece.title}\n"
            f"Parent context: {parent_summary}\n\n"
            f"Child chunk:\n{piece.text}"
        )

    async def _extract(self, chunk: Chunk) -> ExtractionResult:
        prompt = (
            "Extract only explicitly supported technical entities and directional relationships from this chunk. "
            "Use stable, concise entity names. Relationship types must be uppercase snake case such as CALLS, "
            "READS_FROM, WRITES_TO, DEPENDS_ON, PUBLISHES_TO, SUBSCRIBES_TO, OWNS, CONTAINS, FAILS_WHEN, or ROUTES_TO. "
            "Do not invent missing nodes or edges.\n\n"
            f"Title: {chunk.title}\nContext:\n{chunk.context[:20_000]}"
        )
        try:
            async with self._llm_semaphore:
                return await self.llm.structured(
                    prompt,
                    instructions="You are a precise enterprise knowledge-graph information extraction engine.",
                    schema=ExtractionResult,
                    max_output_tokens=3_000,
                )
        except Exception as exc:
            logger.warning("Entity extraction failed for chunk %s; continuing without graph facts: %s", chunk.id, exc)
            return ExtractionResult()

    def _consolidate_graph(
        self,
        chunks: list[Chunk],
        extractions: list[ExtractionResult],
    ) -> tuple[list[EntityRecord], list[RelationRecord]]:
        entity_map: dict[tuple[str, EntityType], EntityRecord] = {}
        alias_lookup: dict[tuple[str, str], str] = {}
        relations: list[RelationRecord] = []

        for chunk, extraction in zip(chunks, extractions, strict=True):
            for entity in extraction.entities:
                normalized = _normalize_name(entity.name)
                key = (normalized, entity.entity_type)
                entity_id = _stable_id(f"entity:{entity.entity_type.value}:{normalized}")
                if key not in entity_map:
                    entity_map[key] = EntityRecord(
                        id=entity_id,
                        name=entity.name,
                        normalized_name=normalized,
                        entity_type=entity.entity_type,
                        description=entity.description,
                        aliases=entity.aliases,
                        chunk_ids=[chunk.id],
                    )
                else:
                    existing = entity_map[key]
                    if chunk.id not in existing.chunk_ids:
                        existing.chunk_ids.append(chunk.id)
                    existing.aliases = sorted(set(existing.aliases + entity.aliases), key=str.casefold)
                    if len(entity.description) > len(existing.description):
                        existing.description = entity.description
                alias_lookup[(normalized, chunk.id)] = entity_id
                for alias in entity.aliases:
                    alias_lookup[(_normalize_name(alias), chunk.id)] = entity_id

        global_by_name: dict[str, list[str]] = defaultdict(list)
        for record in entity_map.values():
            global_by_name[record.normalized_name].append(record.id)
            for alias in record.aliases:
                global_by_name[_normalize_name(alias)].append(record.id)

        for chunk, extraction in zip(chunks, extractions, strict=True):
            for relation in extraction.relations:
                source_id = alias_lookup.get((_normalize_name(relation.source), chunk.id))
                target_id = alias_lookup.get((_normalize_name(relation.target), chunk.id))
                if source_id is None:
                    candidates = global_by_name.get(_normalize_name(relation.source), [])
                    source_id = candidates[0] if len(candidates) == 1 else None
                if target_id is None:
                    candidates = global_by_name.get(_normalize_name(relation.target), [])
                    target_id = candidates[0] if len(candidates) == 1 else None
                if not source_id or not target_id or source_id == target_id:
                    continue
                relation_type = re.sub(r"[^A-Z0-9_]+", "_", relation.relation_type.upper()).strip("_") or "RELATED_TO"
                relation_id = _stable_id(
                    f"relation:{source_id}:{relation_type}:{target_id}:{chunk.id}"
                )
                relations.append(
                    RelationRecord(
                        id=relation_id,
                        source_entity_id=source_id,
                        target_entity_id=target_id,
                        relation_type=relation_type,
                        description=relation.description,
                        confidence=relation.confidence,
                        evidence_chunk_id=chunk.id,
                    )
                )
        return list(entity_map.values()), relations


_PDF_SECTION_HEADING = re.compile(
    r"^\s*(\d+(?:\.\d+)*)\.\s+([A-Z][^\n]{1,78}?)(?::)?\s*$"
)


def _pdf_page_sections(
    page_number: int,
    text: str,
    active_section: str | None,
) -> tuple[list[str], str | None]:
    """Split a PDF page on short numbered headings while retaining page provenance."""
    segments: list[tuple[str, list[str]]] = []
    current_title = active_section or f"Page {page_number}"
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        body = "\n".join(current_lines).strip()
        if body:
            segments.append((current_title, current_lines))
        current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = _PDF_SECTION_HEADING.fullmatch(line)
        if match and _looks_like_policy_heading(line, match.group(2)):
            flush()
            active_section = f"{match.group(1)}. {match.group(2).rstrip(':').strip()}"
            current_title = active_section
            current_lines.append(line)
        else:
            current_lines.append(raw_line)
    flush()

    output: list[str] = []
    for title, lines in segments:
        body = "\n".join(lines).strip()
        heading = (
            f"Page {page_number}"
            if title == f"Page {page_number}"
            else f"Page {page_number} — {title}"
        )
        output.append(f"# {heading}\n\n{body}")
    if not output:
        fallback = active_section or f"Page {page_number}"
        output.append(f"# Page {page_number} — {fallback}\n\n{text}")
    return output, active_section


def _looks_like_policy_heading(line: str, title: str) -> bool:
    normalized = title.strip().rstrip(":")
    if len(line) > 90 or len(normalized.split()) > 8:
        return False
    if re.search(r"\.{3,}\s*\d+$", line):
        return False
    if normalized.endswith((".", ";", ",")):
        return False
    return True


def _piece_metadata(piece: TextPiece) -> dict[str, str | int]:
    match = re.match(r"^Page\s+(\d+)\s+—\s+(.+)$", piece.title)
    if not match:
        return {"section": piece.title}
    return {"page": int(match.group(1)), "section": match.group(2).strip()}


def _source_kind(extension: str) -> SourceKind:
    if extension == ".mmd":
        return SourceKind.DIAGRAM
    if extension in _CODE_EXTENSIONS:
        return SourceKind.CODE
    if extension in _SCHEMA_EXTENSIONS:
        return SourceKind.SCHEMA
    return SourceKind.TEXT


def _indexed_content_hash(path: Path, indexing_version: str) -> str:
    raw_hash = _file_hash(path)
    return hashlib.sha256(f"{raw_hash}:{indexing_version}".encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_id(value: str, length: int = 32) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:length]


def _normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def _split_long_unit(text: str, target_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?;])\s+|\n", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > target_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(sentence[index : index + target_chars] for index in range(0, len(sentence), target_chars))
        elif not current:
            current = sentence
        elif len(current) + len(sentence) + 1 <= target_chars:
            current += " " + sentence
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def _tail_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    tail = text[-max_chars:]
    first_space = tail.find(" ")
    return tail[first_space + 1 :] if first_space >= 0 else tail
