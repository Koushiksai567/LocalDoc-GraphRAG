from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint

from .agent import AgenticGraphRAG, QueryGuardrailError
from .config import Settings, get_settings
from .graph_store import GraphStoreError, Neo4jGraphStore
from .ingestion import IngestionError, IngestionService
from .llm import LLMError, LocalModelProvider
from .models import (
    AnswerResult,
    DeleteDocumentRequest,
    DeleteDocumentResponse,
    DocumentInfo,
    IngestRequest,
    IngestResponse,
    QueryRequest,
)
from .retriever import AgenticRetriever, RetrievalError

logger = logging.getLogger(__name__)


async def _warm_local_models(llm: LocalModelProvider, retriever: AgenticRetriever) -> None:
    try:
        logger.info("Warming local models in the background")
        await asyncio.gather(llm.warmup(), retriever.warmup())
        logger.info("Local model warmup complete")
    except Exception:
        logger.exception("Background model warmup failed; models will load lazily on demand")


async def _await_model_warmup(request: Request) -> None:
    task: asyncio.Task[None] | None = getattr(request.app.state, "model_warmup_task", None)
    if task is not None and not task.done():
        await task



def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings)
    llm = LocalModelProvider(settings)
    store = Neo4jGraphStore(settings)
    warmup_task: asyncio.Task[None] | None = None
    try:
        await asyncio.gather(store.verify_connectivity(), llm.verify_connectivity())
        await store.ensure_schema()
        await store.wait_for_indexes()
        retriever = AgenticRetriever(settings, llm, store)
        if settings.preload_models:
            warmup_task = asyncio.create_task(
                _warm_local_models(llm, retriever),
                name="graphrag-model-warmup",
            )
        agent = AgenticGraphRAG(settings, llm, retriever)
        ingestion = IngestionService(settings, llm, store)
        app.state.settings = settings
        app.state.llm = llm
        app.state.store = store
        app.state.agent = agent
        app.state.ingestion = ingestion
        app.state.model_warmup_task = warmup_task
        logger.info("%s started", settings.app_name)
        yield
    finally:
        if warmup_task is not None and not warmup_task.done():
            warmup_task.cancel()
            with suppress(asyncio.CancelledError):
                await warmup_task
        await store.close()
        await llm.close()
        logger.info("Application shutdown complete")


app = FastAPI(
    title="Free Open-Source Agentic GraphRAG",
    version="3.8.0",
    description="Fast local GraphRAG with one-command startup, on-demand document indexing, stale-document cleanup, Ollama, and Neo4j Community.",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context(request: Request, call_next: RequestResponseEndpoint) -> Response:
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    response.headers["x-process-time-ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
    return response


def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None),
) -> None:
    settings: Settings = request.app.state.settings
    if settings.api_key is None:
        return
    expected = settings.api_key.get_secret_value()
    if not x_api_key or not _constant_time_equal(x_api_key, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    store: Neo4jGraphStore = request.app.state.store
    llm: LocalModelProvider = request.app.state.llm
    graph_health, model_health = await asyncio.gather(store.health(), llm.verify_connectivity(require_models=False))
    settings: Settings = request.app.state.settings
    warmup_task: asyncio.Task[None] | None = getattr(request.app.state, "model_warmup_task", None)
    model_state = "ready" if warmup_task is None or warmup_task.done() else "warming"
    return {
        "status": "ok",
        "mode": "fast" if settings.fast_mode else "full",
        "pdf_parser": settings.pdf_parser,
        "model_state": model_state,
        **graph_health,
        "models": model_health,
    }


@app.post("/v1/query", response_model=AnswerResult, dependencies=[Depends(require_api_key)])
async def query(payload: QueryRequest, request: Request) -> AnswerResult:
    await _await_model_warmup(request)
    agent: AgenticGraphRAG = request.app.state.agent
    result = await agent.answer(
        payload.query,
        payload.conversation_id,
        source_paths=payload.source_paths,
    )
    if not payload.include_contexts:
        result.contexts = []
    return result


def _resolve_document_delete_paths(raw_source_path: str, raw_data_root: Path) -> tuple[Path, Path]:
    return Path(raw_source_path).expanduser().resolve(), raw_data_root.expanduser().resolve()


@app.get("/v1/documents", response_model=list[DocumentInfo], dependencies=[Depends(require_api_key)])
async def documents(request: Request) -> list[DocumentInfo]:
    store: Neo4jGraphStore = request.app.state.store
    return await store.list_documents()


@app.delete(
    "/v1/documents",
    response_model=DeleteDocumentResponse,
    dependencies=[Depends(require_api_key)],
)
async def delete_document(
    payload: DeleteDocumentRequest,
    request: Request,
) -> DeleteDocumentResponse:
    settings: Settings = request.app.state.settings
    store: Neo4jGraphStore = request.app.state.store
    source_path, data_root = await asyncio.to_thread(
        _resolve_document_delete_paths,
        payload.source_path,
        settings.data_dir,
    )
    if source_path == data_root or not source_path.is_relative_to(data_root):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only documents under {data_root} can be deleted through this endpoint.",
        )
    existed = await store.document_hash(str(source_path)) is not None
    if existed:
        await store.delete_document(str(source_path))
    return DeleteDocumentResponse(deleted=existed, source_path=str(source_path))


@app.post("/v1/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
async def ingest(payload: IngestRequest, request: Request) -> IngestResponse:
    ingestion: IngestionService = request.app.state.ingestion
    return await ingestion.ingest_path(
        Path(payload.path),
        recursive=payload.recursive,
        replace_existing=payload.replace_existing,
    )


@app.exception_handler(QueryGuardrailError)
async def query_guardrail_handler(_request: Request, exc: QueryGuardrailError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": "invalid_query", "detail": str(exc)})


@app.exception_handler(IngestionError)
async def ingestion_error_handler(_request: Request, exc: IngestionError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": "ingestion_error", "detail": str(exc)})


@app.exception_handler(GraphStoreError)
@app.exception_handler(LLMError)
@app.exception_handler(RetrievalError)
async def dependency_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Dependency failure", exc_info=exc)
    return JSONResponse(status_code=503, content={"error": "dependency_unavailable", "detail": str(exc)})


@app.exception_handler(Exception)
async def unexpected_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unexpected backend failure", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": "The backend encountered an unexpected error. Check the backend terminal for details.",
        },
    )


def _constant_time_equal(left: str, right: str) -> bool:
    import hmac

    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "enterprise_graphrag.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
