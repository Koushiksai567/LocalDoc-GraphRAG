from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from .agent import AgenticGraphRAG
from .config import get_settings
from .graph_store import Neo4jGraphStore
from .ingestion import IngestionService
from .llm import LocalModelProvider
from .retriever import AgenticRetriever


async def _execute(args: argparse.Namespace) -> None:
    settings = get_settings()
    llm = LocalModelProvider(settings)
    store = Neo4jGraphStore(settings)
    try:
        await asyncio.gather(store.verify_connectivity(), llm.verify_connectivity())
        await store.ensure_schema()
        await store.wait_for_indexes()
        retriever = AgenticRetriever(settings, llm, store)
        agent = AgenticGraphRAG(settings, llm, retriever)
        ingestion = IngestionService(settings, llm, store)
        if args.command == "ingest":
            result = await ingestion.ingest_path(
                Path(args.path), recursive=args.recursive, replace_existing=not args.no_replace
            )
            print(result.model_dump_json(indent=2))
        elif args.command == "ask":
            result = await agent.answer(args.query)
            print(result.model_dump_json(indent=2))
        elif args.command == "health":
            print(json.dumps(await store.health(), indent=2))
    finally:
        await store.close()
        await llm.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="graphrag", description="Enterprise Agentic GraphRAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest = subparsers.add_parser("ingest", help="Ingest a file or directory")
    ingest.add_argument("path")
    ingest.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    ingest.add_argument("--no-replace", action="store_true")
    ask = subparsers.add_parser("ask", help="Ask a grounded question")
    ask.add_argument("query")
    subparsers.add_parser("health", help="Show indexed counts")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_execute(args))


if __name__ == "__main__":
    main()
