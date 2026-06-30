from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from .agent import AgenticGraphRAG
from .config import Settings, get_settings
from .graph_store import Neo4jGraphStore
from .llm import LocalModelProvider
from .retriever import AgenticRetriever

logger = logging.getLogger(__name__)


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=3)
    reference: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class EvalResult:
    query: str
    reference: str
    response: str
    faithfulness: float
    answer_relevance: float
    context_recall: float
    contexts: list[str]
    trace_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "reference": self.reference,
            "response": self.response,
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_recall": self.context_recall,
            "contexts": self.contexts,
            "trace_id": self.trace_id,
        }


class RagasEvaluator:
    """RAGAS metrics evaluated entirely with local Ollama and HF models.

    The OpenAI Python package is used only as an open-source protocol client for
    Ollama's local OpenAI-compatible endpoint. The value "ollama" is a dummy key
    required by the client library; no account, billing, or external API is used.
    """

    def __init__(self, settings: Settings, agent: AgenticGraphRAG) -> None:
        from ragas.embeddings import HuggingfaceEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerRelevancy, ContextRecall, Faithfulness

        self.settings = settings
        self.agent = agent
        self._async_client = AsyncOpenAI(
            api_key="ollama",
            base_url=f"{settings.ollama_base_url}/v1",
            timeout=settings.request_timeout_seconds,
            max_retries=2,
        )
        evaluator_llm = llm_factory(
            settings.reasoning_model,
            provider="openai",
            client=self._async_client,
            system_prompt=(
                "You are an evaluation judge. Return only the structured output requested by the metric. "
                "Use the supplied evidence and do not use external knowledge."
            ),
        )
        evaluator_embeddings = HuggingfaceEmbeddings(
            model_name=settings.embedding_model,
            cache_folder=str(settings.model_cache_dir / "huggingface"),
            model_kwargs={"device": settings.embedding_device},
            encode_kwargs={"normalize_embeddings": settings.normalize_embeddings},
        )
        self.faithfulness_metric = Faithfulness(llm=evaluator_llm)
        self.answer_relevance_metric = AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )
        self.context_recall_metric = ContextRecall(llm=evaluator_llm)

    async def close(self) -> None:
        await self._async_client.close()

    async def evaluate_case(self, case: EvalCase) -> EvalResult:
        answer = await self.agent.answer(case.query)
        contexts = [item.context for item in answer.contexts]
        faithfulness, relevance, recall = await asyncio.gather(
            self.faithfulness_metric.ascore(
                user_input=case.query,
                response=answer.answer,
                retrieved_contexts=contexts,
            ),
            self.answer_relevance_metric.ascore(
                user_input=case.query,
                response=answer.answer,
            ),
            self.context_recall_metric.ascore(
                user_input=case.query,
                reference=case.reference,
                retrieved_contexts=contexts,
            ),
        )
        return EvalResult(
            query=case.query,
            reference=case.reference,
            response=answer.answer,
            faithfulness=float(faithfulness.value),
            answer_relevance=float(relevance.value),
            context_recall=float(recall.value),
            contexts=contexts,
            trace_id=answer.trace_id,
        )

    async def evaluate_file(self, dataset_path: Path, output_dir: Path) -> dict[str, float]:
        cases = await asyncio.to_thread(load_cases, dataset_path)
        results: list[EvalResult] = []
        for index, case in enumerate(cases, start=1):
            logger.info("Evaluating case %d/%d", index, len(cases))
            results.append(await self.evaluate_case(case))
        summary = {
            "faithfulness": fmean(result.faithfulness for result in results),
            "answer_relevance": fmean(result.answer_relevance for result in results),
            "context_recall": fmean(result.context_recall for result in results),
            "cases": float(len(results)),
        }
        await asyncio.to_thread(write_eval_results, output_dir, results, summary)
        return summary


def write_eval_results(
    output_dir: Path,
    results: list[EvalResult],
    summary: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ragas_results.json"
    csv_path = output_dir / "ragas_results.csv"
    json_path.write_text(
        json.dumps([result.as_dict() for result in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query",
                "reference",
                "response",
                "faithfulness",
                "answer_relevance",
                "context_recall",
                "trace_id",
            ],
        )
        writer.writeheader()
        for result in results:
            row = result.as_dict()
            row.pop("contexts")
            writer.writerow(row)
    (output_dir / "ragas_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                cases.append(EvalCase.model_validate_json(stripped))
            except Exception as exc:
                raise ValueError(f"Invalid evaluation case on line {line_number}: {exc}") from exc
    if not cases:
        raise ValueError(f"Evaluation dataset is empty: {path}")
    return cases


async def _run(dataset: Path, output_dir: Path) -> None:
    settings = get_settings()
    llm = LocalModelProvider(settings)
    store = Neo4jGraphStore(settings)
    retriever = AgenticRetriever(settings, llm, store)
    agent = AgenticGraphRAG(settings, llm, retriever)
    evaluator = RagasEvaluator(settings, agent)
    try:
        await asyncio.gather(store.verify_connectivity(), llm.verify_connectivity())
        summary = await evaluator.evaluate_file(dataset, output_dir)
        print(json.dumps(summary, indent=2))
    finally:
        await evaluator.close()
        await store.close()
        await llm.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local RAGAS evaluation against Agentic GraphRAG")
    parser.add_argument("--dataset", type=Path, default=Path("eval/evalset.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_run(args.dataset, args.output))


if __name__ == "__main__":
    main()
