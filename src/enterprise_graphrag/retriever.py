from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from .config import Settings
from .graph_store import Neo4jGraphStore
from .llm import LocalModelProvider
from .models import RetrievedItem

logger = logging.getLogger(__name__)
_WORD = re.compile(r"[A-Za-z0-9_./:-]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do", "does",
    "for", "from", "how", "i", "if", "in", "is", "it", "of", "on", "or", "should",
    "that", "the", "their", "this", "to", "was", "what", "when", "where", "which", "who",
    "why", "with", "would",
}


class RetrievalError(RuntimeError):
    pass


class CrossEncoderReranker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any | None = None
        self._lock = asyncio.Lock()

    async def warmup(self) -> None:
        if not self.settings.enable_neural_reranker:
            return
        try:
            await self._get_model()
        except RetrievalError as exc:
            logger.warning("Neural reranker warmup failed; deterministic reranking will be used: %s", exc)

    async def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is not None:
                return self._model

            def load() -> Any:
                if self.settings.cross_encoder_backend == "fastembed":
                    from fastembed.rerank.cross_encoder import TextCrossEncoder

                    return TextCrossEncoder(
                        model_name=self.settings.cross_encoder_model,
                        cache_dir=str(self.settings.model_cache_dir / "fastembed"),
                        threads=self.settings.embedding_threads,
                    )
                from sentence_transformers import CrossEncoder

                try:
                    return CrossEncoder(
                        self.settings.cross_encoder_model,
                        backend=self.settings.cross_encoder_backend,
                        device=self.settings.cross_encoder_device,
                    )
                except TypeError:
                    return CrossEncoder(
                        self.settings.cross_encoder_model,
                        device=self.settings.cross_encoder_device,
                    )

            try:
                self._model = await asyncio.to_thread(load)
            except Exception as exc:
                logger.exception("Cross-Encoder model loading failed")
                raise RetrievalError(f"Cross-Encoder model loading failed: {exc}") from exc
            return self._model

    async def rerank(self, query: str, items: list[RetrievedItem], top_k: int) -> list[RetrievedItem]:
        if not items:
            return []
        if not self.settings.enable_neural_reranker:
            return _heuristic_rerank(query, items, top_k)
        try:
            model = await self._get_model()
        except RetrievalError as exc:
            logger.warning("Neural reranker unavailable; using deterministic reranker: %s", exc)
            return _heuristic_rerank(query, items, top_k)
        documents = [item.text[:4_000] for item in items]

        def score_fastembed() -> list[float]:
            if self.settings.cross_encoder_backend == "fastembed":
                return [float(value) for value in model.rerank(query, documents, batch_size=32)]
            pairs = list(zip([query] * len(documents), documents, strict=True))
            values = model.predict(pairs, batch_size=32, show_progress_bar=False)
            return [float(value.item() if hasattr(value, "item") else value) for value in values]

        try:
            scores = await asyncio.to_thread(score_fastembed)
        except Exception as exc:
            logger.warning("Neural reranking failed; using deterministic reranker: %s", exc)
            return _heuristic_rerank(query, items, top_k)
        for item, score in zip(items, scores, strict=True):
            item.rerank_score = score
        return sorted(items, key=lambda item: item.rerank_score, reverse=True)[:top_k]


class AgenticRetriever:
    def __init__(self, settings: Settings, llm: LocalModelProvider, store: Neo4jGraphStore) -> None:
        self.settings = settings
        self.llm = llm
        self.store = store
        self.reranker = CrossEncoderReranker(settings)

    async def warmup(self) -> None:
        await self.reranker.warmup()

    async def retrieve(
        self,
        query: str,
        *,
        graph_hops: int = 2,
        source_paths: Sequence[str] = (),
    ) -> list[RetrievedItem]:
        return await self.retrieve_many(
            [query],
            graph_hops=graph_hops,
            source_paths=source_paths,
        )

    async def retrieve_many(
        self,
        queries: Sequence[str],
        *,
        graph_hops: int = 2,
        likely_entities: Sequence[str] = (),
        source_paths: Sequence[str] = (),
    ) -> list[RetrievedItem]:
        base_queries = list(dict.fromkeys(query.strip() for query in queries if query.strip()))
        if not base_queries:
            return []
        if self.settings.fast_mode:
            expanded: list[str] = []
            for query in base_queries:
                expanded.extend(expand_query(query, self.settings.query_expansion_limit))
            unique_queries = list(dict.fromkeys(expanded))[: self.settings.query_expansion_limit]
            return await self._retrieve_fast(
                unique_queries,
                graph_hops=graph_hops,
                source_paths=source_paths,
            )
        return await self._retrieve_full(
            base_queries,
            graph_hops=graph_hops,
            likely_entities=likely_entities,
            source_paths=source_paths,
        )

    async def _retrieve_fast(
        self,
        queries: list[str],
        *,
        graph_hops: int,
        source_paths: Sequence[str],
    ) -> list[RetrievedItem]:
        original_query = queries[0]
        phrase_groups = policy_anchor_phrase_groups(original_query)
        embeddings = await self.llm.embed_queries(queries)
        keyword_tasks = [self.store.keyword_search(query, self.settings.keyword_top_k, list(source_paths)) for query in queries]
        vector_tasks = [
            self.store.vector_search(embedding, self.settings.vector_top_k, list(source_paths))
            for embedding in embeddings
        ]
        # Fetch a wider exact-phrase pool, then rank rule-bearing chunks above
        # tables of contents and headings. This avoids a TOC entry hiding the
        # actual controlling policy paragraph.
        phrase_tasks = [
            self.store.exact_phrase_search(
                group,
                max(30, self.settings.keyword_top_k),
                list(source_paths),
            )
            for group in phrase_groups
        ]
        results = await asyncio.gather(*keyword_tasks, *vector_tasks, *phrase_tasks)
        keyword_end = len(queries)
        vector_end = keyword_end + len(queries)
        keyword_lists = [list(items) for items in results[:keyword_end]]
        vector_lists = [list(items) for items in results[keyword_end:vector_end]]
        anchor_groups = [
            _rank_policy_anchor_items(list(items), phrase_group)[:4]
            for phrase_group, items in zip(phrase_groups, results[vector_end:], strict=True)
            if items
        ]
        ranked_lists: list[Sequence[RetrievedItem]] = [*keyword_lists, *vector_lists, *anchor_groups]
        fused = reciprocal_rank_fusion(ranked_lists, rrf_k=self.settings.rrf_k)

        seed_ids = [item.chunk_id for item in fused[: self.settings.graph_seed_k]]
        for group in anchor_groups:
            seed_ids.extend(item.chunk_id for item in group[:1])
        seeds = list(dict.fromkeys(seed_ids))[: max(self.settings.graph_seed_k, 8)]
        graph_items = await self.store.graph_expand(
            seeds,
            self.settings.graph_expand_k,
            max_hops=graph_hops,
            source_paths=list(source_paths),
        )
        if graph_items:
            fused = reciprocal_rank_fusion([fused, graph_items], rrf_k=self.settings.rrf_k)

        # RRF can dilute a highly relevant result that appears only in one specialized query.
        # Add domain anchors before reranking, then choose a coverage-aware final set.
        candidate_items = list(fused[: self.settings.rerank_candidate_k])
        for group in anchor_groups:
            candidate_items.extend(group)
        candidates = _deduplicate_items(candidate_items)
        reranked = await self.reranker.rerank(queries[0], candidates, len(candidates))
        return _select_with_domain_coverage(
            reranked,
            anchor_groups,
            self.settings.rerank_top_k,
        )

    async def _retrieve_full(
        self,
        queries: list[str],
        *,
        graph_hops: int,
        likely_entities: Sequence[str],
        source_paths: Sequence[str],
    ) -> list[RetrievedItem]:
        query_results, entity_items = await asyncio.gather(
            asyncio.gather(
                *(
                    self._retrieve_with_hyde(
                        query,
                        graph_hops=graph_hops,
                        source_paths=source_paths,
                    )
                    for query in queries
                )
            ),
            self.store.entity_search(
                list(likely_entities),
                self.settings.graph_expand_k,
                list(source_paths),
            ),
        )
        ranked_lists: list[Sequence[RetrievedItem]] = list(query_results)
        if entity_items:
            ranked_lists.append(entity_items)
        fused = reciprocal_rank_fusion(ranked_lists, rrf_k=self.settings.rrf_k)
        return await self.reranker.rerank(
            queries[0],
            fused[: self.settings.rerank_candidate_k],
            self.settings.rerank_top_k,
        )

    async def _retrieve_with_hyde(
        self,
        query: str,
        *,
        graph_hops: int,
        source_paths: Sequence[str],
    ) -> list[RetrievedItem]:
        hypothetical = await self._hyde(query) if self.settings.enable_hyde else query
        embedding, keyword_items = await asyncio.gather(
            self.llm.embed_queries([hypothetical]),
            self.store.keyword_search(query, self.settings.keyword_top_k, list(source_paths)),
        )
        vector_items = await self.store.vector_search(
            embedding[0],
            self.settings.vector_top_k,
            list(source_paths),
        )
        fused = reciprocal_rank_fusion([vector_items, keyword_items], rrf_k=self.settings.rrf_k)
        graph_items = await self.store.graph_expand(
            [item.chunk_id for item in fused[: self.settings.graph_seed_k]],
            self.settings.graph_expand_k,
            max_hops=graph_hops,
            source_paths=list(source_paths),
        )
        return reciprocal_rank_fusion([fused, graph_items], rrf_k=self.settings.rrf_k)

    async def _hyde(self, query: str) -> str:
        prompt = (
            "Write one concise hypothetical answer passage for retrieval. Preserve exact policy, section, "
            "component, table, API, and dependency terminology. Do not invent facts.\n\n"
            f"Query: {query}"
        )
        return await self.llm.generate(
            prompt,
            instructions="Generate retrieval text only.",
            max_output_tokens=300,
        )


def policy_anchor_phrase_groups(query: str) -> list[list[str]]:
    """Return exact-phrase groups for material policy topics in the question.

    Each group represents one part of a multi-part question. At least one result from
    each non-empty group is preserved after reranking, preventing broad lexical results
    from hiding the controlling policy section.
    """
    lowered = re.sub(r"\s+", " ", query).strip().casefold()
    groups: list[list[str]] = []
    if re.search(r"\b(freelance|freelancing|outside employment|outside job|side job|second job|moonlight|moonlighting|external work)\b", lowered):
        groups.append([
            "outside employment",
            "may not hold any type of outside employment",
            "income or material gain",
        ])
    if re.search(r"\b(source code|personal account|personal email|confidential|trade secret|nda|non.?disclosure)\b", lowered):
        groups.append([
            "source code",
            "computer program",
            "confidential business information",
            "non disclosure agreement",
        ])
    if re.search(r"\b(casual leave|optional holiday|optional leave|unused leave|carry forward)\b", lowered):
        groups.append([
            "unused 5 leaves",
            "two optional leaves",
            "approval of your respective team heads",
        ])
    if re.search(r"\b(unattended|unlocked|logged in|logged on|screen ?saver|hotel room)\b", lowered):
        groups.append([
            "avoid leaving your laptop desktop unattended and logged on",
            "always shut down log off",
            "password protected screensaver",
            "security cable",
        ])
    if re.search(r"\b(family member|family members|family|friend|friends|loan|used by others)\b", lowered):
        groups.append([
            "do not loan your laptop",
            "allow it to be used by others",
            "family and friends",
        ])
    if re.search(r"\b(remote control|remote-control|pc anywhere|unauthorized software|hacking tools?|password crackers?|network sniffers?)\b", lowered):
        groups.append([
            "unauthorized software",
            "remote controlled",
            "explicitly forbidden",
            "pre-authorized by management",
        ])
    if re.search(r"\b(virus warning|virus infection|infected|suspect a virus|malware)\b", lowered):
        groups.append([
            "respond immediately to any virus warning",
            "contacting the internal IT manager",
            "do not forward any files or upload data",
        ])
    if re.search(r"\b(backups?|three weeks|daily|weekly)\b", lowered):
        groups.append([
            "take your own backups",
            "ideally daily but weekly at least",
        ])
    if re.search(r"\b(lost|stolen|missing laptop|missing device)\b", lowered):
        groups.append([
            "notify the police immediately",
            "inform the IT manager",
            "within hours not days",
        ])
    if re.search(r"\b(ephi|pii|encrypt|encryption|password protection)\b", lowered):
        groups.append([
            "encrypted with password protection",
            "ephi and pii data safe",
        ])
    if re.search(r"\b(exception|exceptions|approve|approval|pre-authorized)\b", lowered):
        groups.append([
            "approve any exceptions to this policy in advance",
            "information security head",
            "pre-authorized by management",
        ])
    if re.search(r"\b(discipline|disciplinary|consequences?|termination|warning|suspension|legal action)\b", lowered):
        groups.append([
            "disciplinary action",
            "termination of employment",
            "legal action",
            "civil or criminal penalties",
            "withdrawal of access",
            "progressive discipline",
        ])
    if re.search(r"\b(redis|postgres|database commit|http 503|split commit|idempotenc)\b", lowered):
        groups.append([
            "known failure chain",
            "split commit",
            "idempotency marker",
        ])
    return groups


def policy_domain_queries(query: str) -> list[str]:
    """Return high-precision domain searches for common multi-part policy and incident questions."""
    lowered = re.sub(r"\s+", " ", query).strip().casefold()
    domain_queries: list[str] = []
    if re.search(r"\b(freelance|freelancing|outside employment|outside job|side job|second job|moonlight|moonlighting|external work)\b", lowered):
        domain_queries.append(
            "outside employment employee may not hold any type outside employment income material gain services rendered"
        )
    if re.search(r"\b(source code|personal account|personal email|confidential|trade secret|nda|non.?disclosure)\b", lowered):
        domain_queries.append(
            "non disclosure agreement confidential business information computer program codes source code unauthorized person personal email"
        )
    if re.search(r"\b(casual leave|optional holiday|optional leave|unused leave|carry forward)\b", lowered):
        domain_queries.append("casual leave carry forward unused five two optional leaves team head approval HR")
    if re.search(r"\b(unattended|unlocked|logged in|logged on|screen ?saver|hotel room)\b", lowered):
        domain_queries.append(
            "laptop unattended logged on shut down log off password protected screensaver security cable hotel room"
        )
    if re.search(r"\b(family member|family members|family|friend|friends|loan|used by others)\b", lowered):
        domain_queries.append(
            "corporate laptop official use authorized employees do not loan allow used by family friends"
        )
    if re.search(r"\b(remote control|remote-control|pc anywhere|unauthorized software|hacking tools?|password crackers?|network sniffers?)\b", lowered):
        domain_queries.append(
            "unauthorized software remote controlled explicitly forbidden pre-authorized management legitimate business"
        )
    if re.search(r"\b(virus warning|virus infection|infected|suspect a virus|malware)\b", lowered):
        domain_queries.append(
            "virus warning contact Internal IT Manager do not forward files upload data suspected infected"
        )
    if re.search(r"\b(backups?|three weeks|daily|weekly)\b", lowered):
        domain_queries.append("laptop backups regular basis ideally daily weekly at least")
    if re.search(r"\b(lost|stolen|missing laptop|missing device)\b", lowered):
        domain_queries.append("lost stolen notify Police immediately inform IT Manager within hours not days")
    if re.search(r"\b(ephi|pii|encrypt|encryption|password protection)\b", lowered):
        domain_queries.append("laptop encrypted password protection organization ePHI PII data safe")
    if re.search(r"\b(exception|exceptions|approve|approval|pre-authorized)\b", lowered):
        domain_queries.append("policy exception advance approval Information Security Head pre-authorized management")
    if re.search(r"\b(discipline|disciplinary|consequences?|termination|warning|suspension|legal action)\b", lowered):
        domain_queries.append(
            "disciplinary action termination employment legal action civil criminal penalties withdrawal access progressive discipline warning suspension internet privileges"
        )
    if re.search(r"\b(redis|postgres|database commit|http 503|split commit|idempotenc)\b", lowered):
        domain_queries.append("failure chain diagnosis mitigation idempotency database commit redis")
    return domain_queries


def expand_query(query: str, limit: int) -> list[str]:
    """Create clause and policy-domain searches without an LLM call."""
    cleaned = re.sub(r"\s+", " ", query).strip()
    output = [cleaned, *policy_domain_queries(cleaned)]
    clauses = re.split(
        r"[;?]|,\s+|\s+(?:and|but|while|also|then)\s+",
        cleaned,
        flags=re.IGNORECASE,
    )
    for clause in clauses:
        clause = clause.strip(" .:-")
        if len(clause) < 12 or clause.casefold() == cleaned.casefold():
            continue
        output.append(clause)
    return list(dict.fromkeys(output))[:limit]


def _deduplicate_items(items: Sequence[RetrievedItem]) -> list[RetrievedItem]:
    output: list[RetrievedItem] = []
    seen: set[str] = set()
    for item in items:
        if item.chunk_id in seen:
            continue
        seen.add(item.chunk_id)
        output.append(item)
    return output


def _select_with_domain_coverage(
    reranked: list[RetrievedItem],
    anchor_groups: list[list[RetrievedItem]],
    top_k: int,
) -> list[RetrievedItem]:
    """Keep strong policy-section coverage while preserving neural reranker quality."""
    if not reranked or top_k <= 0:
        return []
    by_id = {item.chunk_id: item for item in reranked}
    selected: list[RetrievedItem] = []
    selected_ids: set[str] = set()
    max_coverage_slots = min(len(anchor_groups), max(1, top_k // 2))

    for group in anchor_groups[:max_coverage_slots]:
        available = [by_id[item.chunk_id] for item in group if item.chunk_id in by_id]
        if not available:
            continue
        best = max(
            available,
            key=lambda item: (_policy_anchor_score(item, ()), item.rerank_score),
        )
        if best.chunk_id not in selected_ids:
            selected.append(best)
            selected_ids.add(best.chunk_id)

    for item in reranked:
        if len(selected) >= top_k:
            break
        if item.chunk_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item.chunk_id)
    return selected



def _rank_policy_anchor_items(
    items: list[RetrievedItem],
    phrases: Sequence[str],
) -> list[RetrievedItem]:
    """Rank exact-policy hits by rule content, not only Lucene frequency.

    A table of contents may contain the same section heading as the real policy.
    The real policy paragraph is preferred when it contains explicit control
    language such as "may not", "must", or a stated disciplinary consequence.
    """
    unique = _deduplicate_items(items)
    return sorted(
        unique,
        key=lambda item: (
            _policy_anchor_score(item, phrases),
            item.keyword_score,
            item.vector_score,
        ),
        reverse=True,
    )


def _policy_anchor_score(item: RetrievedItem, phrases: Sequence[str]) -> float:
    text = re.sub(r"\s+", " ", item.text).casefold()
    title = re.sub(r"\s+", " ", item.title).casefold()
    score = 0.0

    # Reward exact phrase coverage. Longer phrases are more discriminative.
    for phrase in phrases:
        normalized = re.sub(r"\s+", " ", phrase).strip().casefold()
        if not normalized:
            continue
        if normalized in text:
            score += 8.0 + min(8.0, len(normalized) / 12.0)
        if normalized in title:
            score += 2.0

    # Strongly reward executable policy language and stated consequences.
    control_patterns = {
        "may not": 16.0,
        "must not": 16.0,
        "shall not": 16.0,
        "not permitted": 15.0,
        "prohibited": 15.0,
        "should never": 13.0,
        "is required": 8.0,
        "are required": 8.0,
        "disciplinary action": 10.0,
        "termination of employment": 10.0,
        "legal action": 8.0,
        "revoking": 6.0,
    }
    for pattern, weight in control_patterns.items():
        if pattern in text:
            score += weight

    # A TOC proves a section exists but is not the controlling rule itself.
    if "table of contents" in text:
        score -= 35.0
    if re.search(r"\.{4,}\s*\d", text):
        score -= 8.0

    # Very short chunks are often headings rather than usable policy evidence.
    if len(text) < 140:
        score -= 5.0
    return score

def _heuristic_rerank(query: str, items: list[RetrievedItem], top_k: int) -> list[RetrievedItem]:
    query_terms = _terms(query)
    phrase = query.casefold().strip()
    max_keyword = max((item.keyword_score for item in items), default=1.0) or 1.0
    for item in items:
        title_terms = _terms(item.title)
        text_terms = _terms(item.text)
        overlap = len(query_terms & text_terms) / max(1, len(query_terms))
        title_overlap = len(query_terms & title_terms) / max(1, len(query_terms))
        exact = 1.0 if phrase and phrase in item.text.casefold() else 0.0
        keyword = math.log1p(max(item.keyword_score, 0.0)) / math.log1p(max_keyword)
        vector = max(0.0, item.vector_score)
        item.rerank_score = (
            0.42 * overlap
            + 0.20 * title_overlap
            + 0.08 * exact
            + 0.15 * keyword
            + 0.10 * vector
            + 0.05 * item.fusion_score
        )
    return sorted(items, key=lambda item: item.rerank_score, reverse=True)[:top_k]


def _terms(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _WORD.findall(text)
        if len(token) > 1 and token.casefold() not in _STOPWORDS
    }


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievedItem]],
    *,
    rrf_k: int = 60,
) -> list[RetrievedItem]:
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    merged: dict[str, RetrievedItem] = {}
    scores: defaultdict[str, float] = defaultdict(float)
    paths: defaultdict[str, set[str]] = defaultdict(set)

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item.chunk_id] += 1.0 / (rrf_k + rank)
            paths[item.chunk_id].update(item.retrieval_paths)
            if item.chunk_id not in merged:
                merged[item.chunk_id] = item.model_copy(deep=True)
            else:
                existing = merged[item.chunk_id]
                existing.vector_score = max(existing.vector_score, item.vector_score)
                existing.keyword_score = max(existing.keyword_score, item.keyword_score)
                existing.graph_score = max(existing.graph_score, item.graph_score)
                existing.rerank_score = max(existing.rerank_score, item.rerank_score)
                existing.entities = sorted(set(existing.entities + item.entities), key=str.casefold)

    output: list[RetrievedItem] = []
    for chunk_id, item in merged.items():
        item.fusion_score = scores[chunk_id]
        item.retrieval_paths = sorted(paths[chunk_id])
        output.append(item)
    return sorted(output, key=lambda item: item.fusion_score, reverse=True)
