from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any, Literal, TypeVar

import httpx
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from .config import Settings

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class TransientLLMError(LLMError):
    """Retryable local-model failure."""


class OllamaTimeoutError(LLMError):
    """Raised when local generation exceeds the configured response timeout."""


class LocalModelProvider:
    """Keyless Ollama generation plus fast local ONNX embeddings."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        timeout = httpx.Timeout(
            connect=30.0,
            read=settings.request_timeout_seconds,
            write=settings.request_timeout_seconds,
            pool=30.0,
        )
        self.client = httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=timeout)
        self._semaphore = asyncio.Semaphore(settings.llm_concurrency)
        self._embedding_model: Any | None = None
        self._embedding_lock = asyncio.Lock()

    async def close(self) -> None:
        await self.client.aclose()

    async def verify_connectivity(self, *, require_models: bool = True) -> dict[str, object]:
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise LLMError(
                f"Cannot connect to Ollama at {self.settings.ollama_base_url}. "
                "Start Ollama and verify `ollama list` works."
            ) from exc

        available = {
            str(model.get("name", ""))
            for model in payload.get("models", [])
            if isinstance(model, dict)
        }
        missing = sorted(model for model in self.settings.required_ollama_models if model not in available)
        if require_models and missing:
            commands = "\n".join(f"ollama pull {model}" for model in missing)
            raise LLMError(f"Required Ollama models are missing:\n{commands}")
        return {
            "ollama_url": self.settings.ollama_base_url,
            "available_models": sorted(available),
            "missing_models": missing,
        }

    async def warmup(self) -> None:
        """Load the text and embedding models before the first user request."""
        await asyncio.gather(self._warmup_generation(), self._get_embedding_model())

    async def _warmup_generation(self) -> None:
        payload: dict[str, object] = {
            "model": self.settings.generation_model,
            "prompt": "Reply with OK.",
            "system": "Be concise.",
            "stream": False,
            "keep_alive": self.settings.ollama_keep_alive,
            "options": {
                "temperature": 0,
                "num_predict": 2,
                "num_ctx": min(self.settings.ollama_num_ctx, 4_096),
            },
        }
        if self.settings.disable_model_thinking:
            payload["think"] = False
        try:
            await self._generate_request(payload)
        except Exception as exc:
            logger.warning("Ollama warmup failed; the first query will retry model loading: %s", exc)

    @retry(
        retry=retry_if_exception_type(TransientLLMError),
        wait=wait_random_exponential(min=0.25, max=3),
        stop=stop_after_attempt(2),
        reraise=True,
    )
    async def generate(
        self,
        prompt: str,
        *,
        instructions: str,
        model: str | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        payload: dict[str, object] = {
            "model": model or self.settings.generation_model,
            "prompt": prompt,
            "system": instructions,
            "stream": False,
            "keep_alive": self.settings.ollama_keep_alive,
            "options": {
                "temperature": 0.0,
                "num_predict": max_output_tokens or self.settings.max_output_tokens,
                "num_ctx": self.settings.ollama_num_ctx,
                "top_p": 0.9,
            },
        }
        if self.settings.disable_model_thinking:
            payload["think"] = False
        result = await self._generate_request(payload)
        text = str(result.get("response", "")).strip()
        if not text:
            raise TransientLLMError("Ollama returned an empty response")
        return text

    @retry(
        retry=retry_if_exception_type(TransientLLMError),
        wait=wait_random_exponential(min=0.25, max=3),
        stop=stop_after_attempt(2),
        reraise=True,
    )
    async def structured(
        self,
        prompt: str,
        *,
        instructions: str,
        schema: type[T],
        model: str | None = None,
        max_output_tokens: int | None = None,
    ) -> T:
        schema_json = schema.model_json_schema()
        current_prompt = prompt
        last_error: Exception | None = None
        for attempt in range(1, 3):
            payload: dict[str, object] = {
                "model": model or self.settings.reasoning_model,
                "prompt": (
                    f"{current_prompt}\n\nReturn only valid JSON matching this JSON Schema exactly:\n"
                    f"{json.dumps(schema_json, ensure_ascii=False)}"
                ),
                "system": instructions,
                "format": schema_json,
                "stream": False,
                "keep_alive": self.settings.ollama_keep_alive,
                "options": {
                    "temperature": 0,
                    "num_predict": max_output_tokens or self.settings.max_output_tokens,
                    "num_ctx": self.settings.ollama_num_ctx,
                },
            }
            if self.settings.disable_model_thinking:
                payload["think"] = False
            result = await self._generate_request(payload)
            text = str(result.get("response", "")).strip()
            try:
                return schema.model_validate_json(text)
            except (ValidationError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning("Structured output validation failed on attempt %d: %s", attempt, exc)
                current_prompt = (
                    f"{prompt}\n\nYour previous JSON failed validation with this error:\n{exc}\n"
                    "Correct the JSON. Do not add markdown fences or explanatory text."
                )
        raise LLMError(f"Local model could not produce valid {schema.__name__} JSON: {last_error}")

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts, kind="passage")

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts, kind="query")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Backward-compatible passage embedding method."""
        return await self.embed_passages(texts)

    async def embed_batched(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        selected_batch_size = batch_size or self.settings.embedding_batch_size
        output: list[list[float]] = []
        for start in range(0, len(texts), selected_batch_size):
            output.extend(await self.embed_passages(texts[start : start + selected_batch_size]))
        return output

    async def _embed(self, texts: list[str], *, kind: Literal["query", "passage"]) -> list[list[float]]:
        cleaned = [text.strip() for text in texts]
        if not cleaned or any(not text for text in cleaned):
            raise ValueError("Embedding inputs must be non-empty strings")
        model = await self._get_embedding_model()

        if self.settings.embedding_backend == "fastembed":
            prefixed = [f"{kind}: {text}" for text in cleaned]

            def encode_fastembed() -> list[list[float]]:
                vectors = model.embed(
                    prefixed,
                    batch_size=min(self.settings.embedding_batch_size, len(prefixed)),
                )
                return [vector.astype("float32").tolist() for vector in vectors]

            encoder = encode_fastembed
        else:

            def encode_sentence_transformers() -> list[list[float]]:
                vectors = model.encode(
                    cleaned,
                    batch_size=min(self.settings.embedding_batch_size, len(cleaned)),
                    convert_to_numpy=True,
                    normalize_embeddings=self.settings.normalize_embeddings,
                    show_progress_bar=False,
                )
                return [vector.astype("float32").tolist() for vector in vectors]

            encoder = encode_sentence_transformers

        try:
            embeddings = await asyncio.to_thread(encoder)
        except Exception as exc:
            logger.exception("Local embedding inference failed")
            raise LLMError(f"Local embedding inference failed: {exc}") from exc
        if len(embeddings) != len(cleaned):
            raise LLMError("Embedding count does not match input count")
        if embeddings and len(embeddings[0]) != self.settings.embedding_dimensions:
            raise LLMError(
                f"Embedding model returned {len(embeddings[0])} dimensions, but "
                f"EMBEDDING_DIMENSIONS={self.settings.embedding_dimensions}."
            )
        return embeddings

    @retry(
        retry=retry_if_exception_type(TransientLLMError),
        wait=wait_random_exponential(min=0.25, max=3),
        stop=stop_after_attempt(2),
        reraise=True,
    )
    async def describe_image(self, path: Path, context_hint: str = "") -> str:
        if not self.settings.enable_vision:
            return f"Image file {path.name}; vision captioning is disabled."
        image_bytes = await asyncio.to_thread(path.read_bytes)
        encoded = base64.b64encode(image_bytes).decode("ascii")
        prompt = (
            "Describe this enterprise architecture image or chart for retrieval. "
            "Identify visible components, arrows, dependencies, labels, protocols, data stores, "
            "failure paths, legends, axes, and quantitative values. Be literal and do not infer hidden facts."
        )
        if context_hint:
            prompt += f"\nNearby document context:\n{context_hint[:3000]}"
        payload: dict[str, object] = {
            "model": self.settings.vision_model,
            "prompt": prompt,
            "images": [encoded],
            "stream": False,
            "keep_alive": self.settings.ollama_keep_alive,
            "options": {
                "temperature": 0,
                "num_predict": min(1_000, self.settings.max_output_tokens),
                "num_ctx": self.settings.ollama_num_ctx,
            },
        }
        if self.settings.disable_model_thinking:
            payload["think"] = False
        result = await self._generate_request(payload)
        description = str(result.get("response", "")).strip()
        if not description:
            raise TransientLLMError(f"Vision model returned no description for {path}")
        return description

    async def json_object(self, prompt: str, *, instructions: str) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.settings.reasoning_model,
            "prompt": prompt,
            "system": instructions,
            "format": "json",
            "stream": False,
            "keep_alive": self.settings.ollama_keep_alive,
            "options": {
                "temperature": 0,
                "num_predict": self.settings.max_output_tokens,
                "num_ctx": self.settings.ollama_num_ctx,
            },
        }
        if self.settings.disable_model_thinking:
            payload["think"] = False
        result = await self._generate_request(payload)
        text = str(result.get("response", "")).strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Expected JSON object, got invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise LLMError("Expected a JSON object")
        return value

    async def _get_embedding_model(self) -> Any:
        if self._embedding_model is not None:
            return self._embedding_model
        async with self._embedding_lock:
            if self._embedding_model is not None:
                return self._embedding_model

            def load() -> Any:
                cache_dir = str(self.settings.model_cache_dir / "fastembed")
                if self.settings.embedding_backend == "fastembed":
                    from fastembed import TextEmbedding

                    return TextEmbedding(
                        model_name=self.settings.embedding_model,
                        cache_dir=cache_dir,
                        threads=self.settings.embedding_threads,
                    )
                from sentence_transformers import SentenceTransformer

                return SentenceTransformer(
                    self.settings.embedding_model,
                    device=self.settings.embedding_device,
                    cache_folder=str(self.settings.model_cache_dir / "huggingface"),
                )

            try:
                self._embedding_model = await asyncio.to_thread(load)
            except Exception as exc:
                logger.exception("Embedding model loading failed")
                raise LLMError(
                    f"Could not load local embedding model {self.settings.embedding_model}: {exc}"
                ) from exc
        return self._embedding_model

    async def _generate_request(self, payload: dict[str, object]) -> dict[str, object]:
        try:
            async with self._semaphore:
                response = await self.client.post("/api/generate", json=payload)
            response.raise_for_status()
            result = response.json()
        except httpx.ReadTimeout as exc:
            raise OllamaTimeoutError(
                "Ollama did not finish within "
                f"{self.settings.request_timeout_seconds:.0f} seconds. "
                "The application will use an extractive evidence fallback."
            ) from exc
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise TransientLLMError(
                f"Could not connect to Ollama at {self.settings.ollama_base_url}. "
                "Confirm that the Ollama application is running."
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:1_000]
            if exc.response.status_code >= 500:
                raise TransientLLMError(f"Ollama server error: {detail}") from exc
            raise LLMError(f"Ollama request failed ({exc.response.status_code}): {detail}") from exc
        except json.JSONDecodeError as exc:
            raise TransientLLMError("Ollama returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise TransientLLMError("Ollama response was not a JSON object")
        return result
