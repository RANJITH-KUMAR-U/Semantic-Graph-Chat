"""
Vector embedding service via OpenRouter.

Uses the nvidia/nemotron-3-embed-1b:free model through OpenRouter's
OpenAI-compatible embeddings endpoint.  No new dependencies — the
existing `openai` SDK is reused with the same API key and base URL.

Design decisions:
  - LRU cache on text→embedding to avoid re-embedding identical
    chunks across retrieval calls within the same server process.
  - Pure-Python cosine similarity (no numpy needed).
  - Batch embedding: sends up to 32 texts per API call.
  - Graceful degradation: callers should catch EmbeddingError and
    fall back to Jaccard when the API is unreachable.

# SECURITY-TODO: Rate-limit embedding calls per session in production.
"""
from __future__ import annotations

import hashlib
import logging
import math
from collections import OrderedDict
from typing import Any

from openai import AsyncOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

MAX_BATCH_SIZE = 32          # OpenRouter batch limit
CACHE_MAX_SIZE = 2048        # max cached embeddings in memory
_EMBEDDING_DIM: int | None = None  # set after first successful call


class EmbeddingError(Exception):
    """Raised when the embedding API call fails."""


# ── Client ─────────────────────────────────────────────────────────────


_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    """Return an AsyncOpenAI client pointed at OpenRouter."""
    global _client
    if _client is None:
        api_key = settings.openrouter_api_key or "sk-or-v1-dummy"
        _client = AsyncOpenAI(
            api_key=api_key,
            base_url=settings.openrouter_base_url,
            default_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "Semantic Graph Chat",
            },
        )
    return _client


# ── In-memory embedding cache (LRU) ───────────────────────────────────

_cache: OrderedDict[str, list[float]] = OrderedDict()


def _cache_key(text: str) -> str:
    """Stable hash key for a text string."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _cache_get(text: str) -> list[float] | None:
    key = _cache_key(text)
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    return None


def _cache_put(text: str, embedding: list[float]) -> None:
    key = _cache_key(text)
    _cache[key] = embedding
    _cache.move_to_end(key)
    while len(_cache) > CACHE_MAX_SIZE:
        _cache.popitem(last=False)


# ── Public API ─────────────────────────────────────────────────────────


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using the configured embedding model.

    Returns a list of embedding vectors in the same order as the input.
    Uses cached embeddings where available; only calls the API for
    cache misses.

    Raises:
        EmbeddingError: If the API call fails for any batch.
    """
    if not texts:
        return []

    # Separate cached from uncached
    results: list[list[float] | None] = [None] * len(texts)
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []

    for i, text in enumerate(texts):
        cached = _cache_get(text)
        if cached is not None:
            results[i] = cached
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)

    if not uncached_texts:
        return results  # type: ignore[return-value]  — all are cached

    # Batch API calls
    client = _get_client()
    model = settings.embedding_model

    for batch_start in range(0, len(uncached_texts), MAX_BATCH_SIZE):
        batch = uncached_texts[batch_start : batch_start + MAX_BATCH_SIZE]
        batch_indices = uncached_indices[batch_start : batch_start + MAX_BATCH_SIZE]

        try:
            response = await client.embeddings.create(
                model=model,
                input=batch,
                encoding_format="float",
            )

            for j, emb_data in enumerate(response.data):
                vec = emb_data.embedding
                idx = batch_indices[j]
                results[idx] = vec
                _cache_put(uncached_texts[batch_start + j], vec)

            logger.debug(
                "Embedded %d texts via %s (dim=%d)",
                len(batch), model, len(response.data[0].embedding),
            )

        except Exception as exc:
            logger.error("Embedding API call failed: %s", exc)
            raise EmbeddingError(f"Embedding call failed: {exc}") from exc

    return results  # type: ignore[return-value]


async def embed_single(text: str) -> list[float]:
    """Convenience wrapper: embed a single text and return its vector."""
    vecs = await embed_texts([text])
    return vecs[0]


# ── Cosine similarity (pure Python — no numpy needed) ──────────────────


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns a float in [-1, 1].  For normalised embedding vectors
    this is typically in [0, 1].
    """
    if len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)
