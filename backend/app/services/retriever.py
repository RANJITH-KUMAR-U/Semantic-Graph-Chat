"""
Document chunk retriever for answer-time RAG.

Given a user query and a list of document chunks stored in a node,
returns the top-k most relevant chunks ranked by semantic similarity.

Primary method:  vector embeddings via OpenRouter (nvidia/nemotron-3-embed-1b).
Fallback method: Jaccard word-overlap (deterministic, zero-latency).

The embedding path catches paraphrased queries that share few exact
words with the source chunk — e.g. "ordered search tree structure"
correctly matches a chunk about "binary search trees".
"""
from __future__ import annotations

import logging
from typing import Any

from app.services.similarity import jaccard_similarity

logger = logging.getLogger(__name__)


async def retrieve_relevant_chunks(
    query: str,
    document_chunks: list[dict[str, Any]],
    top_k: int = 3,
    min_score: float = 0.20,
) -> list[dict[str, Any]]:
    """
    Rank document chunks by relevance to the query.

    Tries embedding-based cosine similarity first (semantic matching).
    Falls back to Jaccard word-overlap if the embedding API is
    unavailable.

    Args:
        query: The user's question / message.
        document_chunks: List of DocumentChunk dicts from NodeData.
        top_k: Maximum number of chunks to return.
        min_score: Minimum similarity score to include a chunk.

    Returns:
        List of the top-k most relevant chunks, each with an added
        'relevance_score' and 'retrieval_method' field, sorted by
        score descending.
    """
    if not document_chunks or not query.strip():
        return []

    # ── Try embedding-based retrieval first ────────────────────────────
    try:
        results = await _retrieve_with_embeddings(
            query, document_chunks, top_k, min_score
        )
        if results:
            return results
        # If embeddings returned no results above threshold, fall through
        # to Jaccard (which may catch exact-match cases embeddings scored low)
    except Exception as exc:
        logger.warning(
            "Embedding retrieval failed, falling back to Jaccard: %s", exc
        )

    # ── Fallback: Jaccard word-overlap ─────────────────────────────────
    return _retrieve_with_jaccard(query, document_chunks, top_k, min_score=0.02)


async def _retrieve_with_embeddings(
    query: str,
    document_chunks: list[dict[str, Any]],
    top_k: int,
    min_score: float,
) -> list[dict[str, Any]]:
    """Score chunks using vector embeddings (cosine similarity)."""
    from app.services.embedding_service import (
        embed_texts,
        cosine_similarity,
    )

    # Collect chunk contents
    contents: list[str] = []
    valid_indices: list[int] = []
    for i, chunk in enumerate(document_chunks):
        content = chunk.get("content", "")
        if content:
            contents.append(content)
            valid_indices.append(i)

    if not contents:
        return []

    # Embed query + all chunk contents in one batch
    all_texts = [query] + contents
    all_embeddings = await embed_texts(all_texts)

    query_vec = all_embeddings[0]
    chunk_vecs = all_embeddings[1:]

    # Score each chunk
    scored: list[tuple[float, int]] = []
    for j, chunk_vec in enumerate(chunk_vecs):
        score = cosine_similarity(query_vec, chunk_vec)
        if score >= min_score:
            scored.append((score, valid_indices[j]))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, idx in scored[:top_k]:
        result = dict(document_chunks[idx])
        result["relevance_score"] = round(score, 4)
        result["retrieval_method"] = "embedding"
        results.append(result)

    if results:
        logger.info(
            "Embedding retrieval: %d/%d chunks (top score=%.3f) for query: %s",
            len(results), len(document_chunks),
            results[0]["relevance_score"],
            query[:80],
        )

    return results


def _retrieve_with_jaccard(
    query: str,
    document_chunks: list[dict[str, Any]],
    top_k: int,
    min_score: float,
) -> list[dict[str, Any]]:
    """Fallback: score chunks using Jaccard word-overlap similarity."""
    scored: list[tuple[float, dict]] = []

    for chunk in document_chunks:
        content = chunk.get("content", "")
        if not content:
            continue

        score = jaccard_similarity(query, content)
        if score >= min_score:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, chunk in scored[:top_k]:
        result = dict(chunk)
        result["relevance_score"] = round(score, 4)
        result["retrieval_method"] = "jaccard_fallback"
        results.append(result)

    if results:
        logger.info(
            "Jaccard fallback retrieval: %d/%d chunks (top score=%.3f) for query: %s",
            len(results), len(document_chunks),
            results[0]["relevance_score"],
            query[:80],
        )

    return results

