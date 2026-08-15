"""
Document chunk retriever for answer-time RAG.

Given a user query and a list of document chunks stored in a node,
returns the top-k most relevant chunks using Jaccard similarity
(reusing the existing similarity.py infrastructure).

This is intentionally simple — no embedding model, no vector DB.
Upgrading to embeddings later only requires swapping the scoring
function below.
"""
from __future__ import annotations

import logging
from typing import Any

from app.services.similarity import jaccard_similarity

logger = logging.getLogger(__name__)


def retrieve_relevant_chunks(
    query: str,
    document_chunks: list[dict[str, Any]],
    top_k: int = 3,
    min_score: float = 0.02,
) -> list[dict[str, Any]]:
    """
    Rank document chunks by relevance to the query.

    Uses Jaccard similarity between the query and each chunk's content.

    Args:
        query: The user's question / message.
        document_chunks: List of DocumentChunk dicts from NodeData.
        top_k: Maximum number of chunks to return.
        min_score: Minimum similarity score to include a chunk.

    Returns:
        List of the top-k most relevant chunks, each with an added
        'relevance_score' field, sorted by score descending.
    """
    if not document_chunks or not query.strip():
        return []

    scored: list[tuple[float, dict]] = []

    for chunk in document_chunks:
        content = chunk.get("content", "")
        if not content:
            continue

        score = jaccard_similarity(query, content)
        if score >= min_score:
            scored.append((score, chunk))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, chunk in scored[:top_k]:
        result = dict(chunk)
        result["relevance_score"] = round(score, 4)
        results.append(result)

    if results:
        logger.info(
            "Retrieved %d/%d chunks (top score=%.3f) for query: %s",
            len(results), len(document_chunks),
            results[0]["relevance_score"],
            query[:80],
        )

    return results
