"""
Cross-node reference detection — Feature Round 3, Feature 1.

Detects when a user message in one node semantically references the
content of a *different* existing node, without permanently merging
or polluting either node's history.

Design contract:
  - Only reads node data; never writes to state.
  - Returns at most ONE referenced node (the most similar one above threshold).
  - The caller injects the referenced node's summary as one-turn-only
    bounded context into the system prompt — it is never appended to
    either node's message list.
"""
from __future__ import annotations

import logging

from app.services.similarity import jaccard_similarity, node_text

logger = logging.getLogger(__name__)

# Minimum similarity for a reference to be reported
REFERENCE_THRESHOLD = 0.30
# Maximum number of sentences to pull from a referenced node's summary
_SNIPPET_MAX_CHARS = 400


def _node_snippet(node_data: dict) -> str:
    """
    Return a compact text snippet representing the referenced node:
      1. Use node_summary if available (already LLM-compressed).
      2. Fall back to the last assistant message (truncated).
      3. Fall back to the title only.
    """
    summary = node_data.get("node_summary", "")
    if summary:
        return summary[:_SNIPPET_MAX_CHARS]

    messages = node_data.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "")[:_SNIPPET_MAX_CHARS]

    return node_data.get("title", "")


def detect_cross_reference(
    user_message: str,
    active_node_id: str | None,
    nodes: dict,
) -> tuple[str | None, str | None]:
    """
    Check whether `user_message` is semantically referencing a node
    other than `active_node_id`.

    Args:
        user_message:   The raw user input for this turn.
        active_node_id: The currently active node (excluded from search).
        nodes:          Full `state["nodes"]` dict.

    Returns:
        (referenced_node_id, snippet) if a reference is detected, else (None, None).
        `snippet` is a short text extract from the referenced node suitable
        for injecting into the system prompt as bounded context.
    """
    # SECURITY-TODO: Sanitise user_message before using it in any LLM-visible context in production.
    if not nodes or len(nodes) < 2:
        return None, None

    best_id: str | None = None
    best_sim: float = REFERENCE_THRESHOLD  # minimum bar

    for node_id, node_data in nodes.items():
        if node_id == active_node_id:
            continue
        candidate_text = node_text(node_data)
        sim = jaccard_similarity(user_message, candidate_text)
        if sim > best_sim:
            best_sim = sim
            best_id = node_id

    if best_id is None:
        return None, None

    snippet = _node_snippet(nodes[best_id])
    logger.info(
        "Cross-reference detected: message references node %r (sim=%.2f)",
        best_id, best_sim,
    )
    return best_id, snippet
