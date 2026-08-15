"""
Lightweight text similarity service.

Uses normalised Jaccard overlap on meaningful word sets — zero external
deps, deterministic, and fast for small node counts (<20 nodes).

Two threshold bands:
  > DUPLICATE_THRESHOLD  (0.85) → possible_duplicate (existing merge detection)
  > RELATED_THRESHOLD    (0.35) → related (new, Feature Round 3)
  < RELATED_THRESHOLD           → unrelated

Why word-overlap instead of embeddings?
  - No embedding API call = no latency cost on every turn.
  - For topic-level similarity (short titles + summaries), overlapping
    domain vocabulary is a reliable signal.
  - Upgrading to vector embeddings later only requires swapping this module.
"""
from __future__ import annotations

import re

RELATED_THRESHOLD = 0.35
DUPLICATE_THRESHOLD = 0.85

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "to", "of", "in", "and", "or", "for",
    "be", "by", "at", "as", "it", "on", "up", "do", "if", "so",
    "no", "we", "me", "my", "he", "she", "they", "you", "us",
    "about", "how", "what", "why", "when", "where", "which", "who",
    "can", "does", "did", "was", "are", "has", "have", "had",
    "will", "with", "from", "also", "into", "that", "this", "than",
    "explain", "tell", "describe", "give", "show", "help", "please",
    "introduction", "overview", "topic", "question", "answer",
})


def _tokenise(text: str) -> frozenset[str]:
    """Lower-case, strip punctuation, remove stop words and short tokens."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return frozenset(t for t in tokens if len(t) >= 4 and t not in _STOP_WORDS)


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Return the Jaccard similarity between two text strings.

    Result is in [0, 1] where 1 = identical vocabulary, 0 = no overlap.
    Both empty inputs → 0.0 (not 1.0).
    """
    a = _tokenise(text_a)
    b = _tokenise(text_b)
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def node_text(node_data: dict) -> str:
    """Build a representative text blob for a node (title + summary)."""
    parts = [node_data.get("title", "")]
    summary = node_data.get("node_summary", "") or ""
    if summary:
        parts.append(summary)
    # Include first user message for richer signal
    messages = node_data.get("messages", [])
    if messages:
        parts.append(messages[0].get("content", "")[:200])
    return " ".join(filter(None, parts))


def compute_all_relations(nodes: dict) -> dict[str, dict]:
    """
    Compute pairwise similarity between all nodes and return per-node
    update dicts with `related_node_ids` and `possible_duplicate_of`.

    O(n²) — fine for <20 nodes per session.

    Returns:
        { node_id: { "related_node_ids": [...], "possible_duplicate_of": ... } }
    """
    node_ids = list(nodes.keys())
    # Pre-compute text blobs
    texts: dict[str, str] = {nid: node_text(nodes[nid]) for nid in node_ids}

    updates: dict[str, dict] = {
        nid: {"related_node_ids": [], "possible_duplicate_of": None}
        for nid in node_ids
    }

    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            a_id, b_id = node_ids[i], node_ids[j]
            sim = jaccard_similarity(texts[a_id], texts[b_id])

            if sim >= DUPLICATE_THRESHOLD:
                # Bidirectional duplicate signal (lower-ID node flags the other)
                if not updates[a_id]["possible_duplicate_of"]:
                    updates[a_id]["possible_duplicate_of"] = b_id
                if not updates[b_id]["possible_duplicate_of"]:
                    updates[b_id]["possible_duplicate_of"] = a_id
            elif sim >= RELATED_THRESHOLD:
                updates[a_id]["related_node_ids"].append(b_id)
                updates[b_id]["related_node_ids"].append(a_id)

    return updates


def check_node_consistency(node_data: dict) -> tuple[bool, float, str]:
    """
    Check if a node's stored messages are semantically consistent with its title.

    Returns:
        (is_consistent, similarity_score, detail_reason)
    """
    import logging
    logger = logging.getLogger(__name__)

    title = node_data.get("title", "")
    messages = node_data.get("messages", [])

    if not messages:
        return True, 1.0, "Empty node (no messages yet)"

    # Extract user prompts from messages
    user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return True, 1.0, "No user messages to evaluate"

    combined_user_text = " ".join(user_msgs)
    sim = jaccard_similarity(title, combined_user_text)

    # If first user message has zero token overlap with title, check if title is descriptive
    first_msg = user_msgs[0]
    first_sim = jaccard_similarity(title, first_msg)

    # Threshold for drift warning
    # Note: Short titles (2-4 words) vs detailed user messages can have modest Jaccard values,
    # but 0.0 or <0.05 indicates complete drift (e.g. title="Pharmacodynamics", msg="clinical trials")
    is_consistent = first_sim >= 0.05 or sim >= 0.05

    if not is_consistent:
        logger.warning(
            "[Semantic Drift Warning] Node title %r conflicts with user message %r (sim=%.3f)",
            title, first_msg[:100], first_sim
        )
        return False, first_sim, f"Title {title!r} does not match message content {first_msg[:60]!r}"

    return True, max(sim, first_sim), "Consistent"
