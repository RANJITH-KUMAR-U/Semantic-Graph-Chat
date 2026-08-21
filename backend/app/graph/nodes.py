"""
Topic Node lifecycle: creation, lookup, isolated context assembly,
and the LLM generation step.

Non-negotiable constraint (AGENTS.md):
  "Never let one Topic Node's LLM call see another node's messages.
   assemble_context() is the isolation boundary — do not widen it."

Hierarchy extension:
  Sub-topic nodes (depth=1) get the parent node's compact summary injected
  as system context — giving the LLM "big picture" awareness without
  loading the parent's full message history.

All node data lives inside GraphState["nodes"], so it travels with
the LangGraph checkpointer and needs no separate database call for
the hot path. The DB layer (db_models.py) is used only for REST
history reads and persistence after each turn.
"""
import logging
import uuid
from datetime import datetime, timezone

from app.graph.state import GraphState, NodeData
from app.services import llm_service
from app.services.text_cleaning import strip_reasoning

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────


def _utcnow() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _new_node_id() -> str:
    """Generate a short, collision-resistant node ID."""
    return f"node_{uuid.uuid4().hex[:8]}"


# ── Node creation (root) ───────────────────────────────────────────────


async def create_node_node(state: GraphState) -> dict:
    """
    LangGraph node: create a brand-new ROOT-level Topic Node (depth=0).

    Uses the first user message as the seed to generate a short title.

    Idempotency guard (PRD section 6.4): before creating, check if any
    existing node's title is semantically very close to the new title.
    """
    current_input: str = state["current_input"]
    existing_nodes: dict = state.get("nodes", {})

    title = await _generate_title(current_input)

    # ── Idempotency check ─────────────────────────────────────────────
    title_lower = title.lower()
    for node_id, node_data in existing_nodes.items():
        existing_title = node_data.get("title", "").lower()
        if _titles_overlap(title_lower, existing_title):
            logger.info(
                "Deduped node creation: %r ≈ %r → routing to %s",
                title, node_data["title"], node_id,
            )
            return {
                "active_node_id": node_id,
                "routing_decision": {
                    "decision": "route_existing",
                    "target_node_id": node_id,
                    "reasoning": f"Deduplicated: new title '{title}' matches existing '{node_data['title']}'.",
                },
            }

    # ── Create root node ──────────────────────────────────────────────
    node_id = _new_node_id()
    now = _utcnow()
    new_node: NodeData = {
        "title": title,
        "messages": [],
        "turn_count": 0,
        "created_at": now,
        "last_active_at": now,
        "parent_node_id": None,
        "depth": 0,
        "node_summary": "",
    }

    updated_nodes = {**existing_nodes, node_id: new_node}
    logger.info("Created ROOT topic node %r: %r", node_id, title)

    return {
        "nodes": updated_nodes,
        "active_node_id": node_id,
    }


# ── Sub-topic creation ─────────────────────────────────────────────────


async def create_subtopic_node(state: GraphState) -> dict:
    """
    LangGraph node: create a SUB-TOPIC node (depth=1) under an existing
    root topic.

    The parent_node_id is taken from routing_decision.target_node_id.
    The sub-topic title is derived from the current user message.

    Context assembly for sub-topics injects the parent node's summary
    as background context (see assemble_context).
    """
    current_input: str = state["current_input"]
    existing_nodes: dict = state.get("nodes", {})
    routing_decision: dict = state.get("routing_decision", {})
    parent_node_id: str | None = routing_decision.get("target_node_id")

    if not parent_node_id or parent_node_id not in existing_nodes:
        # Safety fallback: create a root node instead
        logger.warning(
            "create_subtopic: parent %r not found — falling back to create_node",
            parent_node_id,
        )
        return await create_node_node(state)

    parent_data = existing_nodes[parent_node_id]
    parent_title = parent_data.get("title", "General")

    # Generate sub-topic title (contextualised by parent)
    sub_title = await _generate_title(
        current_input,
        context_hint=f"This is a sub-topic of '{parent_title}'.",
    )

    # ── Idempotency: check if a matching sub-topic already exists ─────
    sub_lower = sub_title.lower()
    for nid, nd in existing_nodes.items():
        if nd.get("parent_node_id") == parent_node_id:
            if _titles_overlap(sub_lower, nd.get("title", "").lower()):
                logger.info(
                    "Deduped sub-topic %r ≈ %r → routing to %s",
                    sub_title, nd["title"], nid,
                )
                return {
                    "active_node_id": nid,
                    "routing_decision": {
                        "decision": "route_existing",
                        "target_node_id": nid,
                        "reasoning": f"Deduplicated sub-topic: '{sub_title}' ≈ '{nd['title']}'.",
                    },
                }

    # ── Create sub-topic node ─────────────────────────────────────────
    node_id = _new_node_id()
    now = _utcnow()
    new_node: NodeData = {
        "title": sub_title,
        "messages": [],
        "turn_count": 0,
        "created_at": now,
        "last_active_at": now,
        "parent_node_id": parent_node_id,
        "depth": 1,
        "node_summary": "",
    }

    updated_nodes = {**existing_nodes, node_id: new_node}
    logger.info(
        "Created SUB-TOPIC node %r: %r (parent=%r: %r)",
        node_id, sub_title, parent_node_id, parent_title,
    )

    return {
        "nodes": updated_nodes,
        "active_node_id": node_id,
    }


async def _generate_title(first_message: str, context_hint: str = "") -> str:
    """
    Ask the generator LLM to produce a short (≤5 word) topic title.
    Falls back to a truncated version of the first message on error.
    """
    extra = f" {context_hint}" if context_hint else ""
    try:
        raw_title = await llm_service.call_generator_once(
            messages=[{"role": "user", "content": first_message}],
            system_prompt=(
                f"Generate a short topic title (3-5 words, no punctuation) that "
                f"describes what the following message is about.{extra}\n"
                "CRITICAL: Output ONLY the 3-5 word title on a single line. Do NOT output any thinking, reasoning, or explanation."
            ),
        )
        cleaned = strip_reasoning(raw_title).strip()
        # Take the first non-empty line
        lines = [line.strip().strip("\"'#* `") for line in cleaned.splitlines() if line.strip()]
        title = lines[0] if lines else ""
        # If title still contains reasoning cues or error indicator, fallback
        if (
            not title
            or title.startswith("[")
            or "error" in title.lower()
            or "thinking process" in title.lower()
            or len(title.split()) > 10
        ):
            return _fallback_title(first_message)
        return title[:80]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Title generation failed: %s", exc)
        return _fallback_title(first_message)


def _fallback_title(text: str) -> str:
    """Truncate the first message to produce a simple title."""
    words = text.split()[:5]
    return " ".join(words).rstrip(",.!?") + ("…" if len(text.split()) > 5 else "")


def _titles_overlap(a: str, b: str) -> bool:
    """
    Idempotency overlap check: True if both titles share 2+ meaningful domain words.
    Prevents false-positive deduplication of distinct topics.
    """
    stop = {
        "a", "an", "the", "is", "to", "of", "in", "and", "or", "for",
        "explain", "about", "what", "how", "topic", "introduction", "overview",
        "basics", "explained", "details", "guide", "concept", "concepts",
        "sentence", "sentences", "summary", "notes", "definition", "meaning",
    }
    words_a = {w for w in a.lower().split() if w not in stop and len(w) > 3}
    words_b = {w for w in b.lower().split() if w not in stop and len(w) > 3}
    # Require at least 2 distinct domain words OR high Jaccard similarity
    overlap = words_a & words_b
    if not overlap:
        return False
    return len(overlap) >= 2 or (len(words_a) <= 2 and len(words_b) <= 2 and len(overlap) >= 1 and words_a == words_b)


# ── Context assembly (isolation boundary) ─────────────────────────────


def assemble_context(state: GraphState, node_id: str) -> dict:
    """
    Return the isolated context for `node_id`.

    **This is the isolation boundary** — only this node's messages plus
    the compact global_summary are returned. No other node's full messages
    are ever included (AGENTS.md non-negotiable constraint).

    Sub-topic enhancement:
        If the node has a parent (depth=1), the parent's node_summary is
        also injected as a lightweight background context, giving the LLM
        "big picture" awareness without loading the parent's full history.

    In-node bounded memory:
        If the node has a `local_summary` (set by the per-node background
        summarizer), it is returned alongside the live messages. The caller
        (websocket.py) prepends it to the system_prompt so the LLM sees:
            [local_summary] + [last N live messages]
        instead of the full unbounded history.

    Returns:
        {
            "messages":       list of {"role", "content"} dicts (live window only),
            "local_summary":  str — compressed digest of archived messages,
            "global_summary": str,
            "title":          str,
            "parent_title":   str | None,
            "parent_summary": str | None,
        }
    """
    nodes = state.get("nodes", {})
    if node_id not in nodes:
        raise KeyError(f"Node {node_id!r} not found in state. Active nodes: {list(nodes)}")

    node_data = nodes[node_id]
    parent_node_id = node_data.get("parent_node_id")
    parent_title: str | None = None
    parent_summary: str | None = None

    if parent_node_id and parent_node_id in nodes:
        parent_data = nodes[parent_node_id]
        parent_title = parent_data.get("title")
        parent_summary = parent_data.get("node_summary") or None

    return {
        "messages": list(node_data.get("messages", [])),
        "document_chunks": list(node_data.get("document_chunks", [])),
        "local_summary": node_data.get("local_summary") or "",
        "global_summary": state.get("global_summary", ""),
        "title": node_data.get("title", "Untitled"),
        "parent_title": parent_title,
        "parent_summary": parent_summary,
    }


# ── Generation step ────────────────────────────────────────────────────


async def generate_node(state: GraphState) -> dict:
    """
    LangGraph node: generate a response using only the active node's context.

    For sub-topic nodes, the parent's node_summary is injected as a
    lightweight background context block in the system prompt.
    """
    routing_decision = state.get("routing_decision", {})
    node_id = (
        routing_decision.get("target_node_id")
        or state.get("active_node_id")
    )

    if not node_id or node_id not in state.get("nodes", {}):
        logger.error("generate_node: no valid node_id. Routing decision: %s", routing_decision)
        return {"last_response": "[Error: no valid node found for generation.]"}

    context = assemble_context(state, node_id)
    current_input: str = state["current_input"]
    node_title: str = context["title"]
    global_summary: str = context["global_summary"]
    parent_title: str | None = context.get("parent_title")
    parent_summary: str | None = context.get("parent_summary")

    # Build system prompt — inject parent context for sub-topics
    if parent_title and parent_summary:
        system_prompt = (
            f"You are an expert assistant focused on '{node_title}', "
            f"which is a sub-topic of '{parent_title}'.\n\n"
            f"## Parent topic context ({parent_title}):\n{parent_summary}\n\n"
            "Use the parent context as background awareness only — "
            "stay focused on the specific sub-topic in this thread."
        )
    elif parent_title:
        system_prompt = (
            f"You are an expert assistant focused on '{node_title}', "
            f"a sub-topic of '{parent_title}'."
        )
    else:
        system_prompt = (
            f"You are an expert assistant focused on the topic: '{node_title}'.\n"
        )

    if global_summary:
        system_prompt += (
            f"\n\n## Cross-topic session summary (background only):\n{global_summary}"
        )

    messages_for_llm = context["messages"] + [{"role": "user", "content": current_input}]

    full_response_parts: list[str] = []
    async for token in llm_service.stream_generator(messages_for_llm, system_prompt):
        full_response_parts.append(token)
    full_response = "".join(full_response_parts)

    # Persist to the active node ONLY
    now = _utcnow()
    updated_nodes = dict(state["nodes"])
    node_data = dict(updated_nodes[node_id])
    node_messages = list(node_data.get("messages", []))
    node_messages.append({"role": "user", "content": current_input})
    node_messages.append({"role": "assistant", "content": full_response})
    node_data["messages"] = node_messages
    node_data["turn_count"] = node_data.get("turn_count", 0) + 1
    node_data["last_active_at"] = now
    updated_nodes[node_id] = node_data

    return {
        "nodes": updated_nodes,
        "active_node_id": node_id,
        "last_response": full_response,
        "turn_count": state.get("turn_count", 0) + 1,
    }
