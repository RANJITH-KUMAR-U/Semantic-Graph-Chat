"""
LangGraph state definition — the single source of truth passed
between every node in the compiled StateGraph.

See PRD section 6.3 "State Conflict Management".

Design notes:
  - `nodes` is the in-graph memory store: node_id → node data dict.
    Each node data dict holds the isolated message history for that topic.
  - `global_summary` is a compact, asynchronously refreshed digest of
    activity across *all* nodes — injected into every generation prompt
    so nodes stay coherent without seeing each other's raw messages.
  - `routing_decision` carries the structured output from the router so
    downstream graph nodes can act on it without re-invoking the LLM.
  - `force_node_id` lets the frontend pin the next message to a specific
    node (manual override, PRD section 5 "Should Have").
"""
from typing import Any, Optional, TypedDict


class NodeData(TypedDict, total=False):
    """
    Per-topic data stored inside `GraphState.nodes`.

    Kept as a plain TypedDict so it serialises cleanly to/from JSON
    by the LangGraph checkpointer.

    Hierarchical sub-topic fields:
        parent_node_id — node_id of parent topic (None for root-level topics).
        depth          — 0 for root topics, 1 for sub-topics (max depth = 2).
        node_summary   — compact LLM-generated summary of this node's content,
                         injected into sub-node context as parent awareness.

    In-node bounded memory fields:
        local_summary      — LLM-compressed digest of archived messages.
                             Prepended to the live context window instead of
                             sending all historical messages verbatim.
        archived_messages  — Messages beyond the rolling live window.
                             Kept in state for search/export but NEVER sent
                             to the LLM in the generation context.
    """

    title: str
    messages: list[dict[str, str]]  # [{"role": "user"|"assistant", "content": "..."}]
    turn_count: int
    created_at: str   # ISO-8601 UTC string
    last_active_at: str
    # Sub-topic hierarchy
    parent_node_id: Optional[str]   # None for root-level topics
    depth: int                       # 0 = root topic, 1 = sub-topic
    node_summary: str                # compact summary for parent context injection
    # Feature Round 3: relatedness graph
    related_node_ids: list           # node_ids with 0.35–0.85 similarity ("related", not duplicate)
    possible_duplicate_of: Optional[str]  # node_id with >0.85 similarity (existing merge detection)
    # In-node bounded memory (rolling window compression)
    local_summary: str               # compressed digest of archived messages
    archived_messages: list          # messages beyond the rolling window (never sent to LLM)
    # Document upload: chunks ingested from uploaded files (RAG retrieval)
    document_chunks: list            # list[DocumentChunk] — indexed file content for retrieval


class GraphState(TypedDict):
    """
    Full state object threaded through the LangGraph StateGraph.

    Every field must be JSON-serialisable (for the MemorySaver /
    future Postgres checkpointer). No datetime objects — use ISO strings.
    """

    # ── Session identity ────────────────────────────────────────────────
    session_id: str

    # ── Node registry (the in-graph memory store) ───────────────────────
    # node_id → NodeData; each entry is its own isolated topic thread.
    nodes: dict[str, NodeData]

    # ── Routing ─────────────────────────────────────────────────────────
    active_node_id: str                # which node is currently active
    current_input: str                 # the raw user message for this turn
    routing_decision: dict[str, Any]   # output of the router LLM call

    # ── Optional manual override (PRD "Should Have") ────────────────────
    force_node_id: Optional[str]       # non-None → skip router for this turn

    # ── Global context ──────────────────────────────────────────────────
    global_summary: str                # compact cross-node digest (async refresh)
    turn_count: int                    # total turns across the whole session

    # ── Last assistant response (passed through to WS for streaming) ────
    last_response: str

    # ── Feature Round 3: Routing decision log ───────────────────────────
    # Append-only list of routing events; never cleared within a session.
    routing_log: list                  # list[dict] — see _persist_turn for schema

    # ── Token accounting ────────────────────────────────────────────────
    session_tokens_used: int
    session_baseline_tokens: int
