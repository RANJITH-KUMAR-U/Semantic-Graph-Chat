"""
Builds and compiles the LangGraph StateGraph.

Graph topology (PRD section 3.3 + sub-topic extension):

    entry → router → ┬─(create_new)──────→ create_node     → generate → summarizer → END
                     ├─(create_subtopic)──→ create_subtopic → generate → summarizer → END
                     └─(route_existing)──────────────────── → generate → summarizer → END

`build_graph()` is called once at server startup and the compiled graph
is kept as a module-level singleton (thread/async-safe for reads).

The graph is compiled with the LangGraph checkpointer returned by
`db.checkpointer.get_checkpointer()` so each session's state is
persisted between WebSocket reconnections.
"""
import logging
from functools import lru_cache

from langgraph.graph import END, StateGraph

from app.db.checkpointer import get_checkpointer
from app.graph.nodes import create_node_node, create_subtopic_node, generate_node
from app.graph.router import router_node, routing_edge
from app.graph.state import GraphState
from app.graph.summarizer import summarizer_node

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def build_graph():
    """
    Construct, wire, and compile the LangGraph StateGraph.

    Returns the compiled graph (a `CompiledStateGraph`). The result is
    cached via `@lru_cache` so the graph is built exactly once per
    process — the same compiled object is reused for every chat turn.

    Graph nodes:
        router           — semantic routing decision (fast LLM): 3-way
        create_node      — idempotent ROOT topic node creation
        create_subtopic  — idempotent SUB-TOPIC node creation under a parent
        generate         — context-isolated response generation
        summarizer       — async global summary refresh (non-blocking)

    Conditional edges:
        router → create_node      when decision == "create_new"
        router → create_subtopic  when decision == "create_subtopic"
        router → generate         when decision == "route_existing"
    """
    graph = StateGraph(GraphState)

    # ── Register nodes ─────────────────────────────────────────────────
    graph.add_node("router", router_node)
    graph.add_node("create_node", create_node_node)
    graph.add_node("create_subtopic", create_subtopic_node)
    graph.add_node("generate", generate_node)
    graph.add_node("summarizer", summarizer_node)

    # ── Entry point ────────────────────────────────────────────────────
    graph.set_entry_point("router")

    # ── Conditional edges from router ──────────────────────────────────
    graph.add_conditional_edges(
        "router",
        routing_edge,
        {
            "create_node":     "create_node",
            "create_subtopic": "create_subtopic",
            "generate":        "generate",
        },
    )

    # ── Both creation paths flow into generate ─────────────────────────
    graph.add_edge("create_node", "generate")
    graph.add_edge("create_subtopic", "generate")

    # ── generate flows into summarizer (which fires async and exits) ───
    graph.add_edge("generate", "summarizer")

    # ── summarizer → END ──────────────────────────────────────────────
    graph.add_edge("summarizer", END)

    # ── Compile with checkpointer ──────────────────────────────────────
    checkpointer = get_checkpointer()
    compiled = graph.compile(checkpointer=checkpointer)

    logger.info("LangGraph StateGraph compiled successfully (with sub-topic support).")
    return compiled


def get_graph():
    """
    Return the compiled graph singleton.

    Prefer this over calling build_graph() directly — it makes the
    calling code more readable and makes the lazy-init contract explicit.
    """
    return build_graph()
