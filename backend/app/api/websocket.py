"""
Main chat WebSocket gateway: /ws/chat/{session_id}

Receives a user prompt, runs it through the compiled LangGraph, and
streams the response back token-by-token along with routing metadata.

WebSocket message protocol (all messages are JSON-encoded WSMessage):
    → connected   server confirms session is ready
    → routing     server is evaluating which node to use (UX indicator)
    → token       one streamed text chunk from the generator LLM
    → done        generation complete; carries final node metadata
    → error       unrecoverable error; client should reconnect

See PRD section 6.2 "Managing the Latency Tax":
  - Stream "routing" indicator *immediately* before LLM call.
  - Keep the WebSocket open across multiple turns (no reconnect per message).
"""
import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.graph.graph_builder import get_graph
from app.graph.nodes import assemble_context, create_node_node, create_subtopic_node
from app.graph.router import router_node
from app.graph.summarizer import trigger_summary_refresh, trigger_node_summary_refresh
from app.models.schemas import WSMessage
from app.services import llm_service
from app.services.cross_reference import detect_cross_reference
from app.services.retriever import retrieve_relevant_chunks
from app.services.similarity import compute_all_relations

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

# LangGraph config key used to scope state per session (thread_id = session_id)
_GRAPH_CONFIG_KEY = "thread_id"


def _ws_msg(type_: str, **kwargs) -> str:
    """Serialise a WSMessage to a JSON string for sending over the wire."""
    msg = WSMessage(type=type_, **kwargs)  # type: ignore[arg-type]
    return msg.model_dump_json(exclude_none=True)


@router.websocket("/ws/chat/{session_id}")
async def chat_ws(websocket: WebSocket, session_id: str) -> None:
    """
    Main chat WebSocket endpoint.

    One WebSocket connection per session. The connection stays open
    across multiple chat turns — no reconnect needed between messages.

    Message flow per turn:
      1. Client sends: {"content": "...", "force_node_id": null}
      2. Server sends: {"type": "routing", "content": "Evaluating topic..."}
      3. Server sends N× {"type": "token", "content": "<chunk>"}
      4. Server sends: {"type": "done", "node_id": "...", "node_title": "...", "reasoning": "..."}
    """
    await websocket.accept()
    graph = get_graph()
    graph_config = {"configurable": {_GRAPH_CONFIG_KEY: session_id}}

    # ── Send connected confirmation ────────────────────────────────────
    await websocket.send_text(_ws_msg("connected", session_id=session_id))
    logger.info("WebSocket connected: session_id=%r", session_id)

    try:
        while True:
            # ── Receive next user message ──────────────────────────────
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                logger.info("Client disconnected: session_id=%r", session_id)
                break

            try:
                payload = json.loads(raw)
                content: str = payload.get("content", "").strip()
                force_node_id: str | None = payload.get("force_node_id")
                # Feature 5: optional router model override from client
                router_model_override: str | None = payload.get("router_model") or None
            except (json.JSONDecodeError, AttributeError) as exc:
                await websocket.send_text(
                    _ws_msg("error", content=f"Invalid message format: {exc}")
                )
                continue

            if not content:
                await websocket.send_text(
                    _ws_msg("error", content="Empty message — please send some text.")
                )
                continue

            # ── Immediately signal that routing is happening (PRD 6.2) ─
            await websocket.send_text(
                _ws_msg("routing", content="Evaluating topic node…")
            )

            # ── Fetch current graph state (or initialise if first turn) ──────
            try:
                checkpoint = await _get_or_init_state(
                    graph, graph_config, session_id, content, force_node_id,
                    router_model_override=router_model_override,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("State init error: %s", exc)
                # SECURITY-TODO: Implement authentication and connection validation on WebSocket handshake in production.
                # SECURITY-TODO: Enforce rate limits per session/IP on incoming WebSocket frames in production.
                await websocket.send_text(
                    _ws_msg("error", content=f"Failed to initialise session: {exc}")
                )
                continue

            # ── Run the routing phase ────────────────────────────────────
            try:
                routing_result = await _run_routing(
                    graph, graph_config, content, force_node_id,
                    router_model_override=router_model_override,
                )
                node_id: str = routing_result["node_id"]
                node_title: str = routing_result["node_title"]
                parent_node_id: str | None = routing_result.get("parent_node_id")
                node_depth: int = routing_result.get("node_depth", 0)
                reasoning: str = routing_result["reasoning"]
                confidence: float = routing_result.get("confidence", 0.90)
                context_messages: list = routing_result["context_messages"]
                global_summary: str = routing_result["global_summary"]
                router_model_used: str = routing_result.get("router_model_used", settings.router_model)
                router_latency_ms: int = routing_result.get("router_latency_ms", 0)
            except Exception as exc:  # noqa: BLE001
                logger.error("Routing error: %s", exc)
                await websocket.send_text(
                    _ws_msg("error", content=f"Routing failed: {exc}")
                )
                continue

            # ── Notify client of the chosen node ───────────────────────────
            await websocket.send_text(
                _ws_msg(
                    "routing",
                    content=f"Routing to: {node_title}",
                    node_id=node_id,
                    node_title=node_title,
                    parent_node_id=parent_node_id,
                    node_depth=node_depth,
                    reasoning=reasoning,
                    confidence=confidence,
                    router_model_used=router_model_used,
                    router_latency_ms=router_latency_ms,
                )
            )

            # ── Feature 1: Cross-node reference detection ──────────────────────
            # Detect if the user's message references a different existing node.
            # If so, inject that node's summary as bounded one-turn-only context.
            # CRITICAL: the reference snippet is ONLY added to the system_prompt
            # for this call; it is never written to either node's message history.
            all_nodes = routing_result.get("working_state", {}).get("nodes") or {}
            referenced_node_id: str | None = None
            referenced_node_title: str | None = None
            cross_ref_id, cross_ref_snippet = detect_cross_reference(
                content, node_id, all_nodes
            )
            if cross_ref_id and cross_ref_snippet:
                referenced_node_id = cross_ref_id
                referenced_node_title = all_nodes.get(cross_ref_id, {}).get("title")
                logger.info(
                    "Cross-ref: message in %r references node %r (%r)",
                    node_id, cross_ref_id, referenced_node_title,
                )

            # ── Stream the generator response token-by-token ───────────────────
            # Build system prompt — inject local_summary (bounded memory) if present
            local_summary: str = routing_result.get("local_summary", "")
            system_prompt = f"You are an expert assistant on the topic: '{node_title}'."
            if local_summary:
                system_prompt += (
                    f"\n\n## Prior conversation summary (compressed context):\n{local_summary}\n"
                    "Use this as background awareness of what was previously discussed in this topic."
                )
            if global_summary:
                system_prompt += (
                    f"\n\n## Session context (background only):\n{global_summary}"
                )
            # Feature 1: inject bounded cross-node context (one-turn only, not persisted)
            if cross_ref_id and cross_ref_snippet:
                system_prompt += (
                    f"\n\n## Cross-topic reference context (for this response only — do NOT store or repeat in future turns):"
                    f"\nThe user's message appears to reference the '{referenced_node_title}' topic. "
                    f"Here is a brief summary of that topic for your awareness:\n{cross_ref_snippet}"
                )

            # ── Document RAG: retrieve relevant chunks from the active node ────
            source_citations: list[dict] | None = None
            doc_chunks = all_nodes.get(node_id, {}).get("document_chunks", [])
            if doc_chunks:
                relevant = await retrieve_relevant_chunks(content, doc_chunks, top_k=3)
                if relevant:
                    doc_context = "\n\n".join(
                        f"[Source: {c['source_filename']}]\n{c['content']}"
                        for c in relevant
                    )
                    system_prompt += (
                        f"\n\n## Uploaded document context (use as reference when answering):\n{doc_context}"
                    )
                    source_citations = [
                        {
                            "source_filename": c.get("source_filename", ""),
                            "chunk_id": c.get("chunk_id", ""),
                            "relevance_score": c.get("relevance_score", 0),
                            "file_path": c.get("file_path", ""),
                        }
                        for c in relevant
                    ]
                    logger.info(
                        "Document RAG: injected %d chunks from %s into context",
                        len(relevant),
                        ", ".join(set(c["source_filename"] for c in relevant)),
                    )

            # Log how many messages and tokens are being sent this turn
            logger.info(
                "Context sent to LLM | node=%s | live_msgs=%d | has_local_summary=%s | "
                "approx_tokens=%d",
                node_id,
                len(context_messages),
                bool(local_summary),
                sum(len(m.get("content", "")) // 4 for m in context_messages)
                + len(local_summary) // 4
                + len(global_summary) // 4,
            )

            messages_for_llm = context_messages + [
                {"role": "user", "content": content}
            ]

            full_response_parts: list[str] = []
            try:
                async for token in llm_service.stream_generator(
                    messages_for_llm, system_prompt
                ):
                    await websocket.send_text(_ws_msg("token", content=token))
                    full_response_parts.append(token)
            except Exception as exc:  # noqa: BLE001
                logger.error("Streaming error: %s", exc)
                await websocket.send_text(
                    _ws_msg("error", content=f"Generation failed: {exc}")
                )
                continue

            full_response = "".join(full_response_parts)

            # ── Persist the completed turn & update token metrics ───────
            token_stats = {"tokens_used": 0, "baseline_tokens": 0}
            try:
                token_stats = await _persist_turn(
                    graph=graph,
                    config=graph_config,
                    session_id=session_id,
                    node_id=node_id,
                    node_title=node_title,
                    user_content=content,
                    assistant_content=full_response,
                    force_node_id=force_node_id,
                    confidence=confidence,
                    reasoning=reasoning,
                    routing_decision=routing_result.get("reasoning", "route_existing"),
                    router_model_used=router_model_used,
                    router_latency_ms=router_latency_ms,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("State persist error (non-fatal): %s", exc)

            # ── Done ─────────────────────────────────────────────────────
            await websocket.send_text(
                _ws_msg(
                    "done",
                    node_id=node_id,
                    node_title=node_title,
                    parent_node_id=parent_node_id,
                    node_depth=node_depth,
                    reasoning=reasoning,
                    confidence=confidence,
                    tokens_used=token_stats.get("tokens_used", 0),
                    baseline_tokens=token_stats.get("baseline_tokens", 0),
                    # Feature 1: cross-node reference
                    referenced_node_id=referenced_node_id,
                    referenced_node_title=referenced_node_title,
                    # Feature 5: router model metadata
                    router_model_used=router_model_used,
                    router_latency_ms=router_latency_ms,
                    # Document RAG: source citations
                    source_citations=source_citations,
                )
            )

            # ── Trigger background global context summary ───────────────
            asyncio.create_task(
                trigger_summary_refresh(graph, graph_config, session_id, websocket)
            )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: session_id=%r", session_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected WebSocket error: %s", exc)
        try:
            await websocket.send_text(_ws_msg("error", content=str(exc)))
        except Exception:  # noqa: BLE001
            pass


# ── Internal helpers ───────────────────────────────────────────────────


async def _get_or_init_state(graph, config: dict, session_id: str, content: str, force_node_id, router_model_override: str | None = None) -> dict:
    """
    Return the current state snapshot for this session, or an empty
    initial state if the session is brand new.
    """
    try:
        snapshot = graph.get_state(config)
        if snapshot and snapshot.values:
            return snapshot.values
    except Exception:  # noqa: BLE001
        pass

    return {
        "session_id": session_id,
        "nodes": {},
        "active_node_id": "",
        "current_input": content,
        "routing_decision": {},
        "force_node_id": force_node_id,
        "global_summary": "",
        "turn_count": 0,
        "last_response": "",
        "session_tokens_used": 0,
        "session_baseline_tokens": 0,
        "routing_log": [],  # Feature 3: routing timeline
    }


async def _run_routing(
    graph, config: dict, content: str, force_node_id: str | None,
    router_model_override: str | None = None,
) -> dict:
    """
    Run only the routing phase and return node & confidence metadata.
    """
    try:
        snapshot = graph.get_state(config)
        state = snapshot.values if (snapshot and snapshot.values) else {}
    except Exception:  # noqa: BLE001
        state = {}

    working_state = {
        "session_id": state.get("session_id", ""),
        "nodes": state.get("nodes", {}),
        "active_node_id": state.get("active_node_id", ""),
        "current_input": content,
        "routing_decision": {},
        "force_node_id": force_node_id,
        "global_summary": state.get("global_summary", ""),
        "turn_count": state.get("turn_count", 0),
        "last_response": "",
        # Feature 5: pass model override into the router node
        "router_model_override": router_model_override,
    }

    router_updates = await router_node(working_state)
    working_state = {**working_state, **router_updates}

    routing_decision = working_state.get("routing_decision", {})
    decision = routing_decision.get("decision")
    if decision == "create_new":
        create_updates = await create_node_node(working_state)
        working_state = {**working_state, **create_updates}
        routing_decision = working_state.get("routing_decision", routing_decision)
    elif decision == "create_subtopic":
        create_updates = await create_subtopic_node(working_state)
        working_state = {**working_state, **create_updates}
        routing_decision = working_state.get("routing_decision", routing_decision)

    node_id = (
        routing_decision.get("target_node_id")
        or working_state.get("active_node_id")
    )
    nodes = working_state.get("nodes", {})
    node_title = nodes.get(node_id, {}).get("title", "General") if node_id else "General"
    parent_node_id = nodes.get(node_id, {}).get("parent_node_id") if node_id else None
    node_depth = nodes.get(node_id, {}).get("depth", 0) if node_id else 0
    confidence = float(routing_decision.get("confidence", 0.90))

    context_messages: list = []
    global_summary: str = working_state.get("global_summary", "")
    local_summary: str = ""
    if node_id and node_id in nodes:
        ctx = assemble_context(working_state, node_id)
        context_messages = ctx["messages"]
        global_summary = ctx["global_summary"]
        local_summary = ctx.get("local_summary", "")

    try:
        if node_id and node_id != "unknown":
            graph.update_state(config, {"nodes": nodes, "active_node_id": node_id, "force_node_id": None})
    except Exception as exc:
        logger.warning("Failed to persist routing nodes state: %s", exc)

    return {
        "node_id": node_id or "unknown",
        "node_title": node_title,
        "parent_node_id": parent_node_id,
        "node_depth": node_depth,
        "reasoning": routing_decision.get("reasoning", ""),
        "confidence": confidence,
        "context_messages": context_messages,
        "global_summary": global_summary,
        "local_summary": local_summary,
        "working_state": working_state,
        # Feature 5: model metadata from router
        "router_model_used": routing_decision.get("model_used", ""),
        "router_latency_ms": routing_decision.get("latency_ms", 0),
    }


def _est_tokens(text: str) -> int:
    """Simple robust token estimate (~4 chars per token)."""
    return max(1, len(text) // 4)


async def _persist_turn(
    graph,
    config: dict,
    session_id: str,
    node_id: str,
    node_title: str,
    user_content: str,
    assistant_content: str,
    force_node_id: str | None,
    confidence: float = 0.90,
    reasoning: str = "",
    routing_decision: str = "route_existing",
    router_model_used: str = "",
    router_latency_ms: int = 0,
) -> dict:
    """
    Update the LangGraph checkpointer with the completed turn and token statistics.
    Also appends to routing_log (Feature 3) and recomputes relatedness (Feature 2).
    """
    try:
        snapshot = graph.get_state(config)
        current_state = snapshot.values if (snapshot and snapshot.values) else {}
    except Exception:  # noqa: BLE001
        current_state = {}

    nodes = dict(current_state.get("nodes", {}))
    now = datetime.now(timezone.utc).isoformat()

    user_msg_obj = {
        "role": "user",
        "content": user_content,
        "created_at": now,
        "confidence": confidence,
        "reasoning": reasoning,
    }
    assistant_msg_obj = {
        "role": "assistant",
        "content": assistant_content,
        "created_at": now,
        "confidence": confidence,
        "reasoning": reasoning,
    }

    if node_id in nodes:
        node_data = dict(nodes[node_id])
        msgs = list(node_data.get("messages", []))
        msgs.append(user_msg_obj)
        msgs.append(assistant_msg_obj)
        node_data["messages"] = msgs
        node_data["turn_count"] = node_data.get("turn_count", 0) + 1
        node_data["last_active_at"] = now
        nodes[node_id] = node_data
    elif node_id:
        nodes[node_id] = {
            "title": node_title or "New Topic",
            "messages": [user_msg_obj, assistant_msg_obj],
            "turn_count": 1,
            "created_at": now,
            "last_active_at": now,
            "parent_node_id": None,
            "depth": 0,
            "node_summary": "",
            "local_summary": "",
            "archived_messages": [],
            "document_chunks": [],
            "related_node_ids": [],
            "possible_duplicate_of": None,
        }

    # ── Token accounting (bounded memory fix) ──────────────────────────────
    # For any turn, the cost difference between graph chat and linear chat is:
    #
    #   Linear chat (baseline): would send ALL messages across ALL nodes every turn.
    #   Graph chat (actual):    sends ONLY the active node's LIVE messages
    #                           + local_summary (compressed prior context)
    #                           + global summary.
    #
    # With in-node bounded memory, actual_turn_tokens is bounded by keep_last_n
    # instead of growing linearly. The local_summary is a small fixed-size
    # compressed digest, so context size plateaus after the window is crossed.

    active_msgs = nodes.get(node_id, {}).get("messages", [])
    local_summary_text = nodes.get(node_id, {}).get("local_summary", "")
    active_tokens = sum(_est_tokens(m.get("content", "")) for m in active_msgs)
    local_summary_tokens = _est_tokens(local_summary_text)
    global_sum_tokens = _est_tokens(current_state.get("global_summary", ""))

    # Actual this turn = live messages + local_summary (compressed prior) + global summary
    # This is bounded by keep_last_n * avg_message_size, not total history length.
    actual_turn_tokens = active_tokens + local_summary_tokens + global_sum_tokens

    # Baseline this turn = ALL messages across ALL nodes (archived + live)
    # This represents what a naive linear chat would have sent.
    baseline_turn_tokens = sum(
        sum(_est_tokens(m.get("content", "")) for m in (
            list(nd.get("archived_messages") or []) + list(nd.get("messages") or [])
        ))
        for nd in nodes.values()
    )
    # Baseline must be at least as large as actual (sanity floor)
    baseline_turn_tokens = max(baseline_turn_tokens, actual_turn_tokens)

    prev_used = current_state.get("session_tokens_used", 0)
    prev_baseline = current_state.get("session_baseline_tokens", 0)

    new_tokens_used = prev_used + actual_turn_tokens
    new_baseline_tokens = prev_baseline + baseline_turn_tokens

    logger.info(
        "Token accounting | node=%s | live_msgs=%d live_tokens=%d "
        "local_summary_tokens=%d global_tokens=%d actual_turn=%d | "
        "archived_msgs=%d baseline_turn=%d | "
        "session used=%d baseline=%d savings=%.1f%%",
        node_id,
        len(active_msgs),
        active_tokens,
        local_summary_tokens,
        global_sum_tokens,
        actual_turn_tokens,
        len(nodes.get(node_id, {}).get("archived_messages") or []),
        baseline_turn_tokens,
        new_tokens_used,
        new_baseline_tokens,
        (100.0 * (new_baseline_tokens - new_tokens_used) / new_baseline_tokens)
        if new_baseline_tokens > 0 else 0.0,
    )

    new_turn_count = current_state.get("turn_count", 0) + 1

    # Feature 3: Append routing decision to routing_log
    existing_log: list = list(current_state.get("routing_log") or [])
    existing_log.append({
        "timestamp": now,
        "message_excerpt": user_content[:120],
        "decision": routing_decision,
        "target_node_id": node_id,
        "node_title": node_title,
        "confidence": confidence,
        "reasoning": reasoning,
        "router_model": router_model_used,
        "latency_ms": router_latency_ms,
    })

    # Feature 2: Recompute relatedness between all nodes
    try:
        relation_updates = compute_all_relations(nodes)
        for nid, updates in relation_updates.items():
            if nid in nodes:
                node_copy = dict(nodes[nid])
                node_copy["related_node_ids"] = updates.get("related_node_ids", [])
                # Only set possible_duplicate_of if not already set (don't overwrite existing merge flags)
                if not node_copy.get("possible_duplicate_of") and updates.get("possible_duplicate_of"):
                    node_copy["possible_duplicate_of"] = updates["possible_duplicate_of"]
                nodes[nid] = node_copy
    except Exception as exc:
        logger.warning("Relatedness computation failed (non-fatal): %s", exc)

    graph.update_state(
        config,
        {
            "session_id": session_id,
            "nodes": nodes,
            "active_node_id": node_id,
            "current_input": user_content,
            "routing_decision": {},
            "force_node_id": None,
            "global_summary": current_state.get("global_summary", ""),
            "turn_count": new_turn_count,
            "last_response": assistant_content,
            "session_tokens_used": new_tokens_used,
            "session_baseline_tokens": new_baseline_tokens,
            "routing_log": existing_log,  # Feature 3
        },
    )

    # ── Trigger per-node compression if live message count exceeds window ──
    # This mirrors the global summarizer trigger — fire-and-forget background task.
    keep_last_n = settings.node_keep_last_n
    node_msg_count = len(nodes.get(node_id, {}).get("messages") or [])
    if node_msg_count > keep_last_n:
        logger.info(
            "Node %s has %d live messages (threshold=%d) — scheduling local summary compression.",
            node_id, node_msg_count, keep_last_n,
        )
        asyncio.create_task(
            trigger_node_summary_refresh(graph, config, node_id, keep_last_n)
        )

    return {
        "tokens_used": new_tokens_used,
        "baseline_tokens": new_baseline_tokens,
    }
