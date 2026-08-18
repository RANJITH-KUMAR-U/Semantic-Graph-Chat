"""
Topic Node REST endpoints.

Routes:
    GET  /api/sessions/{session_id}/nodes           — list all topic nodes
    GET  /api/nodes/{node_id}/messages              — message history for a node
    POST /api/nodes/{node_id}/force-route           — pin next message to this node
    POST /api/messages/{message_id}/reassign        — reassign message to another node
    POST /api/nodes/merge                           — merge duplicate topic nodes
    GET  /api/sessions/{session_id}/search          — search session messages
    GET  /api/sessions/{session_id}/recap           — return session recap on load
"""
import uuid
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from app.graph.graph_builder import get_graph
from app.models.schemas import ForceRouteIn, MessageOut, NodeOut, ReassignMessageIn, MergeNodesIn

logger = logging.getLogger(__name__)

router = APIRouter(tags=["nodes"])

_GRAPH_CONFIG_KEY = "thread_id"

# SECURITY-TODO: Enforce user authentication, session ownership checks, and rate limits on all endpoints in production.


def _make_config(session_id: str) -> dict:
    return {"configurable": {_GRAPH_CONFIG_KEY: session_id}}


def _parse_dt(value: str | None) -> datetime:
    """Parse an ISO-8601 string or return now() on failure."""
    if value:
        try:
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            pass
    return datetime.now(timezone.utc)


@router.get("/sessions/{session_id}/nodes", response_model=list[NodeOut])
async def list_nodes(session_id: str) -> list[NodeOut]:
    """
    Return all active topic nodes for a session.
    """
    # SECURITY-TODO: Verify authorization for session_id.
    graph = get_graph()
    config = _make_config(session_id)

    try:
        snapshot = graph.get_state(config)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not snapshot or not snapshot.values:
        return []

    nodes_raw: dict = snapshot.values.get("nodes", {})
    result: list[NodeOut] = []
    from app.services.similarity import check_node_consistency
    for node_id, node_data in nodes_raw.items():
        msg_count = len(node_data.get("messages", []))
        archived_count = len(node_data.get("archived_messages") or [])
        doc_chunk_count = len(node_data.get("document_chunks") or [])
        # Guard: Only list nodes that have at least 1 real message or document chunks (prevents ghost nodes)
        if msg_count == 0 and archived_count == 0 and doc_chunk_count == 0:
            continue

        # Semantic consistency check
        is_consistent, score, reason = check_node_consistency(node_data)
        if not is_consistent:
            logger.warning(
                "Node %r (%s) consistency check failed: %s (score=%.3f)",
                node_id, node_data.get("title"), reason, score
            )

        archived_count = len(node_data.get("archived_messages") or [])
        doc_chunk_count = len(node_data.get("document_chunks") or [])
        result.append(
            NodeOut(
                node_id=node_id,
                session_id=session_id,
                title=node_data.get("title", "Untitled"),
                message_count=msg_count + archived_count,
                created_at=_parse_dt(node_data.get("created_at")),
                last_active_at=_parse_dt(node_data.get("last_active_at")),
                parent_node_id=node_data.get("parent_node_id"),
                depth=node_data.get("depth", 0),
                possible_duplicate_of=node_data.get("possible_duplicate_of"),
                related_node_ids=node_data.get("related_node_ids", []),
                local_summary=node_data.get("local_summary") or "",
                archived_message_count=archived_count,
                document_chunk_count=doc_chunk_count,
            )
        )
    result.sort(key=lambda n: n.created_at)
    return result


@router.get("/nodes/{node_id}/messages", response_model=list[MessageOut])
async def get_node_messages(node_id: str, session_id: str) -> list[MessageOut]:
    """
    Return the full message history for a single topic node.
    """
    # SECURITY-TODO: Ensure session scoping to prevent unauthorized message reads.
    graph = get_graph()
    config = _make_config(session_id)

    try:
        snapshot = graph.get_state(config)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")

    nodes_raw: dict = snapshot.values.get("nodes", {})
    if node_id not in nodes_raw:
        raise HTTPException(
            status_code=404,
            detail=f"Node {node_id!r} not found in session {session_id!r}.",
        )

    node_data = nodes_raw[node_id]
    # Return full message history: archived (oldest first) + live messages
    # This ensures search and export see the complete history even after compression.
    archived_messages: list[dict] = list(node_data.get("archived_messages") or [])
    live_messages: list[dict] = list(node_data.get("messages") or [])
    messages_raw: list[dict] = archived_messages + live_messages

    result: list[MessageOut] = []
    for idx, msg in enumerate(messages_raw):
        created_at_dt = _parse_dt(msg.get("created_at"))
        result.append(
            MessageOut(
                message_id=msg.get("id") or f"{node_id}_msg_{idx}",
                node_id=node_id,
                role=msg.get("role", "user"),  # type: ignore[arg-type]
                content=msg.get("content", ""),
                created_at=created_at_dt,
                confidence=msg.get("confidence", 0.90),
                reasoning=msg.get("reasoning"),
            )
        )
    return result


@router.post("/nodes/{node_id}/force-route", status_code=200)
async def force_route(node_id: str, body: ForceRouteIn) -> dict:
    """
    Pin the next incoming message to a specific topic node.
    """
    # SECURITY-TODO: Verify CSRF token and caller session permissions.
    session_id = body.session_id
    graph = get_graph()
    config = _make_config(session_id)

    try:
        snapshot = graph.get_state(config)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")

    nodes_raw = snapshot.values.get("nodes", {})
    if node_id not in nodes_raw:
        raise HTTPException(
            status_code=404,
            detail=f"Node {node_id!r} not found in session {session_id!r}.",
        )

    graph.update_state(config, {"force_node_id": node_id})

    return {
        "message": f"Next message will be routed to node {node_id!r}.",
        "node_id": node_id,
        "node_title": nodes_raw[node_id].get("title", "Untitled"),
    }


# ── Feature 2: Manual Re-route ─────────────────────────────────────────


@router.post("/messages/{message_id}/reassign", status_code=200)
async def reassign_message(message_id: str, body: ReassignMessageIn) -> dict:
    """
    Reassign a message to another existing node or a new node.
    """
    # SECURITY-TODO: Sanitize message inputs and enforce session ownership.
    session_id = body.session_id
    graph = get_graph()
    config = _make_config(session_id)

    try:
        snapshot = graph.get_state(config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")

    nodes = dict(snapshot.values.get("nodes", {}))

    # Find the target message across nodes
    target_msg = None
    source_node_id = None
    msg_index = -1

    for nid, ndata in nodes.items():
        msgs = ndata.get("messages", [])
        for idx, m in enumerate(msgs):
            m_id = m.get("id") or f"{nid}_msg_{idx}"
            if m_id == message_id:
                target_msg = m
                source_node_id = nid
                msg_index = idx
                break
        if target_msg:
            break

    if not target_msg or not source_node_id:
        raise HTTPException(status_code=404, detail=f"Message {message_id!r} not found.")

    # Determine target node ID (create new node if requested)
    target_node_id = body.target_node_id
    now = datetime.now(timezone.utc).isoformat()

    if body.new_topic_title:
        target_node_id = f"node_{uuid.uuid4().hex[:8]}"
        nodes[target_node_id] = {
            "title": body.new_topic_title,
            "messages": [],
            "turn_count": 0,
            "created_at": now,
            "last_active_at": now,
            "parent_node_id": None,
            "depth": 0,
        }
    elif not target_node_id or target_node_id not in nodes:
        raise HTTPException(status_code=400, detail="Invalid target_node_id specified.")

    # Move message from source to target node
    source_node = dict(nodes[source_node_id])
    source_msgs = list(source_node.get("messages", []))
    if 0 <= msg_index < len(source_msgs):
        source_msgs.pop(msg_index)
    source_node["messages"] = source_msgs
    nodes[source_node_id] = source_node

    target_node = dict(nodes[target_node_id])
    target_msgs = list(target_node.get("messages", []))
    target_msgs.append(target_msg)
    target_node["messages"] = target_msgs
    target_node["last_active_at"] = now
    nodes[target_node_id] = target_node

    graph.update_state(config, {"nodes": nodes, "active_node_id": target_node_id})
    logger.info("Reassigned message %s from node %s to %s", message_id, source_node_id, target_node_id)

    return {
        "success": True,
        "source_node_id": source_node_id,
        "target_node_id": target_node_id,
        "target_node_title": target_node.get("title", "Untitled"),
    }


# ── Feature 3: Topic Merge Detection & Execution ───────────────────────


@router.post("/nodes/merge", status_code=200)
async def merge_nodes(body: MergeNodesIn) -> dict:
    """
    Merge source_node_id into target_node_id, combining message histories.
    """
    # SECURITY-TODO: Ensure session ownership check before deleting/merging nodes.
    session_id = body.session_id
    graph = get_graph()
    config = _make_config(session_id)

    try:
        snapshot = graph.get_state(config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")

    nodes = dict(snapshot.values.get("nodes", {}))

    if body.source_node_id not in nodes or body.target_node_id not in nodes:
        raise HTTPException(status_code=404, detail="Source or target node not found.")

    source_node = nodes[body.source_node_id]
    target_node = dict(nodes[body.target_node_id])

    # Combine message histories
    target_msgs = list(target_node.get("messages", [])) + list(source_node.get("messages", []))
    target_node["messages"] = target_msgs
    target_node["possible_duplicate_of"] = None
    target_node["last_active_at"] = datetime.now(timezone.utc).isoformat()
    nodes[body.target_node_id] = target_node

    # Remove source node
    del nodes[body.source_node_id]

    graph.update_state(config, {"nodes": nodes, "active_node_id": body.target_node_id})
    logger.info("Merged node %s into %s", body.source_node_id, body.target_node_id)

    return {
        "success": True,
        "target_node_id": body.target_node_id,
        "target_node_title": target_node.get("title"),
        "total_messages": len(target_msgs),
    }


# ── Feature 5: Topic Search ─────────────────────────────────────────────


@router.get("/sessions/{session_id}/search", status_code=200)
async def search_session_messages(session_id: str, q: str = Query(..., min_length=1)) -> list[dict]:
    """
    Search across all message contents in the session.
    """
    # SECURITY-TODO: Escape query strings to prevent search injection attack vectors.
    graph = get_graph()
    config = _make_config(session_id)

    try:
        snapshot = graph.get_state(config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not snapshot or not snapshot.values:
        return []

    nodes: dict = snapshot.values.get("nodes", {})
    query = q.lower()
    matches = []

    for node_id, node_data in nodes.items():
        node_title = node_data.get("title", "Untitled")
        # Search both archived and live messages — full history must be searchable
        # even after in-node compression has moved older messages to the archive.
        archived = list(node_data.get("archived_messages") or [])
        live = list(node_data.get("messages") or [])
        all_messages = archived + live
        for idx, m in enumerate(all_messages):
            content = m.get("content", "")
            if query in content.lower():
                matches.append({
                    "message_id": m.get("id") or f"{node_id}_msg_{idx}",
                    "node_id": node_id,
                    "node_title": node_title,
                    "role": m.get("role", "user"),
                    "content": content,
                    "created_at": m.get("created_at"),
                })

    return matches


# ── Feature 6: Session Recap on Return ──────────────────────────────────


@router.get("/sessions/{session_id}/recap", status_code=200)
async def get_session_recap(session_id: str) -> dict:
    """
    Return session recap metadata on session load.
    """
    # SECURITY-TODO: Ensure session privacy validation.
    graph = get_graph()
    config = _make_config(session_id)

    try:
        snapshot = graph.get_state(config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not snapshot or not snapshot.values:
        return {
            "has_history": False,
            "active_node_title": None,
            "node_count": 0,
            "global_summary": "",
        }

    state = snapshot.values
    nodes = state.get("nodes", {})
    active_node_id = state.get("active_node_id")
    active_title = nodes.get(active_node_id, {}).get("title") if active_node_id else None

    # Fallback to most recently active node title
    if not active_title and nodes:
        sorted_nodes = sorted(
            nodes.values(),
            key=lambda n: n.get("last_active_at", ""),
            reverse=True,
        )
        active_title = sorted_nodes[0].get("title")

    return {
        "has_history": len(nodes) > 0,
        "active_node_id": active_node_id,
        "active_node_title": active_title,
        "node_count": len(nodes),
        "global_summary": state.get("global_summary", ""),
        "session_tokens_used": state.get("session_tokens_used", 0),
        "session_baseline_tokens": state.get("session_baseline_tokens", 0),
    }


# ── Feature Round 3, Feature 3: Routing Decision Timeline ──────────────


@router.get("/sessions/{session_id}/routing-log", status_code=200)
async def get_routing_log(session_id: str) -> list[dict]:
    """
    Return the chronological routing decision log for this session.
    """
    # SECURITY-TODO: Validate session ownership before exposing routing log.
    graph = get_graph()
    config = _make_config(session_id)

    try:
        snapshot = graph.get_state(config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not snapshot or not snapshot.values:
        return []

    return list(snapshot.values.get("routing_log") or [])


# ── Feature Round 3, Feature 4: Session Export as Markdown ─────────────


@router.get("/sessions/{session_id}/export", status_code=200)
async def export_session_markdown(session_id: str):
    """
    Export the whole session as a structured Markdown document,
    organised by topic node (not one flat transcript).
    """
    # SECURITY-TODO: Validate session ownership before allowing export.
    from fastapi.responses import Response

    graph = get_graph()
    config = _make_config(session_id)

    try:
        snapshot = graph.get_state(config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")

    state = snapshot.values
    nodes: dict = state.get("nodes", {})
    global_summary: str = state.get("global_summary", "")
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    lines.append(f"# Session Export — {now_str}")
    lines.append(f"**Session ID:** `{session_id}`")
    lines.append(f"**Topics:** {len(nodes)}")
    lines.append("")

    # Global summary section
    if global_summary:
        lines.append("## Global Summary")
        lines.append("")
        lines.append(global_summary)
        lines.append("")

    # Sort nodes by creation time
    sorted_nodes = sorted(
        nodes.items(),
        key=lambda kv: kv[1].get("created_at", ""),
    )

    for node_id, node_data in sorted_nodes:
        title = node_data.get("title", "Untitled")
        msg_count = len(node_data.get("messages", []))
        if msg_count == 0:
            continue

        depth = node_data.get("depth", 0)
        prefix = "##" if depth == 0 else "###"
        parent_note = ""
        if node_data.get("parent_node_id"):
            parent_title = nodes.get(node_data["parent_node_id"], {}).get("title", "")
            if parent_title:
                parent_note = f" *(sub-topic of: {parent_title})*"

        lines.append(f"{prefix} {title}{parent_note}")
        lines.append("")

        for msg in node_data.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            timestamp = msg.get("created_at", "")
            ts_str = f" *({timestamp[:19]})*" if timestamp else ""

            if role == "user":
                lines.append(f"**You:**{ts_str}")
            else:
                lines.append(f"**Assistant:**{ts_str}")
            lines.append("")
            lines.append(content)
            lines.append("")

        lines.append("---")
        lines.append("")

    md_content = "\n".join(lines)
    filename = f"session-{session_id[:8]}.md"

    return Response(
        content=md_content,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


# ── Feature Round 3, Feature 5: Available Router Models ────────────────


@router.get("/config/available-router-models", status_code=200)
async def get_available_router_models() -> list[dict]:
    """
    Return the list of configured router models that the frontend can
    select from for the model playground toggle.
    """
    from app.core.config import settings

    models = []
    seen = set()

    for model_id in [settings.router_model, settings.router_model_alt]:
        if model_id and model_id not in seen:
            seen.add(model_id)
            # Derive a short display name from the model ID
            display_name = model_id.split("/")[-1].replace(":free", "").replace(":", " ")
            models.append({
                "model_id": model_id,
                "display_name": display_name,
                "is_default": model_id == settings.router_model,
            })

    return models
