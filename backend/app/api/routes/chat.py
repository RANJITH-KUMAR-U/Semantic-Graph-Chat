"""
Session bootstrap REST endpoints.

Routes:
    POST /api/sessions          — create (or resume) a session
    GET  /api/sessions/{id}     — get session metadata + node list
    GET  /api/sessions/{id}/summary — get current global summary

These are lightweight bookkeeping routes. The main chat loop lives in
api/websocket.py — these routes just let the frontend initialise state
and display the session's topic graph.
"""
import uuid
import logging

from fastapi import APIRouter, HTTPException

from app.graph.graph_builder import get_graph
from app.models.schemas import CreateSessionIn, SessionOut, NodeOut

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"])

# LangGraph config key
_GRAPH_CONFIG_KEY = "thread_id"


def _make_config(session_id: str) -> dict:
    return {"configurable": {_GRAPH_CONFIG_KEY: session_id}}


# SECURITY-TODO: Enforce session token authorization and CORS origin strictness in production.


def _state_to_session_out(session_id: str, state: dict) -> SessionOut:
    """Convert raw graph state to a SessionOut DTO."""
    from datetime import datetime, timezone

    nodes_raw: dict = state.get("nodes", {})
    node_outs = []
    for node_id, node_data in nodes_raw.items():
        try:
            created_at = datetime.fromisoformat(node_data.get("created_at", datetime.now(timezone.utc).isoformat()))
            last_active_at = datetime.fromisoformat(node_data.get("last_active_at", datetime.now(timezone.utc).isoformat()))
        except (ValueError, TypeError):
            now = datetime.now(timezone.utc)
            created_at = last_active_at = now

        node_outs.append(
            NodeOut(
                node_id=node_id,
                session_id=session_id,
                title=node_data.get("title", "Untitled"),
                message_count=len(node_data.get("messages", [])),
                created_at=created_at,
                last_active_at=last_active_at,
                parent_node_id=node_data.get("parent_node_id"),
                depth=node_data.get("depth", 0),
                possible_duplicate_of=node_data.get("possible_duplicate_of"),
            )
        )

    node_outs.sort(key=lambda n: n.created_at)

    return SessionOut(
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
        global_summary=state.get("global_summary", ""),
        nodes=node_outs,
    )


@router.post("", response_model=SessionOut, status_code=201)
async def create_session(body: CreateSessionIn | None = None) -> SessionOut:
    """
    Create a new chat session (or acknowledge an existing one).

    If `session_id` is provided in the request body, the session is
    registered/resumed. If omitted, a new UUID is generated.

    The graph state is initialised on the first WebSocket message — this
    endpoint just reserves the session ID and returns metadata.
    """
    from datetime import datetime, timezone

    session_id = (body.session_id if body and body.session_id else None) or str(uuid.uuid4())

    graph = get_graph()
    config = _make_config(session_id)

    # Check if this session already exists in the checkpointer
    try:
        snapshot = graph.get_state(config)
        if snapshot and snapshot.values:
            logger.info("Resuming existing session %r", session_id)
            return _state_to_session_out(session_id, snapshot.values)
    except Exception:  # noqa: BLE001
        pass

    # New session — seed minimal state
    logger.info("Creating new session %r", session_id)
    return SessionOut(
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
        global_summary="",
        nodes=[],
    )


@router.get("/{session_id}", response_model=SessionOut)
async def get_session(session_id: str) -> SessionOut:
    """
    Return full session metadata including all topic nodes.

    Used by the frontend to populate the graph sidebar on page load.
    """
    graph = get_graph()
    config = _make_config(session_id)

    try:
        snapshot = graph.get_state(config)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to read session state: {exc}") from exc

    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")

    return _state_to_session_out(session_id, snapshot.values)


@router.get("/{session_id}/summary")
async def get_global_summary(session_id: str) -> dict:
    """
    Return the current global context summary for this session.

    The summary is refreshed asynchronously every N turns by the
    summarizer node — this endpoint just reads the latest cached value.
    """
    graph = get_graph()
    config = _make_config(session_id)

    try:
        snapshot = graph.get_state(config)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not snapshot or not snapshot.values:
        return {
            "session_id": session_id,
            "global_summary": "",
            "turn_count": 0,
        }

    return {
        "session_id": session_id,
        "global_summary": snapshot.values.get("global_summary", ""),
        "turn_count": snapshot.values.get("turn_count", 0),
    }
