"""
File upload endpoint — POST /api/sessions/{session_id}/upload

Accepts multipart/form-data with a single file.  The file is chunked
using the type-aware chunker, then each chunk group is routed to the
appropriate topic node using the existing semantic router.

# SECURITY-TODO: Add file content validation (magic bytes check).
# SECURITY-TODO: Add rate limiting per session for uploads.
# SECURITY-TODO: Sanitize uploaded filenames to prevent path-traversal.
"""
from __future__ import annotations

import logging
import uuid
from collections import defaultdict

from fastapi import APIRouter, HTTPException, UploadFile, File, Path

from app.graph.graph_builder import get_graph
from app.models.schemas import UploadStatusOut
from app.services.chunker import (
    chunk_file,
    SUPPORTED_EXTENSIONS,
    MAX_FILE_SIZE,
    DocumentChunk,
)
from app.services import llm_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["upload"])

# LangGraph config key — same as websocket.py
_GRAPH_CONFIG_KEY = "thread_id"


def _get_extension(filename: str) -> str:
    """Get lowercase file extension."""
    import os
    _, ext = os.path.splitext(filename.lower())
    return ext


@router.post(
    "/sessions/{session_id}/upload",
    response_model=UploadStatusOut,
    summary="Upload a document for chunking and RAG indexing",
)
async def upload_file_endpoint(
    session_id: str = Path(..., description="Session that owns the upload"),
    file: UploadFile = File(..., description="File to upload"),
) -> UploadStatusOut:
    """
    Upload a file, chunk it, route chunks to topic nodes, and store them.

    Supported formats: .pdf, .docx, .txt, .md, .zip
    Max size: 10MB per file.

    The endpoint is synchronous — it returns after all chunks have been
    indexed. For very large files this may take a few seconds.
    """
    upload_id = f"upload_{uuid.uuid4().hex[:8]}"
    filename = file.filename or "unknown"
    ext = _get_extension(filename)

    logger.info("Upload %s: file=%r ext=%r session=%r", upload_id, filename, ext, session_id)

    # ── Validate file type ────────────────────────────────────────────
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type: {ext}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    # ── Read file bytes ───────────────────────────────────────────────
    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large: {len(file_bytes) / (1024*1024):.1f}MB "
                f"(max {MAX_FILE_SIZE / (1024*1024):.0f}MB)"
            ),
        )

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # ── Chunk the file ────────────────────────────────────────────────
    try:
        chunks: list[DocumentChunk] = chunk_file(filename, file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Chunking failed for %s: %s", filename, exc)
        return UploadStatusOut(
            upload_id=upload_id,
            filename=filename,
            status="failed",
            error=f"Chunking failed: {exc}",
        )

    if not chunks:
        return UploadStatusOut(
            upload_id=upload_id,
            filename=filename,
            status="indexed",
            total_chunks=0,
            node_assignments={},
        )

    logger.info("Upload %s: %d chunks produced", upload_id, len(chunks))

    # ── Get current graph state ───────────────────────────────────────
    graph = get_graph()
    graph_config = {"configurable": {_GRAPH_CONFIG_KEY: session_id}}

    try:
        snapshot = graph.get_state(graph_config)
        state = snapshot.values if (snapshot and snapshot.values) else {}
    except Exception:
        state = {}

    nodes = dict(state.get("nodes", {}))
    active_node_id = state.get("active_node_id", "")

    # ── Route chunks to nodes ─────────────────────────────────────────
    # Strategy: batch chunks into groups of 5, summarise each group,
    # and make one router call per group to classify.
    BATCH_SIZE = 5
    chunk_batches: list[list[DocumentChunk]] = []
    for i in range(0, len(chunks), BATCH_SIZE):
        chunk_batches.append(chunks[i:i + BATCH_SIZE])

    # Track which node each chunk is assigned to
    node_chunk_map: dict[str, list[DocumentChunk]] = defaultdict(list)

    for batch in chunk_batches:
        # Create a summary of this batch for routing
        batch_summary = "\n---\n".join(
            c["content"][:500] for c in batch
        )

        # Route the batch
        target_node_id = await _route_chunk_batch(
            batch_summary, nodes, active_node_id, graph, graph_config
        )

        for chunk in batch:
            node_chunk_map[target_node_id].append(chunk)

    # ── Store chunks in their assigned nodes ──────────────────────────
    node_assignments: dict[str, dict] = {}

    for node_id, assigned_chunks in node_chunk_map.items():
        if node_id not in nodes:
            # Node was just created by routing — re-fetch state
            try:
                snapshot = graph.get_state(graph_config)
                state = snapshot.values if (snapshot and snapshot.values) else {}
                nodes = dict(state.get("nodes", {}))
            except Exception:
                pass

        if node_id in nodes:
            node_data = dict(nodes[node_id])
            existing_chunks = list(node_data.get("document_chunks") or [])
            existing_chunks.extend(assigned_chunks)
            node_data["document_chunks"] = existing_chunks
            nodes[node_id] = node_data

            node_title = node_data.get("title", "Untitled")
            node_assignments[node_id] = {
                "title": node_title,
                "chunk_count": len(assigned_chunks),
            }
            logger.info(
                "Upload %s: assigned %d chunks to node %r (%r)",
                upload_id, len(assigned_chunks), node_id, node_title,
            )

    # ── Persist updated state & trigger global summary refresh ───────
    try:
        graph.update_state(graph_config, {"nodes": nodes})
        from app.graph.summarizer import trigger_summary_refresh
        asyncio.create_task(trigger_summary_refresh(graph, graph_config, session_id))
    except Exception as exc:
        logger.error("Failed to persist chunks to state: %s", exc)
        return UploadStatusOut(
            upload_id=upload_id,
            filename=filename,
            status="failed",
            total_chunks=len(chunks),
            error=f"State persistence failed: {exc}",
        )

    logger.info(
        "Upload %s complete: %d chunks → %d nodes",
        upload_id, len(chunks), len(node_assignments),
    )

    return UploadStatusOut(
        upload_id=upload_id,
        filename=filename,
        status="indexed",
        total_chunks=len(chunks),
        node_assignments=node_assignments,
    )


# ── Internal helpers ───────────────────────────────────────────────────


async def _route_chunk_batch(
    batch_text: str,
    nodes: dict,
    active_node_id: str,
    graph,
    graph_config: dict,
) -> str:
    """
    Route a batch of chunks to the most appropriate node.

    Uses the same LLM router as chat messages. If no nodes exist,
    creates a new one.
    """
    from app.graph.router import router_node, RoutingDecision
    from app.graph.nodes import create_node_node

    # Build a minimal state for routing
    routing_state = {
        "session_id": "",
        "nodes": nodes,
        "active_node_id": active_node_id,
        "current_input": batch_text[:2000],  # Cap for router context
        "routing_decision": {},
        "force_node_id": None,
        "global_summary": "",
        "turn_count": 0,
        "last_response": "",
    }

    try:
        router_updates = await router_node(routing_state)
        routing_state = {**routing_state, **router_updates}

        decision = routing_state.get("routing_decision", {})
        decision_type = decision.get("decision", "create_new")

        if decision_type == "create_new" or decision_type == "create_subtopic":
            # Create a new node for this batch
            create_updates = await create_node_node(routing_state)
            routing_state = {**routing_state, **create_updates}
            # Persist the new node into the graph state
            new_nodes = routing_state.get("nodes", {})
            nodes.update(new_nodes)
            try:
                graph.update_state(graph_config, {"nodes": nodes})
            except Exception as exc:
                logger.warning("Failed to persist new node from upload routing: %s", exc)

        target_node_id = (
            routing_state.get("routing_decision", {}).get("target_node_id")
            or routing_state.get("active_node_id")
        )

        if target_node_id:
            return target_node_id

    except Exception as exc:
        logger.warning("Chunk routing failed: %s — using active node", exc)

    # Fallback: use active node or first available node
    if active_node_id and active_node_id in nodes:
        return active_node_id

    if nodes:
        return next(iter(nodes))

    # Last resort: create a node manually
    from app.graph.nodes import _new_node_id, _utcnow
    node_id = _new_node_id()
    now = _utcnow()
    nodes[node_id] = {
        "title": "Uploaded Document",
        "messages": [],
        "turn_count": 0,
        "created_at": now,
        "last_active_at": now,
        "parent_node_id": None,
        "depth": 0,
        "node_summary": "",
        "document_chunks": [],
    }
    try:
        graph.update_state(graph_config, {"nodes": nodes, "active_node_id": node_id})
    except Exception:
        pass

    return node_id
