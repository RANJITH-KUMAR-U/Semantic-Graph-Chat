"""
Global Context Summarizer — runs asynchronously every N turns
(settings.summarizer_every_n_turns) to refresh state["global_summary"]
without blocking the main response path.

Non-negotiable constraint (AGENTS.md):
  "The global summarizer runs async and must never block the main
   response path (PRD section 6.3)."

The summarizer compresses ALL nodes' recent activity into a compact,
cross-topic digest. This summary is then injected as background context
into each node's generation prompt — giving the LLM global coherence
without violating isolation (because the summary is derived, not raw
messages from other nodes).

In-node bounded memory:
  Each node can also have its own local_summary — a compressed digest of
  messages beyond the rolling window (node_keep_last_n). The per-node
  summarizer (refresh_node_local_summary / trigger_node_summary_refresh)
  mirrors this global pattern but operates on a single node's overflow.
"""
import asyncio
import logging

from app.graph.state import GraphState
from app.services import llm_service

logger = logging.getLogger(__name__)


# SECURITY-TODO: Ensure session ownership validation before reading or updating session state in production.

async def refresh_global_summary(state: GraphState) -> str:
    """
    Summarise recent activity across all topic nodes into a compact string (2-4 sentences).

    Reads both archived_messages and live messages for each node so the
    global summary maintains awareness of the full topic history even
    after in-node compression has archived older messages.

    Args:
        state: Full graph state.

    Returns:
        A compact multi-line string describing the session's topic threads.
    """
    nodes: dict = state.get("nodes", {})
    if not nodes:
        return ""

    # Build a digest of each node's last 3 messages (archived + live)
    node_digests: list[str] = []
    for node_id, node_data in nodes.items():
        title = node_data.get("title", "Untitled")
        # Combine archived and live messages for the global summarizer
        archived = list(node_data.get("archived_messages") or [])
        live = list(node_data.get("messages") or [])
        all_msgs = archived + live
        recent = all_msgs[-6:]
        if not recent:
            continue
        turns_text = "\n".join(
            f"  {m['role'].upper()}: {m['content'][:200]}"
            for m in recent
        )
        node_digests.append(f"### Topic: {title} (node_id={node_id})\n{turns_text}")

    if not node_digests:
        return state.get("global_summary", "")

    combined = "\n\n".join(node_digests)
    system_prompt = (
        "You are a context summariser for a multi-topic chat system.\n"
        "Below is recent activity across topic threads in a user session.\n"
        "Write a concise, high-level summary (2-4 sentences, max 100 words) summarizing:\n"
        "  - Main topics discussed so far\n"
        "  - Key concepts, decisions, or answers provided\n"
        "Be factual, clear, and direct. Do not include meta-commentary."
    )
    user_content = f"Recent session activity:\n\n{combined}"

    try:
        summary = await llm_service.call_generator_once(
            messages=[{"role": "user", "content": user_content}],
            system_prompt=system_prompt,
        )
        logger.info("Global summary refreshed (%d chars).", len(summary))
        return summary.strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Summarizer LLM call failed: %s", exc)
        return state.get("global_summary", "")


async def trigger_summary_refresh(graph, config: dict, session_id: str, websocket=None) -> None:
    """
    Helper function to generate summary in background and save to checkpointer state,
    notifying WebSocket if connected.
    """
    try:
        snapshot = graph.get_state(config)
        state = snapshot.values if (snapshot and snapshot.values) else {}
        nodes = state.get("nodes", {})
        if not nodes:
            return

        if websocket:
            try:
                from app.models.schemas import WSMessage
                await websocket.send_text(
                    WSMessage(type="summary_status", is_summarizing=True).model_dump_json(exclude_none=True)
                )
            except Exception:
                pass

        new_summary = await refresh_global_summary(state)
        if new_summary:
            graph.update_state(config, {"global_summary": new_summary})
            logger.info("Updated global_summary in state for session %s", session_id)

            if websocket:
                try:
                    from app.models.schemas import WSMessage
                    await websocket.send_text(
                        WSMessage(
                            type="summary_status",
                            is_summarizing=False,
                            global_summary=new_summary,
                        ).model_dump_json(exclude_none=True)
                    )
                except Exception:
                    pass
    except Exception as exc:
        logger.error("Failed background summary refresh: %s", exc)


async def summarizer_node(state: GraphState) -> dict:
    """
    LangGraph node: trigger an async refresh of the global summary.
    """
    turn_count: int = state.get("turn_count", 0)
    current_summary: str = state.get("global_summary", "")

    # Trigger if turn_count > 0 and (summary is empty or turn_count % 2 == 0)
    if turn_count > 0 and (not current_summary or turn_count % 2 == 0):
        logger.info("Triggering summary refresh at turn %d.", turn_count)
        # Background task can be spawned by caller / websocket handler
    return {}


# ── Per-node local summarizer (In-node bounded memory) ─────────────────


async def refresh_node_local_summary(
    node_id: str,
    node_data: dict,
    keep_last_n: int,
) -> dict:
    """
    Compress messages beyond the rolling window into node_data["local_summary"]
    and move them to node_data["archived_messages"].

    Only the last `keep_last_n` messages remain in the live "messages" list.
    All older messages are moved to "archived_messages" — they are kept for
    search and export but are NEVER sent to the LLM in future generation calls.

    Args:
        node_id:     The node being compressed (used only for logging).
        node_data:   The node's current data dict (will not be mutated).
        keep_last_n: How many recent messages to keep in the live window.

    Returns:
        Updated node_data dict with local_summary + archived_messages populated.
    """
    messages: list = list(node_data.get("messages") or [])
    existing_archived: list = list(node_data.get("archived_messages") or [])
    existing_local_summary: str = node_data.get("local_summary") or ""

    if len(messages) <= keep_last_n:
        # Nothing to archive yet
        return node_data

    # Split into overflow (to archive) + live (to keep)
    overflow = messages[:-keep_last_n]
    live = messages[-keep_last_n:]

    if not overflow:
        return node_data

    # Build a text digest for the summarizer prompt
    previous_summary_section = ""
    if existing_local_summary:
        previous_summary_section = (
            f"## Previously summarised context:\n{existing_local_summary}\n\n"
        )

    overflow_text = "\n".join(
        f"  {m['role'].upper()}: {m['content'][:300]}"
        for m in overflow
    )
    prompt_content = (
        f"{previous_summary_section}"
        f"## New messages to compress:\n{overflow_text}"
    )

    system_prompt = (
        f"You are compressing the older portion of a conversation in the topic "
        f"'{node_data.get('title', 'Unknown')}' into a running summary.\n"
        "Write a concise summary (3-6 sentences, max 150 words) that captures:\n"
        "  - The key questions asked\n"
        "  - The key answers and explanations given\n"
        "  - Any decisions, code examples, or important details covered\n"
        "This summary will be shown to the AI on future turns as 'prior context' "
        "so it must be accurate and self-contained. Be factual, not generic."
    )

    try:
        new_summary = await llm_service.call_generator_once(
            messages=[{"role": "user", "content": prompt_content}],
            system_prompt=system_prompt,
        )
        new_summary = new_summary.strip()
        logger.info(
            "Node local summary refreshed for node %s (%d chars, archived %d messages).",
            node_id, len(new_summary), len(overflow),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Node local summarizer failed for %s: %s — skipping compression.", node_id, exc)
        return node_data

    # Build updated node_data (no mutation of the original dict)
    updated = dict(node_data)
    updated["local_summary"] = new_summary
    updated["archived_messages"] = existing_archived + overflow
    updated["messages"] = live
    return updated


async def trigger_node_summary_refresh(
    graph,
    config: dict,
    node_id: str,
    keep_last_n: int,
) -> None:
    """
    Non-blocking background task: fetch latest state, compress node if needed,
    persist back to the LangGraph checkpointer.

    Mirrors trigger_summary_refresh() in pattern — fire-and-forget via
    asyncio.create_task().
    """
    try:
        snapshot = graph.get_state(config)
        state = snapshot.values if (snapshot and snapshot.values) else {}
        nodes = dict(state.get("nodes") or {})

        if node_id not in nodes:
            logger.warning("trigger_node_summary_refresh: node %r not found in state.", node_id)
            return

        node_data = nodes[node_id]
        messages = list(node_data.get("messages") or [])

        if len(messages) <= keep_last_n:
            return  # Nothing to do

        updated_node = await refresh_node_local_summary(node_id, node_data, keep_last_n)

        if updated_node is not node_data:  # Only update if compression happened
            nodes[node_id] = updated_node
            graph.update_state(config, {"nodes": nodes})
            logger.info(
                "Persisted node local summary for node %s — live msgs now %d, archived %d.",
                node_id,
                len(updated_node.get("messages") or []),
                len(updated_node.get("archived_messages") or []),
            )
    except Exception as exc:
        logger.error("trigger_node_summary_refresh failed for node %s: %s", node_id, exc)
