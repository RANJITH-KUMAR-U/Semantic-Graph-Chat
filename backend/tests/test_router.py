"""
Tests for the Semantic Graph Chat backend.

Month 1 milestone (PRD section 9 / AGENTS.md "Definition of done"):
  Prove memory isolation — no cross-node context leakage.

Test suite:
  1. test_memory_isolation            — messages written to Node A must NOT
                                        appear when assembling context for Node B.
  2. test_router_returns_existing_node — follow-up on same topic routes back
                                        to the same node_id.
  3. test_router_creates_new_node      — genuinely new topic spawns a new node.
  4. test_assemble_context_missing_node — KeyError on invalid node_id.
  5. test_routing_decision_schema      — RoutingDecision validates correctly.
  6. test_force_route_override         — force_node_id bypasses the LLM call.

Run with:
    cd backend
    pytest tests/test_router.py -v
"""
import asyncio
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.graph.nodes import assemble_context, create_node_node, generate_node
from app.graph.router import RoutingDecision, router_node
from app.graph.state import GraphState, NodeData


# ── Test fixtures ──────────────────────────────────────────────────────


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_node(title: str, messages: list | None = None) -> NodeData:
    """Helper: build a NodeData dict for a given title."""
    return {
        "title": title,
        "messages": messages or [],
        "turn_count": len(messages or []) // 2,
        "created_at": _utcnow(),
        "last_active_at": _utcnow(),
    }


def _make_state(
    nodes: dict | None = None,
    active_node_id: str = "",
    current_input: str = "Hello",
    force_node_id: str | None = None,
    global_summary: str = "",
    turn_count: int = 0,
) -> GraphState:
    """Helper: build a minimal GraphState for testing."""
    return {
        "session_id": str(uuid.uuid4()),
        "nodes": nodes or {},
        "active_node_id": active_node_id,
        "current_input": current_input,
        "routing_decision": {},
        "force_node_id": force_node_id,
        "global_summary": global_summary,
        "turn_count": turn_count,
        "last_response": "",
    }


# ── 1. Memory isolation ────────────────────────────────────────────────


class TestMemoryIsolation:
    """
    Core correctness tests — the most critical invariant of the system.

    AGENTS.md: "Never let one Topic Node's LLM call see another node's messages."
    """

    def test_assemble_context_returns_only_target_node_messages(self):
        """
        Context assembled for node_a must NOT contain any messages from node_b.
        """
        node_a_id = "node_aaaa1111"
        node_b_id = "node_bbbb2222"

        node_a_messages = [
            {"role": "user", "content": "Tell me about Python async."},
            {"role": "assistant", "content": "Python async uses the asyncio library..."},
        ]
        node_b_messages = [
            {"role": "user", "content": "Explain React hooks."},
            {"role": "assistant", "content": "React hooks let you use state in functions..."},
        ]

        state = _make_state(
            nodes={
                node_a_id: _make_node("Python Async", node_a_messages),
                node_b_id: _make_node("React Hooks", node_b_messages),
            },
            active_node_id=node_a_id,
            global_summary="User is building a web app with Python backend and React frontend.",
        )

        # Assemble context for node_a
        ctx_a = assemble_context(state, node_a_id)
        # Assemble context for node_b
        ctx_b = assemble_context(state, node_b_id)

        # ── Isolation assertions ───────────────────────────────────────
        # node_a context must not contain node_b messages
        a_contents = {m["content"] for m in ctx_a["messages"]}
        b_contents = {m["content"] for m in ctx_b["messages"]}

        for b_msg in node_b_messages:
            assert b_msg["content"] not in a_contents, (
                f"ISOLATION BREACH: node_b message found in node_a context!\n"
                f"  Message: {b_msg['content']!r}"
            )

        for a_msg in node_a_messages:
            assert a_msg["content"] not in b_contents, (
                f"ISOLATION BREACH: node_a message found in node_b context!\n"
                f"  Message: {a_msg['content']!r}"
            )

        # Each context should have exactly the right messages
        assert len(ctx_a["messages"]) == len(node_a_messages)
        assert len(ctx_b["messages"]) == len(node_b_messages)

    def test_global_summary_is_shared_not_raw_messages(self):
        """
        The global_summary string is the same across both contexts.
        Raw messages from other nodes are NOT included.
        """
        global_summary = "Session covers: Python async (node A) and React hooks (node B)."
        node_a_id = "node_aaaa1111"
        node_b_id = "node_bbbb2222"

        state = _make_state(
            nodes={
                node_a_id: _make_node("Python Async", [{"role": "user", "content": "What is asyncio?"}]),
                node_b_id: _make_node("React Hooks", [{"role": "user", "content": "What are hooks?"}]),
            },
            global_summary=global_summary,
        )

        ctx_a = assemble_context(state, node_a_id)
        ctx_b = assemble_context(state, node_b_id)

        assert ctx_a["global_summary"] == global_summary
        assert ctx_b["global_summary"] == global_summary

        # The raw messages from node_b must NOT appear in node_a's message list
        a_all_content = " ".join(m["content"] for m in ctx_a["messages"])
        assert "What are hooks?" not in a_all_content

    def test_assemble_context_missing_node_raises(self):
        """KeyError when assembling context for a non-existent node."""
        state = _make_state(nodes={"node_real": _make_node("Real Node")})
        with pytest.raises(KeyError, match="node_fake"):
            assemble_context(state, "node_fake")

    @pytest.mark.asyncio
    async def test_generate_node_persists_to_active_node_only(self):
        """
        After generate_node runs, the assistant response must appear in the
        active node's messages and NOWHERE ELSE.
        """
        node_a_id = "node_aaaa1111"
        node_b_id = "node_bbbb2222"
        state = _make_state(
            nodes={
                node_a_id: _make_node("Database Design", []),
                node_b_id: _make_node("React UI", []),
            },
            active_node_id=node_a_id,
            current_input="How do I design a users table?",
        )
        state["routing_decision"] = {
            "decision": "route_existing",
            "target_node_id": node_a_id,
            "reasoning": "Database question.",
        }

        fake_response = "A users table typically has id, email, created_at columns."

        with patch(
            "app.graph.nodes.llm_service.stream_generator",
            return_value=_async_token_gen(fake_response),
        ):
            updates = await generate_node(state)

        updated_nodes = updates["nodes"]

        # node_a should have the new messages
        node_a_messages = updated_nodes[node_a_id]["messages"]
        assert any(m["content"] == fake_response for m in node_a_messages), (
            "Assistant response not found in active node's messages."
        )

        # node_b should be completely unchanged
        node_b_messages = updated_nodes[node_b_id]["messages"]
        assert len(node_b_messages) == 0, (
            f"ISOLATION BREACH: node_b gained messages after writing to node_a!\n"
            f"  node_b messages: {node_b_messages}"
        )
        assert not any(m["content"] == fake_response for m in node_b_messages), (
            "ISOLATION BREACH: assistant response leaked into node_b!"
        )


# ── 2. Router returns existing node ───────────────────────────────────


class TestRouterExistingNode:
    """Router should return the same node_id for a follow-up on the same topic."""

    @pytest.mark.asyncio
    async def test_router_routes_to_existing_node_on_same_topic(self):
        """
        When there is one active node and the message is clearly related,
        the router must return route_existing with that node's ID.
        """
        node_id = "node_db001"
        state = _make_state(
            nodes={node_id: _make_node("Database Design")},
            active_node_id=node_id,
            current_input="Add a last_login column to the users table.",
        )

        fake_router_response = {
            "decision": "route_existing",
            "target_node_id": node_id,
            "reasoning": "Message is about database schema, matching existing 'Database Design' node.",
        }

        with patch(
            "app.graph.router.llm_service.call_router_llm",
            new_callable=AsyncMock,
            return_value=fake_router_response,
        ):
            result = await router_node(state)
        decision = result["routing_decision"]

        assert decision["decision"] == "route_existing"
        assert decision["target_node_id"] == node_id

    @pytest.mark.asyncio
    async def test_router_no_nodes_always_creates(self):
        """With no active nodes, router must always return create_new."""
        state = _make_state(nodes={}, current_input="What is the capital of France?")

        result = await router_node(state)
        decision = result["routing_decision"]
        assert decision["decision"] == "create_new"
        assert decision["target_node_id"] is None


# ── 3. Router creates new node ─────────────────────────────────────────


class TestRouterCreatesNewNode:
    """Router must spawn a new node when the topic is genuinely novel."""

    @pytest.mark.asyncio
    async def test_router_creates_new_node_for_different_topic(self):
        """After discussing databases, a React question should create a new node."""
        node_id = "node_db001"
        state = _make_state(
            nodes={node_id: _make_node("Database Design")},
            active_node_id=node_id,
            current_input="How do I set up React Router v6?",
        )

        fake_router_response = {
            "decision": "create_new",
            "target_node_id": None,
            "reasoning": "React Router is a frontend topic, unrelated to database design.",
        }

        with patch(
            "app.graph.router.llm_service.call_router_llm",
            new_callable=AsyncMock,
            return_value=fake_router_response,
        ):
            result = await router_node(state)
        decision = result["routing_decision"]

        assert decision["decision"] == "create_new"
        assert decision["target_node_id"] is None

    @pytest.mark.asyncio
    async def test_create_node_generates_unique_id(self):
        """Each call to create_node_node should produce a different node_id."""
        state1 = _make_state(current_input="Explain SQL joins.")
        state2 = _make_state(current_input="Explain React state.")

        fake_title = "Test Topic"

        with patch(
            "app.graph.nodes.llm_service.call_generator_once",
            new_callable=AsyncMock,
            return_value=fake_title,
        ):
            u1 = await create_node_node(state1)
            u2 = await create_node_node(state2)

        node_id_1 = u1["active_node_id"]
        node_id_2 = u2["active_node_id"]
        assert node_id_1 != node_id_2, "Two separate create_node calls produced the same node_id!"


# ── 4. RoutingDecision schema ──────────────────────────────────────────


class TestRoutingDecisionSchema:
    """RoutingDecision Pydantic model validates correctly."""

    def test_valid_route_existing(self):
        d = RoutingDecision(
            decision="route_existing",
            target_node_id="node_abc",
            reasoning="Test reason",
        )
        assert d.decision == "route_existing"
        assert d.target_node_id == "node_abc"

    def test_valid_create_new(self):
        d = RoutingDecision(
            decision="create_new",
            target_node_id=None,
            reasoning="New topic detected.",
        )
        assert d.decision == "create_new"
        assert d.target_node_id is None

    def test_invalid_decision_raises(self):
        with pytest.raises(Exception):
            RoutingDecision(
                decision="unknown_decision",  # type: ignore[arg-type]
                target_node_id=None,
                reasoning="Bad value",
            )


# ── 5. Force-route override ────────────────────────────────────────────


class TestForceRouteOverride:
    """force_node_id must bypass the LLM entirely."""

    @pytest.mark.asyncio
    async def test_force_node_skips_llm(self):
        """When force_node_id is set, the router must NOT call the LLM."""
        node_id = "node_forced"
        state = _make_state(
            nodes={node_id: _make_node("Forced Node")},
            active_node_id="node_other",
            current_input="This could go anywhere.",
            force_node_id=node_id,
        )

        with patch(
            "app.graph.router.llm_service.call_router_llm",
            new_callable=AsyncMock,
        ) as mock_llm:
            result = await router_node(state)
            # LLM must NOT have been called
            mock_llm.assert_not_called()

        decision = result["routing_decision"]

        assert decision["decision"] == "route_existing"
        assert decision["target_node_id"] == node_id
        # force_node_id should be consumed (set to None)
        assert result.get("force_node_id") is None


# ── Async helpers ──────────────────────────────────────────────────────


async def _async_token_gen(text: str):
    """Async generator that yields a full string as a single 'token'."""
    yield text
