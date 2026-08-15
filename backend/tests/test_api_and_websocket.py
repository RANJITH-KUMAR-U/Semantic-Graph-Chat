"""
Integration tests for REST API endpoints and WebSocket gateway in Semantic Graph Chat.

Tests:
  1. REST Health & Root endpoints (/health, /)
  2. Session REST routes (POST /api/sessions, GET /api/sessions/{id}, GET /api/sessions/{id}/summary)
  3. Topic Node REST routes (GET /api/sessions/{id}/nodes, GET /api/nodes/{id}/messages, POST /api/nodes/{id}/force-route)
  4. WebSocket chat loop (/ws/chat/{session_id}) — connection, routing event, streaming tokens, done signal, and state persistence.
"""
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from app.main import app


# Helper token generator for mocking stream_generator
async def _mock_token_stream(text: str):
    words = text.split(" ")
    for idx, word in enumerate(words):
        space = " " if idx < len(words) - 1 else ""
        yield word + space


@pytest.fixture
def test_client():
    return TestClient(app)


# ── 1. Health & Root endpoints ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "router_model" in data
    assert "generator_model" in data


@pytest.mark.asyncio
async def test_root_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Semantic Graph Chat API" in data["message"]
    assert data["docs"] == "/docs"


# ── 2. Session REST endpoints ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_session_new():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/api/sessions", json={})
    assert response.status_code == 201
    data = response.json()
    assert "session_id" in data
    assert data["nodes"] == []
    assert data["global_summary"] == ""


@pytest.mark.asyncio
async def test_create_session_with_id():
    custom_id = f"test_session_{uuid.uuid4().hex[:8]}"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/api/sessions", json={"session_id": custom_id})
    assert response.status_code == 201
    data = response.json()
    assert data["session_id"] == custom_id


@pytest.mark.asyncio
async def test_get_session_not_found():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get(f"/api/sessions/non_existent_{uuid.uuid4().hex}")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# ── 3. Node REST endpoints ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_nodes_empty_session():
    session_id = f"empty_session_{uuid.uuid4().hex[:8]}"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get(f"/api/sessions/{session_id}/nodes")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_get_node_messages_not_found():
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get(f"/api/nodes/fake_node/messages?session_id={session_id}")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_force_route_non_existent_node():
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/api/nodes/fake_node/force-route",
            json={"session_id": session_id},
        )
    assert response.status_code == 404


# ── 4. End-to-End WebSocket & REST Integration ──────────────────────────


def test_websocket_chat_flow_and_rest_persistence(test_client):
    session_id = f"ws_session_{uuid.uuid4().hex[:8]}"

    mock_router_response = {
        "decision": "create_new",
        "target_node_id": None,
        "reasoning": "First user turn — creating Python topic node.",
    }
    mock_generated_title = "Python Async Basics"
    mock_assistant_reply = "Asyncio provides event loop capabilities in Python."

    with patch(
        "app.services.llm_service.call_router_llm",
        new=AsyncMock(return_value=mock_router_response),
    ), patch(
        "app.services.llm_service.call_generator_once",
        new=AsyncMock(return_value=mock_generated_title),
    ), patch(
        "app.services.llm_service.stream_generator",
        side_effect=lambda messages, system_prompt: _mock_token_stream(mock_assistant_reply),
    ):
        with test_client.websocket_connect(f"/ws/chat/{session_id}") as websocket:
            # 1. Server sends connected message
            msg1 = websocket.receive_json()
            assert msg1["type"] == "connected"
            assert msg1["session_id"] == session_id

            # 2. Client sends user prompt
            websocket.send_json({"content": "How does Python asyncio work?"})

            # 3. Server sends immediate 'evaluating topic' routing status
            msg2 = websocket.receive_json()
            assert msg2["type"] == "routing"
            assert "Evaluating" in msg2["content"]

            # 4. Server sends 'routing to node' status
            msg3 = websocket.receive_json()
            assert msg3["type"] == "routing"
            assert "node_id" in msg3
            created_node_id = msg3["node_id"]
            assert msg3["node_title"] == mock_generated_title

            # 5. Tokens streamed back
            tokens_received = []
            while True:
                msg = websocket.receive_json()
                if msg["type"] == "token":
                    tokens_received.append(msg["content"])
                elif msg["type"] == "done":
                    assert msg["node_id"] == created_node_id
                    break

            full_text = "".join(tokens_received)
            assert full_text == mock_assistant_reply

    # 6. Verify REST API returns updated session state
    resp_session = test_client.get(f"/api/sessions/{session_id}")
    assert resp_session.status_code == 200
    session_data = resp_session.json()
    assert len(session_data["nodes"]) == 1
    assert session_data["nodes"][0]["node_id"] == created_node_id
    assert session_data["nodes"][0]["title"] == mock_generated_title

    # 7. Verify REST API node message history
    resp_msgs = test_client.get(f"/api/nodes/{created_node_id}/messages?session_id={session_id}")
    assert resp_msgs.status_code == 200
    messages = resp_msgs.json()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "How does Python asyncio work?"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == mock_assistant_reply

    # 8. Test Force Route REST API
    resp_force = test_client.post(
        f"/api/nodes/{created_node_id}/force-route",
        json={"session_id": session_id},
    )
    assert resp_force.status_code == 200
    assert resp_force.json()["node_id"] == created_node_id

    # 9. Verify WebSocket respects force_route override for next turn
    mock_router_override = {
        "decision": "route_existing",
        "target_node_id": created_node_id,
        "reasoning": "Manual override via force_node_id.",
    }
    with patch(
        "app.services.llm_service.call_router_llm",
        new=AsyncMock(return_value=mock_router_override),
    ), patch(
        "app.services.llm_service.stream_generator",
        side_effect=lambda messages, system_prompt: _mock_token_stream("Followup response."),
    ):
        with test_client.websocket_connect(f"/ws/chat/{session_id}") as websocket:
            _ = websocket.receive_json()  # connected
            websocket.send_json({"content": "Tell me more details."})

            _ = websocket.receive_json()  # routing evaluating
            routing_done = websocket.receive_json()
            assert routing_done["type"] == "routing"
            assert routing_done["node_id"] == created_node_id

            # Consume tokens and done
            while True:
                m = websocket.receive_json()
                if m["type"] == "done":
                    break
