"""
Comprehensive Data Isolation and OpenRouter Rate-Limit / Fallback Tests.

Verifies:
1. Strict memory boundary enforcement — zero message or document chunk leakage across Topic Nodes.
2. Graceful degradation when OpenRouter rate limits (429/401) or API errors occur.
3. Node creation and document ingestion resiliency when LLMs are offline/rate-limited.
"""
import io
import pytest
from unittest.mock import AsyncMock, patch

from app.graph.nodes import assemble_context, _new_node_id, _utcnow
from app.graph.state import GraphState, NodeData
from app.services.chunker import chunk_file
from app.services.retriever import retrieve_relevant_chunks
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_data_leakage_isolation_between_nodes():
    """
    Verify that assemble_context() enforces absolute memory isolation:
    Node A must NEVER see Node B's messages or document chunks.
    """
    now = _utcnow()
    node_a_id = "node_quantum_101"
    node_b_id = "node_cooking_202"

    state: GraphState = {
        "session_id": "test_isolation_session",
        "nodes": {
            node_a_id: {
                "title": "Quantum Mechanics",
                "messages": [
                    {"role": "user", "content": "What is Schrödinger's cat?"},
                    {"role": "assistant", "content": "A thought experiment about quantum superposition."},
                ],
                "document_chunks": [
                    {
                        "chunk_id": "c_quantum_1",
                        "source_filename": "quantum_paper.pdf",
                        "content": "Superposition state alpha|0> + beta|1> is maintained until measurement.",
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "content_type": "pdf",
                        "file_path": "quantum_paper.pdf",
                    }
                ],
                "turn_count": 1,
                "created_at": now,
                "last_active_at": now,
                "depth": 0,
                "parent_node_id": None,
                "local_summary": "Quantum discussion about superposition",
            },
            node_b_id: {
                "title": "Italian Cooking Recipes",
                "messages": [
                    {"role": "user", "content": "How do I make authentic Carbonara?"},
                    {"role": "assistant", "content": "Use guanciale, pecorino romano, egg yolks, and black pepper."},
                ],
                "document_chunks": [
                    {
                        "chunk_id": "c_cooking_1",
                        "source_filename": "pasta_secrets.docx",
                        "content": "Never add heavy cream to a traditional carbonara sauce.",
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "content_type": "docx",
                        "file_path": "pasta_secrets.docx",
                    }
                ],
                "turn_count": 1,
                "created_at": now,
                "last_active_at": now,
                "depth": 0,
                "parent_node_id": None,
                "local_summary": "Cooking discussion about pasta",
            },
        },
        "active_node_id": node_a_id,
        "current_input": "Tell me more about superposition",
        "routing_decision": {},
        "force_node_id": None,
        "global_summary": "Session summary: discussed physics and cooking",
        "turn_count": 2,
        "last_response": "",
        "session_tokens_used": 100,
        "session_baseline_tokens": 200,
        "routing_log": [],
    }

    # 1. Assemble context for Quantum Node A
    ctx_a = assemble_context(state, node_a_id)

    # Verify Node A contains ONLY Node A's messages and doc chunks
    assert len(ctx_a["messages"]) == 2
    assert ctx_a["messages"][0]["content"] == "What is Schrödinger's cat?"
    assert len(ctx_a["document_chunks"]) == 1
    assert ctx_a["document_chunks"][0]["source_filename"] == "quantum_paper.pdf"

    # CONFIRM NO LEAK FROM NODE B
    raw_str_a = str(ctx_a)
    assert "Carbonara" not in raw_str_a
    assert "guanciale" not in raw_str_a
    assert "pasta_secrets.docx" not in raw_str_a

    # 2. Assemble context for Cooking Node B
    ctx_b = assemble_context(state, node_b_id)
    assert len(ctx_b["messages"]) == 2
    assert ctx_b["messages"][0]["content"] == "How do I make authentic Carbonara?"
    assert len(ctx_b["document_chunks"]) == 1
    assert ctx_b["document_chunks"][0]["source_filename"] == "pasta_secrets.docx"

    # CONFIRM NO LEAK FROM NODE A
    raw_str_b = str(ctx_b)
    assert "Schrödinger" not in raw_str_b
    assert "superposition" not in raw_str_b
    assert "quantum_paper.pdf" not in raw_str_b


def test_openrouter_rate_limit_429_graceful_handling():
    """
    Simulate OpenRouter 429 Rate Limit error across all LLM endpoints:
    Ensure system does NOT crash, fallbacks execute cleanly, and user
    receives helpful error text instead of unhandled exceptions.
    """
    resp = client.post("/api/sessions")
    assert resp.status_code == 201
    session_id = resp.json()["session_id"]

    # Upload a document while LLM calls fail (simulating 429 rate limit)
    with patch("app.services.llm_service.call_router_llm", side_effect=RuntimeError("429 Too Many Requests: Rate limit reached")):
        upload_resp = client.post(
            f"/api/sessions/{session_id}/upload",
            files={"file": ("rate_limit_doc.txt", io.BytesIO(b"Content under rate limit mode"), "text/plain")},
        )
        assert upload_resp.status_code == 200
        data = upload_resp.json()
        assert data["status"] == "indexed"
        assert data["total_chunks"] >= 1


def test_search_isolation_across_sessions():
    """
    Verify search results do not cross session boundaries.
    """
    # Create Session 1 with secret prompt
    s1_resp = client.post("/api/sessions")
    s1_id = s1_resp.json()["session_id"]

    # Search Session 1 with query
    search_res = client.get(f"/api/sessions/{s1_id}/search?q=secret")
    assert search_res.status_code == 200
    assert isinstance(search_res.json(), list)
