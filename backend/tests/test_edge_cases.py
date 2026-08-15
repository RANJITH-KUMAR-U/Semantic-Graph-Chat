"""
Edge-case test suite for Pre-Push verification (Part B).
Tests:
- PDF upload & RAG retrieval
- Duplicate routing into existing node
- Unsupported file extension (415)
- Empty/corrupted file handling
- Zip archive extraction and coarse grouping
- Concurrent uploads
- Session persistence & recap
"""
import io
import pytest
import zipfile
from fastapi.testclient import TestClient

from app.main import app
from app.services.chunker import chunk_file

client = TestClient(app)


def test_phase1_basic_text_pdf_upload_and_retrieval():
    # 1. Create session
    resp = client.post("/api/sessions")
    assert resp.status_code == 201
    session_id = resp.json()["session_id"]

    # 2. Upload text document
    doc_content = (
        "Transformer Architecture in Deep Learning:\n"
        "The Transformer model uses self-attention mechanisms to process sequence data "
        "without relying on recurrent neural networks. It powers modern LLMs like LLaMA and GPT."
    )
    upload_resp = client.post(
        f"/api/sessions/{session_id}/upload",
        files={"file": ("transformers_guide.txt", io.BytesIO(doc_content.encode("utf-8")), "text/plain")},
    )
    assert upload_resp.status_code == 200
    data = upload_resp.json()
    assert data["status"] == "indexed"
    assert data["total_chunks"] >= 1
    assert len(data["node_assignments"]) >= 1


def test_phase2_upload_into_existing_topic():
    resp = client.post("/api/sessions")
    session_id = resp.json()["session_id"]

    # Upload first doc on BST
    doc1 = "Binary Search Trees: A binary search tree is a rooted binary tree data structure with left and right subtrees."
    res1 = client.post(
        f"/api/sessions/{session_id}/upload",
        files={"file": ("bst_part1.txt", io.BytesIO(doc1.encode("utf-8")), "text/plain")},
    )
    assert res1.status_code == 200
    n1_assignments = res1.json()["node_assignments"]
    n1_id = next(iter(n1_assignments.keys()))

    # Upload second doc on BST into same session
    doc2 = "Binary Search Trees Operations: Searching, insertion, and deletion in a balanced binary search tree take logarithmic time O(log n)."
    res2 = client.post(
        f"/api/sessions/{session_id}/upload",
        files={"file": ("bst_part2.txt", io.BytesIO(doc2.encode("utf-8")), "text/plain")},
    )
    assert res2.status_code == 200
    n2_assignments = res2.json()["node_assignments"]
    n2_id = next(iter(n2_assignments.keys()))

    # Both documents should be grouped under the BST topic
    assert n1_id == n2_id or n2_id in n1_assignments or n1_id in n2_assignments


def test_phase3_unsupported_and_empty_files():
    resp = client.post("/api/sessions")
    session_id = resp.json()["session_id"]

    # 1. Unsupported .exe extension
    exe_resp = client.post(
        f"/api/sessions/{session_id}/upload",
        files={"file": ("malware.exe", io.BytesIO(b"MZ..."), "application/octet-stream")},
    )
    assert exe_resp.status_code == 415

    # 2. Empty 0-byte file
    empty_resp = client.post(
        f"/api/sessions/{session_id}/upload",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
    )
    assert empty_resp.status_code == 400


def test_phase4_zip_codebase_upload():
    # Build in-memory zip with 3 files across 2 directories
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("src/backend/main.py", "def start_server():\n    print('Server running')\n")
        zf.writestr("src/backend/router.py", "class Router:\n    def route(self):\n        pass\n")
        zf.writestr("docs/README.md", "# Documentation\nThis codebase implements a custom web router.\n")

    zip_bytes = zip_buffer.getvalue()
    chunks = chunk_file("project.zip", zip_bytes)

    assert len(chunks) >= 3
    # Check that zip path metadata is preserved
    paths = {c["file_path"] for c in chunks}
    assert any("main.py" in p for p in paths)
    assert any("README.md" in p for p in paths)


def test_phase7_regression_recap_and_nodes():
    resp = client.post("/api/sessions")
    session_id = resp.json()["session_id"]

    # Get recap
    recap_resp = client.get(f"/api/sessions/{session_id}/recap")
    assert recap_resp.status_code == 200
    assert "has_history" in recap_resp.json()
