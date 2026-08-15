"""
Tests for document chunking, retrieval, and upload endpoints.
"""
import pytest
from app.services.chunker import chunk_file, DocumentChunk, MAX_CHUNK_CHARS
from app.services.retriever import retrieve_relevant_chunks


def test_chunk_plain_text():
    sample_text = (
        "Binary Search Trees (BST) are data structures used for fast searching.\n\n"
        "Each node in a BST has at most two children: left and right.\n\n"
        "The left child contains values less than the parent, while the right child "
        "contains values greater than or equal to the parent."
    )
    bytes_data = sample_text.encode("utf-8")
    chunks = chunk_file("bst_guide.txt", bytes_data)

    assert len(chunks) >= 1
    assert chunks[0]["source_filename"] == "bst_guide.txt"
    assert chunks[0]["content_type"] == "text"
    assert "Binary Search Trees" in chunks[0]["content"]


def test_chunk_markdown():
    md_text = (
        "# Quantum Computing Basics\n\n"
        "Quantum computing harnesses quantum mechanics to solve complex problems.\n\n"
        "## Qubits\n\n"
        "Unlike classical bits which are 0 or 1, qubits can exist in superposition."
    )
    chunks = chunk_file("quantum.md", md_text.encode("utf-8"))

    assert len(chunks) >= 1
    assert chunks[0]["source_filename"] == "quantum.md"
    assert "Quantum Computing" in chunks[0]["content"]


def test_chunk_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported file type"):
        chunk_file("executable.exe", b"binary content")


def test_retriever_ranking():
    chunks = [
        {
            "chunk_id": "c1",
            "source_filename": "algorithms.txt",
            "content": "Sorting algorithms include QuickSort, MergeSort, and HeapSort.",
        },
        {
            "chunk_id": "c2",
            "source_filename": "trees.txt",
            "content": "Binary search trees allow log(n) searching when balanced.",
        },
        {
            "chunk_id": "c3",
            "source_filename": "astronomy.txt",
            "content": "Black holes are regions of space where gravity is so strong nothing escapes.",
        },
    ]

    # Query about binary trees
    results = retrieve_relevant_chunks("Tell me about binary search trees", chunks, top_k=2)
    assert len(results) >= 1
    assert results[0]["chunk_id"] == "c2"
    assert results[0]["source_filename"] == "trees.txt"

    # Query about sorting
    results_sort = retrieve_relevant_chunks("How does quicksort work?", chunks, top_k=2)
    assert len(results_sort) >= 1
    assert results_sort[0]["chunk_id"] == "c1"
