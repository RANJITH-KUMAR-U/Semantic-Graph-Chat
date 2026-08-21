import pytest
from app.services.text_cleaning import strip_reasoning, is_reasoning_line

def test_strip_think_tags():
    raw = "<think>Let me analyze this question carefully.\n1. It asks for Python summary.\n2. In 3 sentences.</think>Python is a high-level language. It is versatile. It has great libraries."
    cleaned = strip_reasoning(raw)
    assert cleaned == "Python is a high-level language. It is versatile. It has great libraries."

def test_strip_reasoning_tags():
    raw = "<reasoning>\nStep 1: Check topics.\nStep 2: Generate title.\n</reasoning>\nPython Basics"
    cleaned = strip_reasoning(raw)
    assert cleaned == "Python Basics"

def test_strip_thinking_process_preamble():
    raw = """Here's a thinking process:

1.  **Analyze the Request:**
    - User wants an explanation of Python in exactly 4 sentences.
    - Constraint: Exactly 4 sentences.

2.  **Drafting:**
    Python is a high-level language.

Python is an interpreted, high-level programming language with dynamic semantics. Its high-level built-in data structures make it very attractive for Rapid Application Development."""
    cleaned = strip_reasoning(raw)
    assert "Here's a thinking process:" not in cleaned
    assert "Python is an interpreted" in cleaned

def test_strip_clean_text_untouched():
    raw = "Binary Search Trees"
    cleaned = strip_reasoning(raw)
    assert cleaned == "Binary Search Trees"

def test_strip_drafting_marker():
    raw = """Let's analyze what the user is asking.
    1. The user wants a title.
    
    Final Output:
    Understanding Machine Learning"""
    cleaned = strip_reasoning(raw)
    assert cleaned == "Understanding Machine Learning"

def test_strip_numbered_check_constraints_header():
    raw = "5.  **Select the Best One:** Quantum Computing Overview"
    cleaned = strip_reasoning(raw)
    assert cleaned == "Quantum Computing Overview"

def test_strip_check_word_counts():
    raw = "Check word counts:\nQuantum Computing Basics"
    cleaned = strip_reasoning(raw)
    assert cleaned == "Quantum Computing Basics"

def test_is_reasoning_line():
    assert is_reasoning_line("Check word counts:") is True
    assert is_reasoning_line("5. **Select the Best One:**") is True
    assert is_reasoning_line("Determine What to Summarize:") is True
    assert is_reasoning_line("Quantum Computing Overview") is False
