"""
Type-aware document chunking service.

Splits uploaded files into chunks suitable for retrieval-augmented
generation (RAG). Each file type gets a specialised strategy:

    .txt / .md  → recursive paragraph → sentence split
    .pdf        → page-level extraction via pypdf, then recursive split
    .docx       → paragraph extraction via python-docx, then recursive split
    .zip        → extract → group by directory → per-file chunking
    Code files  → split on function/class boundaries

Every chunk carries source metadata so citations can trace back to the
original file and location.

# SECURITY-TODO: Validate file contents match declared type (magic bytes).
# SECURITY-TODO: Sanitize filenames to prevent path-traversal attacks.
"""
from __future__ import annotations

import io
import logging
import re
import uuid
import zipfile
from typing import TypedDict

logger = logging.getLogger(__name__)

# ── Chunk metadata type ────────────────────────────────────────────────

class DocumentChunk(TypedDict):
    """A single chunk produced by the chunking pipeline."""
    chunk_id: str           # unique ID
    source_filename: str    # original filename (or path within zip)
    content: str            # chunk text
    chunk_index: int        # position within file
    total_chunks: int       # total chunks from this file
    content_type: str       # "text" | "pdf" | "docx" | "code"
    file_path: str          # full path (useful for zips)


# ── Constants ──────────────────────────────────────────────────────────

MAX_CHUNK_CHARS = 3200          # ~800 tokens at 4 chars/token
OVERLAP_CHARS = 200             # overlap between consecutive chunks
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_ZIP_UNCOMPRESSED = 50 * 1024 * 1024  # 50 MB

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".zip"}

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
    ".kt", ".scala", ".sh", ".bash", ".yaml", ".yml", ".json",
    ".toml", ".cfg", ".ini", ".xml", ".html", ".css", ".scss",
}

# Files to skip inside zips
SKIP_PATTERNS = {
    "__pycache__", "node_modules", ".git", ".svn", ".hg",
    ".DS_Store", "Thumbs.db", ".env",
}


# ── Public API ─────────────────────────────────────────────────────────


def chunk_file(filename: str, file_bytes: bytes) -> list[DocumentChunk]:
    """
    Main entry point: chunk a file based on its extension.

    Args:
        filename: Original filename (used to determine strategy).
        file_bytes: Raw file content.

    Returns:
        List of DocumentChunk dicts ready for storage and retrieval.

    Raises:
        ValueError: If the file type is unsupported or the file is too large.
    """
    ext = _get_extension(filename)

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if len(file_bytes) > MAX_FILE_SIZE:
        raise ValueError(
            f"File too large: {len(file_bytes) / (1024*1024):.1f}MB "
            f"(max {MAX_FILE_SIZE / (1024*1024):.0f}MB)"
        )

    logger.info("Chunking %r (%s, %.1f KB)", filename, ext, len(file_bytes) / 1024)

    if ext == ".pdf":
        return _chunk_pdf(filename, file_bytes)
    elif ext == ".docx":
        return _chunk_docx(filename, file_bytes)
    elif ext == ".zip":
        return _chunk_zip(filename, file_bytes)
    else:
        # .txt, .md — plain text
        text = file_bytes.decode("utf-8", errors="replace")
        return _chunk_text(filename, text, content_type="text")


# ── PDF chunking ───────────────────────────────────────────────────────


def _chunk_pdf(filename: str, file_bytes: bytes) -> list[DocumentChunk]:
    """Extract text page-by-page using pypdf, then recursive-split each page."""
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.error("pypdf not installed — cannot parse PDF files")
        raise ValueError("PDF parsing requires the 'pypdf' package. Install with: pip install pypdf")

    reader = PdfReader(io.BytesIO(file_bytes))
    all_chunks: list[DocumentChunk] = []

    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        page_text = page_text.strip()
        if not page_text:
            continue

        # Prefix page number for citation context
        prefixed = f"[Page {page_num + 1}] {page_text}"
        page_chunks = _recursive_split(prefixed, MAX_CHUNK_CHARS, OVERLAP_CHARS)

        for i, chunk_text in enumerate(page_chunks):
            all_chunks.append(DocumentChunk(
                chunk_id=f"chunk_{uuid.uuid4().hex[:8]}",
                source_filename=filename,
                content=chunk_text,
                chunk_index=len(all_chunks),
                total_chunks=0,  # filled after loop
                content_type="pdf",
                file_path=f"{filename}#page{page_num + 1}",
            ))

    # Backfill total_chunks
    for c in all_chunks:
        c["total_chunks"] = len(all_chunks)

    logger.info("PDF %r → %d chunks from %d pages", filename, len(all_chunks), len(reader.pages))
    return all_chunks


# ── DOCX chunking ──────────────────────────────────────────────────────


def _chunk_docx(filename: str, file_bytes: bytes) -> list[DocumentChunk]:
    """Extract paragraphs from a .docx file using python-docx."""
    try:
        from docx import Document
    except ImportError:
        logger.error("python-docx not installed — cannot parse DOCX files")
        raise ValueError("DOCX parsing requires 'python-docx'. Install with: pip install python-docx")

    doc = Document(io.BytesIO(file_bytes))
    full_text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

    if not full_text.strip():
        return []

    return _chunk_text(filename, full_text, content_type="docx")


# ── ZIP chunking (coarse grouping) ─────────────────────────────────────


def _chunk_zip(filename: str, file_bytes: bytes) -> list[DocumentChunk]:
    """
    Extract a ZIP, group by top-level directory, and chunk each file.

    Coarse grouping: files are batched by their top-level directory to
    avoid node explosion. Each directory group gets routed as one unit.
    """
    all_chunks: list[DocumentChunk] = []

    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            # Safety check: total uncompressed size
            total_size = sum(info.file_size for info in zf.infolist() if not info.is_dir())
            if total_size > MAX_ZIP_UNCOMPRESSED:
                raise ValueError(
                    f"ZIP uncompressed size {total_size / (1024*1024):.1f}MB "
                    f"exceeds limit of {MAX_ZIP_UNCOMPRESSED / (1024*1024):.0f}MB"
                )

            # Group files by top-level directory
            dir_groups: dict[str, list[zipfile.ZipInfo]] = {}
            for info in zf.infolist():
                if info.is_dir():
                    continue
                # Skip binary/hidden/cache files
                if _should_skip_zip_entry(info.filename):
                    continue

                parts = info.filename.split("/")
                group_key = parts[0] if len(parts) > 1 else "(root)"
                dir_groups.setdefault(group_key, []).append(info)

            # Process each directory group
            for group_name, entries in dir_groups.items():
                # Large directory sampling: cap at 20 files + README/docs
                if len(entries) > 50:
                    priority = [e for e in entries if _is_priority_file(e.filename)]
                    remaining = [e for e in entries if e not in priority]
                    entries = priority + remaining[:20]
                    logger.info(
                        "ZIP group %r has %d files — sampled %d",
                        group_name, len(entries), len(priority) + min(20, len(remaining))
                    )

                for info in entries:
                    try:
                        raw = zf.read(info.filename)
                        ext = _get_extension(info.filename)

                        if ext in CODE_EXTENSIONS:
                            text = raw.decode("utf-8", errors="replace")
                            file_chunks = _chunk_code(info.filename, text)
                        elif ext in {".txt", ".md"}:
                            text = raw.decode("utf-8", errors="replace")
                            file_chunks = _chunk_text(info.filename, text, content_type="text")
                        elif ext == ".pdf":
                            file_chunks = _chunk_pdf(info.filename, raw)
                        elif ext == ".docx":
                            file_chunks = _chunk_docx(info.filename, raw)
                        else:
                            # Try as text, skip if binary
                            try:
                                text = raw.decode("utf-8")
                                file_chunks = _chunk_text(info.filename, text, content_type="text")
                            except UnicodeDecodeError:
                                logger.debug("Skipping binary file in zip: %s", info.filename)
                                continue

                        # Tag chunks with zip path
                        for c in file_chunks:
                            c["file_path"] = f"{filename}/{info.filename}"

                        all_chunks.extend(file_chunks)
                    except Exception as exc:
                        logger.warning("Failed to chunk %s in zip: %s", info.filename, exc)
                        continue

    except zipfile.BadZipFile:
        raise ValueError("Invalid or corrupted ZIP file")

    logger.info("ZIP %r → %d total chunks", filename, len(all_chunks))
    return all_chunks


# ── Code-aware chunking ────────────────────────────────────────────────


def _chunk_code(filename: str, text: str) -> list[DocumentChunk]:
    """
    Split code files on function/class boundaries.

    For Python files (.py): uses ast.parse() for exact, syntax-aware
    splitting at function/class boundaries.  Falls back to regex on
    SyntaxError.

    For all other languages: uses a regex heuristic to split on lines
    that start with `def `, `class `, `function `, `export `, etc.
    Falls back to line-window splitting if no boundaries are found.
    """
    if not text.strip():
        return []

    # ── Python-specific AST path ──────────────────────────────────────
    ext = _get_extension(filename)
    if ext == ".py":
        try:
            ast_chunks = _chunk_python_ast(filename, text)
            if ast_chunks:
                return ast_chunks
            # Empty result (e.g. script with no top-level defs) → fall through
        except SyntaxError:
            logger.debug(
                "AST parse failed for %s — falling back to regex chunking",
                filename,
            )

    # ── Regex-based path (all non-Python files, or AST fallback) ──────
    boundary_pattern = re.compile(
        r"^(?:def |async def |class |function |export (?:default )?(?:function |class |const |let ))",
        re.MULTILINE,
    )

    boundaries = [m.start() for m in boundary_pattern.finditer(text)]

    if len(boundaries) >= 2:
        segments: list[str] = []
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
            segment = text[start:end].strip()
            if segment:
                segments.append(segment)

        # Add any preamble (imports, comments before first definition)
        if boundaries[0] > 0:
            preamble = text[:boundaries[0]].strip()
            if preamble:
                segments.insert(0, preamble)
    else:
        # Fallback: split by line windows
        segments = _split_by_lines(text, max_lines=80)

    # Now split any oversized segments
    all_chunks: list[DocumentChunk] = []
    for segment in segments:
        if len(segment) > MAX_CHUNK_CHARS:
            sub_chunks = _recursive_split(segment, MAX_CHUNK_CHARS, OVERLAP_CHARS)
            for sc in sub_chunks:
                all_chunks.append(DocumentChunk(
                    chunk_id=f"chunk_{uuid.uuid4().hex[:8]}",
                    source_filename=filename,
                    content=sc,
                    chunk_index=len(all_chunks),
                    total_chunks=0,
                    content_type="code",
                    file_path=filename,
                ))
        else:
            all_chunks.append(DocumentChunk(
                chunk_id=f"chunk_{uuid.uuid4().hex[:8]}",
                source_filename=filename,
                content=segment,
                chunk_index=len(all_chunks),
                total_chunks=0,
                content_type="code",
                file_path=filename,
            ))

    for c in all_chunks:
        c["total_chunks"] = len(all_chunks)

    return all_chunks


# ── AST-based Python chunking ─────────────────────────────────────────


def _chunk_python_ast(filename: str, text: str) -> list[DocumentChunk]:
    """
    Split a Python source file using the `ast` module for exact,
    syntax-aware boundaries.

    Walks top-level nodes in the parsed AST and slices source lines
    using each node's `.lineno` and `.end_lineno`.  This correctly
    handles strings/docstrings that contain text like ``def fake()``
    — unlike regex, the AST parser knows these are string literals,
    not real function definitions.

    Falls back by raising SyntaxError if the source cannot be parsed.

    Returns an empty list if the file has no top-level function/class
    definitions (the caller should fall through to regex/line-window
    splitting in that case).
    """
    import ast as _ast

    tree = _ast.parse(text, filename=filename)
    source_lines = text.splitlines(keepends=True)

    # Collect top-level function and class definitions
    definition_nodes: list[_ast.AST] = []
    for node in _ast.iter_child_nodes(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
            definition_nodes.append(node)

    if not definition_nodes:
        return []  # No definitions → let caller fall through

    # Sort by line number (should already be ordered, but be safe)
    definition_nodes.sort(key=lambda n: n.lineno)

    segments: list[str] = []

    # Preamble: everything before the first definition (imports, module docstring, etc.)
    first_def_line = definition_nodes[0].lineno  # 1-indexed
    if first_def_line > 1:
        preamble = "".join(source_lines[: first_def_line - 1]).strip()
        if preamble:
            segments.append(preamble)

    # Each definition as its own chunk
    for node in definition_nodes:
        start_line = node.lineno - 1      # convert to 0-indexed
        end_line = node.end_lineno         # end_lineno is 1-indexed, inclusive → use as exclusive slice end
        if end_line is None:
            # Shouldn't happen with Python 3.8+ but handle gracefully
            end_line = len(source_lines)
        segment = "".join(source_lines[start_line:end_line]).strip()
        if segment:
            segments.append(segment)

    # Epilogue: anything after the last definition (top-level statements, if __name__ blocks)
    last_end = definition_nodes[-1].end_lineno or len(source_lines)
    if last_end < len(source_lines):
        epilogue = "".join(source_lines[last_end:]).strip()
        if epilogue:
            segments.append(epilogue)

    # Build DocumentChunks, splitting oversized segments
    all_chunks: list[DocumentChunk] = []
    for segment in segments:
        if len(segment) > MAX_CHUNK_CHARS:
            sub_chunks = _recursive_split(segment, MAX_CHUNK_CHARS, OVERLAP_CHARS)
            for sc in sub_chunks:
                all_chunks.append(DocumentChunk(
                    chunk_id=f"chunk_{uuid.uuid4().hex[:8]}",
                    source_filename=filename,
                    content=sc,
                    chunk_index=len(all_chunks),
                    total_chunks=0,
                    content_type="code",
                    file_path=filename,
                ))
        else:
            all_chunks.append(DocumentChunk(
                chunk_id=f"chunk_{uuid.uuid4().hex[:8]}",
                source_filename=filename,
                content=segment,
                chunk_index=len(all_chunks),
                total_chunks=0,
                content_type="code",
                file_path=filename,
            ))

    for c in all_chunks:
        c["total_chunks"] = len(all_chunks)

    logger.info(
        "AST-chunked Python file %r → %d chunks from %d definitions",
        filename, len(all_chunks), len(definition_nodes),
    )
    return all_chunks



# ── Generic text chunking ──────────────────────────────────────────────


def _chunk_text(
    filename: str,
    text: str,
    content_type: str = "text",
) -> list[DocumentChunk]:
    """Recursive split for plain text files."""
    if not text.strip():
        return []

    segments = _recursive_split(text, MAX_CHUNK_CHARS, OVERLAP_CHARS)
    chunks: list[DocumentChunk] = []

    for i, segment in enumerate(segments):
        chunks.append(DocumentChunk(
            chunk_id=f"chunk_{uuid.uuid4().hex[:8]}",
            source_filename=filename,
            content=segment,
            chunk_index=i,
            total_chunks=len(segments),
            content_type=content_type,
            file_path=filename,
        ))

    return chunks


# ── Internal helpers ───────────────────────────────────────────────────


def _recursive_split(
    text: str,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap: int = OVERLAP_CHARS,
) -> list[str]:
    """
    Split text into chunks of at most `max_chars`, trying paragraph → sentence
    → word boundaries before hard-cutting.
    """
    if len(text) <= max_chars:
        return [text.strip()] if text.strip() else []

    # Try splitting by double newline (paragraphs)
    parts = re.split(r"\n\n+", text)
    if len(parts) > 1:
        return _merge_splits(parts, max_chars, overlap)

    # Try splitting by single newline
    parts = text.split("\n")
    if len(parts) > 1:
        return _merge_splits(parts, max_chars, overlap)

    # Try splitting by sentence
    parts = re.split(r"(?<=[.!?])\s+", text)
    if len(parts) > 1:
        return _merge_splits(parts, max_chars, overlap)

    # Hard split by character
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end].strip())
        start = end - overlap if end < len(text) else end
    return [c for c in chunks if c]


def _merge_splits(
    parts: list[str],
    max_chars: int,
    overlap: int,
) -> list[str]:
    """Merge small splits into chunks that respect max_chars."""
    chunks: list[str] = []
    current = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if current and len(current) + len(part) + 2 > max_chars:
            chunks.append(current.strip())
            # Keep overlap from end of current chunk
            if overlap > 0 and len(current) > overlap:
                current = current[-overlap:] + "\n\n" + part
            else:
                current = part
        else:
            current = current + "\n\n" + part if current else part

    if current.strip():
        chunks.append(current.strip())

    # Recursively split any chunks that are still too large
    result: list[str] = []
    for chunk in chunks:
        if len(chunk) > max_chars:
            result.extend(_recursive_split(chunk, max_chars, overlap))
        else:
            result.append(chunk)

    return result


def _split_by_lines(text: str, max_lines: int = 80) -> list[str]:
    """Split text into segments of at most max_lines lines."""
    lines = text.split("\n")
    segments = []
    for i in range(0, len(lines), max_lines):
        segment = "\n".join(lines[i:i + max_lines]).strip()
        if segment:
            segments.append(segment)
    return segments


def _get_extension(filename: str) -> str:
    """Get lowercase file extension."""
    import os
    _, ext = os.path.splitext(filename.lower())
    return ext


def _should_skip_zip_entry(filepath: str) -> bool:
    """Return True if this zip entry should be skipped."""
    parts = filepath.split("/")
    for part in parts:
        if part in SKIP_PATTERNS or part.startswith("."):
            return True
    # Skip very large binary extensions
    ext = _get_extension(filepath)
    binary_exts = {
        ".exe", ".dll", ".so", ".dylib", ".o", ".a",
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
        ".mp3", ".mp4", ".avi", ".mov", ".wav",
        ".woff", ".woff2", ".ttf", ".eot",
        ".pyc", ".pyo", ".class",
        ".lock", ".sum",
    }
    return ext in binary_exts


def _is_priority_file(filepath: str) -> bool:
    """Return True if this file should always be included in sampled zips."""
    name = filepath.split("/")[-1].lower()
    return name in {
        "readme.md", "readme.txt", "readme",
        "index.ts", "index.js", "index.py",
        "main.py", "main.ts", "main.js", "main.go",
        "app.py", "app.ts", "app.js",
        "package.json", "pyproject.toml", "cargo.toml",
        "setup.py", "requirements.txt",
        "dockerfile", "docker-compose.yml",
    }
