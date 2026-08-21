"""
Text cleaning utilities to strip reasoning/thinking meta-text from LLM outputs.

Many open-weights / free-tier reasoning models (like Gemma-4, DeepSeek-R1, Nemotron variants)
output internal chain-of-thought before the final answer, e.g.:
- `<think>...</think>` tags
- `<reasoning>...</reasoning>` tags
- Preambles like "Here's a thinking process:\n1. **Analyze...**"
- Meta-commentary blocks before the actual content

This module strips out those artifacts so node titles, summaries, and user-facing
responses remain clean and direct.
"""
import re


# Compiled regex patterns for thinking / reasoning tags
THINK_TAG_PATTERN = re.compile(
    r"<(think|reasoning|thought|antthinking)>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)

# Unclosed opening think tag at start (e.g. streaming or truncated)
UNCLOSED_THINK_START = re.compile(
    r"^<(think|reasoning|thought|antthinking)>.*?(?:\n\n|\Z)",
    re.DOTALL | re.IGNORECASE,
)

OUTPUT_MARKER_PATTERN = re.compile(
    r"(?:^|\n)(?:[#*\s]*)(?:Final\s+(?:Output|Answer|Response|Result|Title|Summary)|Drafting(?:\s*-\s*Attempt\s*\d+)?|Main\s+topic|Output|Title|Summary):\s*\n?",
    re.IGNORECASE,
)

PREAMBLE_PREFIXES = [
    "here's a thinking process",
    "here is a thinking process",
    "here's the thinking process",
    "here is the thinking process",
    "let me analyze",
    "let's analyze",
    "let us analyze",
    "let's think",
    "let me think",
    "let us think",
    "but wait, the user",
    "but wait",
    "wait, the user",
    "the user said",
    "below is recent activity",
    "let's see",
    "let me see",
    "okay, the user",
    "ok, the user",
    "the user is asking",
]

# Pattern for inline reasoning prefixes like "5.  **Select the Best One:** ", "1. **Analyze:** "
INLINE_REASONING_PREFIX = re.compile(
    r"^\s*(?:\d+[\.\)]|\*|\-)?\s*(?:\*\*[^*]+\*\*|[A-Z][a-zA-Z\s]{1,30}:)\s*",
    re.IGNORECASE,
)


def is_reasoning_line(line: str) -> bool:
    """Check if a single line is a standalone reasoning step header or meta-commentary."""
    l = line.strip().lower()
    if not l:
        return True
    if any(l.startswith(prefix) for prefix in PREAMBLE_PREFIXES):
        return True
    if re.match(r"^\s*\d+[\.\)]\s+\*\*", l):
        return True
    if re.match(r"^\s*(?:check|determine|analyze|identify|draft|formulate|review|select|refine|count|note|step|wait|looking)\s+", l):
        return True
    if l.endswith(":") and len(l.split()) <= 6:
        return True
    return False


def strip_reasoning(text: str) -> str:
    """
    Remove reasoning/thinking meta-commentary from LLM output.

    Args:
        text: Raw text output from an LLM.

    Returns:
        Cleaned text with thinking blocks and preambles removed.
    """
    if not text:
        return ""

    cleaned = text.strip()

    # 1. Strip explicit <think>...</think> tags
    cleaned = THINK_TAG_PATTERN.sub("", cleaned).strip()

    # 2. Handle unclosed opening tag if present at the very beginning
    cleaned = UNCLOSED_THINK_START.sub("", cleaned).strip()

    # 3. Check for explicit output markers like "Final Output:", "Drafting - Attempt X:", "Title:"
    match = OUTPUT_MARKER_PATTERN.search(cleaned)
    if match:
        prefix = cleaned[:match.start()].lower()
        if any(kw in prefix for kw in ("think", "analyz", "draft", "step", "user wants", "request", "mental", "check", "determine", "select", "wait")):
            remaining = cleaned[match.end():].strip()
            if remaining:
                cleaned = remaining

    # 4. Strip preamble headers if at the beginning of the text
    for prefix in PREAMBLE_PREFIXES:
        if cleaned.lower().startswith(prefix):
            paragraphs = cleaned.split("\n\n")
            if len(paragraphs) > 1:
                for i, p in enumerate(paragraphs):
                    if not is_reasoning_line(p):
                        cleaned = "\n\n".join(paragraphs[i:]).strip()
                        break
            else:
                # Single paragraph starting with preamble — try splitting by quotes or newlines
                lines = cleaned.splitlines()
                for i, line in enumerate(lines):
                    if not is_reasoning_line(line):
                        cleaned = "\n".join(lines[i:]).strip()
                        break
            break

    # 5. Filter out leading standalone reasoning lines
    lines = cleaned.splitlines()
    start_idx = 0
    for i, line in enumerate(lines):
        if is_reasoning_line(line):
            start_idx = i + 1
        else:
            break

    if start_idx < len(lines):
        cleaned = "\n".join(lines[start_idx:]).strip()

    # 6. Strip leading inline reasoning prefix if present (e.g. "5. **Select the Best One:** ")
    while INLINE_REASONING_PREFIX.match(cleaned):
        m = INLINE_REASONING_PREFIX.match(cleaned)
        matched_str = m.group(0).lower()
        if any(kw in matched_str for kw in ("select", "check", "determine", "analyze", "draft", "step", "refine", "count", "option", "wait")):
            cleaned = cleaned[m.end():].strip()
        else:
            break

    # 7. Strip prompt-echo lines (e.g. quotes of system instructions or prompt headers)
    filtered_lines = []
    for line in cleaned.splitlines():
        l_lower = line.lower().strip()
        if "below is recent activity" in l_lower or "you are a context summariser" in l_lower or "main topics discussed so far" in l_lower:
            continue
        filtered_lines.append(line)
    cleaned = "\n".join(filtered_lines).strip()

    return cleaned
