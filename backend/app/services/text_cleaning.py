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
    r"(?:^|\n)(?:[#*\s]*)(?:Final\s+(?:Output|Answer|Response|Result|Title)|Drafting(?:\s*-\s*Attempt\s*\d+)?|Main\s+topic|Output|Title):\s*\n?",
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
]

# Pattern for numbered/bulleted reasoning lines at the start of text (e.g. "4.  **Check Constraints:** ")
NUMBERED_REASONING_PREFIX = re.compile(
    r"^\s*(?:\d+\.\s+|\*\s*|\-\s*)?\*\*(?:Analyze|Identify|Draft|Check|Mental|Formulate|Concept|Review|Refinement|Constraints)[^*]*\*\*[:\s]*",
    re.IGNORECASE,
)


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
        # If the text before the marker contains reasoning keywords, take what comes after the marker
        if any(kw in prefix for kw in ("think", "analyz", "draft", "step", "user wants", "request", "mental", "check")):
            remaining = cleaned[match.end():].strip()
            if remaining:
                cleaned = remaining

    # 4. Strip preamble headers if at the beginning of the text
    for prefix in PREAMBLE_PREFIXES:
        if cleaned.lower().startswith(prefix):
            paragraphs = cleaned.split("\n\n")
            if len(paragraphs) > 1:
                for i, p in enumerate(paragraphs):
                    p_strip = p.strip().lower()
                    if not any(p_strip.startswith(pfx) for pfx in PREAMBLE_PREFIXES) and not p_strip.startswith(("1.", "2.", "3.", "4.", "step ", "*", "-")):
                        cleaned = "\n\n".join(paragraphs[i:]).strip()
                        break
            break

    # 5. Strip any leftover leading numbered/bulleted reasoning headers (e.g. "4.  **Check Constraints:** ")
    while NUMBERED_REASONING_PREFIX.match(cleaned):
        cleaned = NUMBERED_REASONING_PREFIX.sub("", cleaned, count=1).strip()

    return cleaned
