"""
LLM service — thin wrapper around the OpenRouter API.

OpenRouter exposes an OpenAI-compatible REST API, so we use the
official `openai` SDK pointed at `https://openrouter.ai/api/v1`.

Two public surfaces:
  - call_router_llm()   -> structured RoutingDecision (JSON mode)
  - stream_generator()  -> async token generator for the chat response

Centralising all LLM calls here means you can swap models or
providers in one place without touching graph logic.
"""
import json
import logging
import re
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from openai import AsyncOpenAI

from app.core.config import settings
from app.services.text_cleaning import strip_reasoning

logger = logging.getLogger(__name__)

# ── Gemini Direct API Helpers ──────────────────────────────────────────

async def _call_gemini_api(
    messages: list[dict[str, str]],
    system_prompt: str = "",
    model_name: str = "gemini-3.6-flash",
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """Call Google Gemini generateContent REST API directly."""
    key = settings.gemini_api_key
    if not key:
        raise ValueError("GEMINI_API_KEY is not configured.")

    clean_model = model_name.replace("models/", "").replace(":free", "")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={key}"

    contents = []
    for m in messages:
        role = "model" if m.get("role") in ("assistant", "model") else "user"
        contents.append({"role": role, "parts": [{"text": m.get("content", "")}]})

    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if system_prompt:
        payload["system_instruction"] = {"parts": [{"text": system_prompt}]}

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API error ({resp.status_code}): {resp.text[:200]}")
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("No candidates returned from Gemini API.")
        return candidates[0]["content"]["parts"][0]["text"]


async def _stream_gemini_api(
    messages: list[dict[str, str]],
    system_prompt: str = "",
    model_name: str = "gemini-3.6-flash",
    temperature: float = 0.7,
) -> AsyncGenerator[str, None]:
    """Stream tokens directly from Google Gemini streamGenerateContent REST API."""
    key = settings.gemini_api_key
    if not key:
        raise ValueError("GEMINI_API_KEY is not configured.")

    clean_model = model_name.replace("models/", "").replace(":free", "")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:streamGenerateContent?alt=sse&key={key}"

    contents = []
    for m in messages:
        role = "model" if m.get("role") in ("assistant", "model") else "user"
        contents.append({"role": role, "parts": [{"text": m.get("content", "")}]})

    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 2048,
        },
    }
    if system_prompt:
        payload["system_instruction"] = {"parts": [{"text": system_prompt}]}

    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"Gemini streaming error ({resp.status_code})")
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    try:
                        chunk_json = json.loads(line[6:])
                        part_text = (
                            chunk_json.get("candidates", [{}])[0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text", "")
                        )
                        if part_text:
                            yield part_text
                    except Exception:
                        pass

# ── Client singleton ───────────────────────────────────────────────────
_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    """Return the OpenAI-compatible AsyncOpenAI client using current settings."""
    global _client
    if _client is None:
        api_key = settings.openrouter_api_key or "sk-or-v1-dummy"
        _client = AsyncOpenAI(
            api_key=api_key,
            base_url=settings.openrouter_base_url,
            default_headers={
                "HTTP-Referer": "https://semantic-graph-chat.vercel.app",
                "X-Title": "Semantic Graph Chat",
            },
            max_retries=1,
        )
    return _client


# ── Helpers ─────────────────────────────────────────────────────────────


def _clean_json_str(raw: str) -> str:
    """Extract valid JSON substring from raw model output, handling code fences."""
    cleaned = raw.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            p = part.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{") and p.endswith("}"):
                return p
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and start < end:
        return cleaned[start : end + 1]
    return cleaned


# ── Live free-tier OpenRouter models (verified 2026-08-21) ─────────────
# Full list from GET https://openrouter.ai/api/v1/models filtered by :free
LIVE_FREE_MODELS = [
    "nvidia/nemotron-3.5-lightning:free",       # Fast, good quality – primary
    "nvidia/nemotron-3-super-120b-a12b:free",   # Large, high quality
    "nvidia/nemotron-nano-9b-v2:free",          # Small, fast
    "nvidia/nemotron-3-nano-30b-a3b:free",      # Medium
    "openai/gpt-oss-20b:free",                  # GPT-compatible
    "google/gemma-4-31b-it:free",               # Google Gemma 31B
    "google/gemma-4-26b-a4b-it:free",           # Google Gemma 26B
    "liquid/lfm-2.5-2.6b:free",                 # Tiny, ultra-fast fallback
    "poolside/laguna-s-2.1:free",               # Poolside small
    "poolside/laguna-xs-2.1:free",              # Poolside extra-small
    "z-ai/glm-5.2:free",                        # GLM
    "cohere/north-mini-code:free",              # Cohere code model
    "dots-studio/dots-3-note-preview:free",     # Dots
    "nvidia/nemotron-3-ultra-550b-a55b:free",   # Ultra large (may be slow)
    "nvidia/nemotron-nano-12b-v2-vl:free",      # Vision-language
]

# Models known to be removed/deprecated — will be swapped to primary
DEPRECATED_MODELS = {
    "nvidia/nemotron-3-embed-1b:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "huggingfaceh4/zephyr-7b-beta:free",
    "openchat/openchat-7b:free",
    "deepseek/deepseek-r1:free",
    "google/gemini-2.0-flash-exp:free",
    "qwen/qwen-2.5-coder-32b-instruct:free",
    "mistralai/mistral-small-24b-instruct-2501:free",
    "inclusionai/ling-3.0-flash:free",
    "nvidia/nemotron-3.5-content-safety:free",  # Safety model, not for chat
}

PRIMARY_FREE_MODEL = "nvidia/nemotron-3.5-lightning:free"


def _sanitize_model_name(model_name: str | None) -> str:
    """Return an active model if model_name is empty or in DEPRECATED_MODELS."""
    if not model_name or model_name in DEPRECATED_MODELS:
        logger.warning("Model %r is deprecated or unset. Using primary: %s", model_name, PRIMARY_FREE_MODEL)
        return PRIMARY_FREE_MODEL
    return model_name


def _get_candidate_models(configured_model: str, secondary_model: str | None = None) -> list[str]:
    """
    Return an ordered list of candidate models for resilience against 404s.
    - Puts the configured (sanitized) model first
    - Puts secondary_model second if specified
    - Follows with all other verified live free models
    - Removes any deprecated models
    """
    sanitized = _sanitize_model_name(configured_model)
    prefix = [sanitized]
    if secondary_model:
        sec_sanitized = _sanitize_model_name(secondary_model)
        if sec_sanitized != sanitized:
            prefix.append(sec_sanitized)

    candidates = prefix + [m for m in LIVE_FREE_MODELS if m not in prefix]
    # Filter out deprecated and remove duplicates
    seen: set[str] = set()
    result = []
    for m in candidates:
        if m not in DEPRECATED_MODELS and m not in seen:
            seen.add(m)
            result.append(m)
    return result


# ── Router LLM ─────────────────────────────────────────────────────────


async def call_router_llm(
    user_message: str,
    active_nodes: dict[str, dict],
    current_node_id: str | None,
    model_override: str | None = None,
) -> dict[str, Any]:
    """
    Ask the router model to classify the incoming message using 3-way routing.

    Decision options:
        route_existing   → message belongs to an existing node (any depth)
        create_subtopic  → message is a sub-aspect of a root topic
        create_new       → genuinely new top-level topic

    Args:
        user_message:    The raw user input to classify.
        active_nodes:    Dict of {node_id: NodeData} for all active nodes.
        current_node_id: The node that was active before this turn.
        model_override:  If set, try this model first (Feature 5 playground).

    Returns:
        A dict with keys: decision, target_node_id, reasoning, confidence,
        model_used (str), latency_ms (int).
    """
    # Build a hierarchical tree description for the prompt
    # Exclude ghost nodes with 0 messages
    root_nodes = {
        nid: nd for nid, nd in active_nodes.items()
        if nd.get("depth", 0) == 0 and (len(nd.get("messages", [])) > 0 or len(nd.get("document_chunks") or []) > 0)
    }
    sub_nodes = {
        nid: nd for nid, nd in active_nodes.items()
        if nd.get("depth", 0) == 1 and (len(nd.get("messages", [])) > 0 or len(nd.get("document_chunks") or []) > 0)
    }

    if root_nodes or sub_nodes:
        lines = []
        for nid, nd in root_nodes.items():
            turns = len(nd.get("messages", []))
            lines.append(
                f"ROOT node_id={nid!r} title={nd.get('title','Untitled')!r} turns={turns}"
            )
            # Attach children
            children = {
                cid: cd for cid, cd in sub_nodes.items()
                if cd.get("parent_node_id") == nid
            }
            for cid, cd in children.items():
                cturns = len(cd.get("messages", []))
                lines.append(
                    f"  SUB  node_id={cid!r} title={cd.get('title','Untitled')!r} turns={cturns}"
                )
        nodes_desc = "\n".join(lines)
    else:
        nodes_desc = "(none — this is the first message)"

    system_prompt = (
        "You are a semantic topic router for a multi-topic chat system that supports "
        "hierarchical topic nodes (root topics and sub-topics).\n"
        "You MUST respond with ONLY a valid JSON object. Do not output any reasoning, thinking process, or explanation outside the JSON.\n\n"
        "## Active Topic Tree\n"
        f"{nodes_desc}\n\n"
        "## 3-Way Classification Rules\n"
        "1. **route_existing** — The message is a direct follow-up, continuation, or "
        "closely related to an EXISTING node (root or sub-topic). "
        "Set target_node_id to that node's ID.\n"
        "2. **create_subtopic** — The message introduces a SPECIFIC ASPECT or "
        "NARROWER ANGLE of an existing ROOT topic. Example: if 'Space Science' exists "
        "and the user asks about 'nebulae', that is a sub-topic of Space Science. "
        "Set target_node_id to the PARENT ROOT node's ID.\n"
        "3. **create_new** — The message introduces a COMPLETELY DIFFERENT domain "
        "or subject with no existing root topic. Set target_node_id to null.\n\n"
        "## Important Rules\n"
        "- Only create a sub-topic (create_subtopic) if there is a clear ROOT node it belongs under.\n"
        "- Do NOT create sub-topics of sub-topics (max depth = 1).\n"
        "- Distinct specialized concepts are SEPARATE topics. Do NOT route a distinct concept to an existing node unless the message explicitly asks to compare them.\n"
        "- Provide a confidence score between 0.0 and 1.0 (e.g. 0.92 for high confidence, 0.65 for medium).\n\n"
        "## Response Format\n"
        "Return ONLY a JSON object:\n"
        "{\n"
        '  "decision": "route_existing" | "create_subtopic" | "create_new",\n'
        '  "target_node_id": "<node_id>" | null,\n'
        '  "confidence": 0.92,\n'
        '  "reasoning": "<short sentence explanation>"\n'
        "}"
    )

    user_prompt = f"User message: {user_message!r}"
    client = _get_client()
    # Feature 5: model_override is tried first if provided; otherwise router_model then router_model_alt
    base_candidates = _get_candidate_models(settings.router_model, settings.router_model_alt)
    if model_override:
        sanitized_override = _sanitize_model_name(model_override)
        candidate_models = [sanitized_override] + [m for m in base_candidates if m != sanitized_override]
    else:
        candidate_models = base_candidates

    last_exception = None
    t_start = time.monotonic()
    model_used: str = candidate_models[0]
    # ── Try Gemini Direct API first if GEMINI_API_KEY is configured ──
    if settings.gemini_api_key:
        try:
            gemini_model = "gemini-3.6-flash"
            if model_override and "gemini" in model_override.lower():
                gemini_model = model_override.replace("models/", "").replace(":free", "").split("/")[-1]
            
            raw_response = await _call_gemini_api(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                model_name=gemini_model,
                temperature=0.0,
                max_tokens=1024,
            )
            cleaned_json = _clean_json_str(raw_response)
            result: dict = json.loads(cleaned_json)

            if "decision" in result:
                if result["decision"] in ("route_existing", "create_subtopic"):
                    tid = result.get("target_node_id")
                    if not tid or tid not in active_nodes:
                        result["decision"] = "route_existing"
                        result["target_node_id"] = current_node_id
                        result["reasoning"] = (
                            result.get("reasoning", "")
                            + " [fallback: target_node_id not found]"
                        )

                if "confidence" not in result or not isinstance(result["confidence"], (int, float)):
                    result["confidence"] = 0.95
                else:
                    result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))

                if "reasoning" in result and isinstance(result["reasoning"], str):
                    result["reasoning"] = strip_reasoning(result["reasoning"])

                model_used = f"google/{gemini_model}"
                latency_ms = int((time.monotonic() - t_start) * 1000)
                result["model_used"] = model_used
                result["latency_ms"] = latency_ms
                logger.info("Router decision via Gemini Direct (%s, %dms): %s", gemini_model, latency_ms, result)
                return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini Direct router failed (%s). Falling back to OpenRouter candidates...", exc)

    for model_name in candidate_models:
        raw_response: str = ""
        try:
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=256,
                )
                raw_response = response.choices[0].message.content or "{}"
            except Exception:  # noqa: BLE001
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=256,
                )
                raw_response = response.choices[0].message.content or "{}"

            cleaned_json = _clean_json_str(raw_response)
            result = json.loads(cleaned_json)

            if "decision" not in result:
                raise ValueError("Missing 'decision' field in router response.")

            # Validate target_node_id exists for route_existing / create_subtopic
            if result["decision"] in ("route_existing", "create_subtopic"):
                tid = result.get("target_node_id")
                if not tid or tid not in active_nodes:
                    # Fall back to current node
                    result["decision"] = "route_existing"
                    result["target_node_id"] = current_node_id
                    result["reasoning"] = (
                        result.get("reasoning", "")
                        + " [fallback: target_node_id not found]"
                    )

            if "confidence" not in result or not isinstance(result["confidence"], (int, float)):
                result["confidence"] = 0.90
            else:
                result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))

            # Clean any reasoning meta-text from reasoning field
            if "reasoning" in result and isinstance(result["reasoning"], str):
                result["reasoning"] = strip_reasoning(result["reasoning"])

            model_used = model_name
            latency_ms = int((time.monotonic() - t_start) * 1000)
            result["model_used"] = model_used
            result["latency_ms"] = latency_ms
            logger.info("Router decision (%s, %dms): %s", model_name, latency_ms, result)
            return result
        except Exception as exc:  # noqa: BLE001
            last_exception = exc
            logger.warning("Router model %s failed (%s). Retrying next candidate...", model_name, exc)

    logger.error("All router models failed (%s). Defaulting based on state.", last_exception)
    latency_ms = int((time.monotonic() - t_start) * 1000)
    return {
        "decision": "create_new" if not current_node_id else "route_existing",
        "target_node_id": current_node_id if current_node_id else None,
        "confidence": 0.70,
        "reasoning": f"Router fallback due to error: {last_exception}",
        "model_used": model_used,
        "latency_ms": latency_ms,
    }


# ── Generator LLM (streaming) ──────────────────────────────────────────


async def stream_generator(
    messages: list[dict[str, str]],
    system_prompt: str = "",
) -> AsyncGenerator[str, None]:
    """
    Stream response tokens from the generator model with reasoning filter.

    Args:
        messages:      List of {"role": ..., "content": ...} dicts
                       for the *active node only* (context isolation).
        system_prompt: Optional system message prepended to the context.

    Yields:
        Individual text tokens/chunks as they arrive from the API.
    """
    clean_sys_prompt = system_prompt
    if "Respond directly" not in clean_sys_prompt:
        clean_sys_prompt = (
            f"{system_prompt}\n\n"
            "IMPORTANT: Output your response directly. Do NOT output thinking steps, reasoning preambles, or analysis of your thought process."
        )

    # ── Try Gemini Direct API streaming if configured ──
    if settings.gemini_api_key:
        try:
            gemini_model = "gemini-3.6-flash"
            async for token in _stream_gemini_api(messages, clean_sys_prompt, model_name=gemini_model):
                yield token
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini Direct streaming failed (%s). Falling back to OpenRouter...", exc)

    full_messages = []
    if clean_sys_prompt:
        full_messages.append({"role": "system", "content": clean_sys_prompt})
    full_messages.extend(messages)

    client = _get_client()
    candidate_models = _get_candidate_models(settings.generator_model)

    for model_name in candidate_models:
        try:
            stream = await client.chat.completions.create(
                model=model_name,
                messages=full_messages,
                stream=True,
                temperature=0.7,
                max_tokens=2048,
            )
            # Buffer initial tokens to catch and discard <think> tags or reasoning preambles if generated
            in_think_tag = False
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    token = delta.content
                    if "<think>" in token or "<reasoning>" in token:
                        in_think_tag = True
                        continue
                    if "</think>" in token or "</reasoning>" in token:
                        in_think_tag = False
                        continue
                    if in_think_tag:
                        continue
                    yield token
            return  # Successful stream completion
        except Exception as exc:  # noqa: BLE001
            logger.warning("Generator model %s failed (%s). Trying next candidate...", model_name, exc)

    yield "\n\n[Error generating response: All LLM models unavailable]"


async def call_generator_once(
    messages: list[dict[str, str]],
    system_prompt: str = "",
) -> str:
    """
    Non-streaming generator call - returns the clean response as a string.

    Used by the global summarizer, title generation, and other non-interactive paths.
    """
    clean_sys_prompt = system_prompt
    if "Do NOT include any reasoning" not in clean_sys_prompt:
        clean_sys_prompt = (
            f"{system_prompt}\n"
            "Provide ONLY the final output. Do NOT include any reasoning, thinking process, or meta-commentary."
        )

    # ── Try Gemini Direct API if configured ──
    if settings.gemini_api_key:
        try:
            gemini_model = "gemini-3.6-flash"
            res = await _call_gemini_api(
                messages=messages,
                system_prompt=clean_sys_prompt,
                model_name=gemini_model,
                temperature=0.3,
                max_tokens=512,
            )
            cleaned = strip_reasoning(res)
            if cleaned:
                return cleaned
            return res
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini Direct once call failed (%s). Falling back to OpenRouter...", exc)

    full_messages = []
    if clean_sys_prompt:
        full_messages.append({"role": "system", "content": clean_sys_prompt})
    full_messages.extend(messages)

    client = _get_client()
    candidate_models = _get_candidate_models(settings.generator_model)

    for model_name in candidate_models:
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=full_messages,
                temperature=0.3,
                max_tokens=512,
            )
            content = response.choices[0].message.content or ""
            if content:
                # Strip any thinking / reasoning preambles
                cleaned = strip_reasoning(content)
                if cleaned:
                    return cleaned
                return content
        except Exception as exc:  # noqa: BLE001
            logger.warning("Generator once model %s failed (%s). Trying next candidate...", model_name, exc)

    return "[Generator error: All LLM models unavailable]"
