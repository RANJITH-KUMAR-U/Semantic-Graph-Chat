"""
Semantic Router — classifies each incoming message against the list
of currently active Topic Nodes and returns a structured RoutingDecision.

Supports 3-way routing decisions (PRD §6.1 + sub-topic extension):
  1. route_existing   → message belongs to an existing node (any depth)
  2. create_subtopic  → message is a sub-aspect of an existing root topic
                        (e.g., "nebulae" under "Space Science")
  3. create_new       → genuinely new top-level topic

Non-negotiable constraint (AGENTS.md):
  "Router output must be structured (Pydantic/tool-calling), never free text."

Fallback rule (PRD section 6.4):
  On low confidence or error, default to the currently active node
  rather than spawning a spurious new one.
"""
import logging
from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.graph.state import GraphState
from app.services import llm_service

logger = logging.getLogger(__name__)


# ── Structured output schema ───────────────────────────────────────────


# SECURITY-TODO: Ensure input sanitization and rate limits on LLM routing endpoints in production.

class RoutingDecision(BaseModel):
    """
    Structured output returned by the semantic router.

    decision options:
        "route_existing"  → route to an existing node (root or sub-topic)
        "create_subtopic" → create a child sub-topic under target_node_id
        "create_new"      → create a brand-new root-level topic node
    """

    decision: Literal["route_existing", "create_subtopic", "create_new"] = Field(
        description="Routing action to take."
    )
    target_node_id: Optional[str] = Field(
        default=None,
        description=(
            "For route_existing: the node to activate. "
            "For create_subtopic: the parent root node to create the child under. "
            "For create_new: null."
        ),
    )
    confidence: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )
    reasoning: str = Field(
        description="Brief explanation of why this route was chosen."
    )


# ── Graph node function ────────────────────────────────────────────────


async def router_node(state: GraphState) -> dict:
    """
    LangGraph node: evaluate the current user input against active nodes.

    Reads:
        state["current_input"]  — the raw user message
        state["nodes"]          — all active topic nodes (root + sub-topics)
        state["active_node_id"] — current node (used as fallback)
        state["force_node_id"]  — non-None skips the LLM call entirely

    Returns:
        A partial state dict updating `routing_decision`.
    """
    current_input: str = state["current_input"]
    active_nodes: dict = state.get("nodes", {})
    current_node_id: Optional[str] = state.get("active_node_id") or None
    force_node_id: Optional[str] = state.get("force_node_id")
    # Feature 5: per-message router model override
    model_override: Optional[str] = state.get("router_model_override")

    # ── Manual override path ───────────────────────────────────────────
    if force_node_id and force_node_id in active_nodes:
        logger.info("Manual override: routing to node %r", force_node_id)
        decision = RoutingDecision(
            decision="route_existing",
            target_node_id=force_node_id,
            reasoning="Manual override via force_node_id.",
        )
        return {
            "routing_decision": decision.model_dump(),
            "force_node_id": None,  # consume the override
        }

    # ── No nodes yet: always create ───────────────────────────────────
    if not active_nodes:
        logger.info("No active nodes — will create first node.")
        return {
            "routing_decision": RoutingDecision(
                decision="create_new",
                target_node_id=None,
                reasoning="No active nodes exist yet — creating the first topic node.",
            ).model_dump()
        }

    # ── LLM routing call (hierarchy-aware) ──────────────────────────────────
    raw = await llm_service.call_router_llm(
        user_message=current_input,
        active_nodes=active_nodes,
        current_node_id=current_node_id,
        model_override=model_override,
    )

    # Validate via Pydantic
    try:
        decision = RoutingDecision(**raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Pydantic validation of routing decision failed: %s", exc)
        decision = RoutingDecision(
            decision="route_existing" if current_node_id else "create_new",
            target_node_id=current_node_id,
            confidence=0.50,
            reasoning=f"Validation fallback: {exc}",
        )

    # ── Confidence Calibration ──────────────────────────────────────────
    # Adjust self-reported LLM confidence if word overlap with target node is low
    from app.services.similarity import jaccard_similarity, node_text
    if decision.decision in ("route_existing", "create_subtopic") and decision.target_node_id in active_nodes:
        target_node = active_nodes[decision.target_node_id]
        sim = jaccard_similarity(current_input, node_text(target_node))
        # If vocabulary overlap is very low (<0.08) and router reported very high confidence (>0.85),
        # calibrate confidence down to reflect potential domain-level generalization
        if sim < 0.08 and decision.confidence > 0.85:
            calibrated_conf = round(max(0.55, decision.confidence * 0.70), 2)
            logger.info(
                "Calibrated router confidence down from %.2f to %.2f due to low vocabulary overlap (sim=%.3f)",
                decision.confidence, calibrated_conf, sim
            )
            decision = RoutingDecision(
                decision=decision.decision,
                target_node_id=decision.target_node_id,
                confidence=calibrated_conf,
                reasoning=f"{decision.reasoning} [Calibrated: weak word overlap (sim={sim:.2f})]",
            )

    logger.info(
        "Routing decision: %s → %s (conf=%.2f, %s)",
        decision.decision,
        decision.target_node_id,
        decision.confidence,
        decision.reasoning,
    )
    return {"routing_decision": decision.model_dump()}


# ── Conditional edge helper ────────────────────────────────────────────


def routing_edge(state: GraphState) -> str:
    """
    Conditional edge function used by the LangGraph StateGraph.

    Returns the name of the next node to transition to based on the
    routing decision stored in state.

    Returns:
        "create_node"      if decision == "create_new"
        "create_subtopic"  if decision == "create_subtopic"
        "generate"         if decision == "route_existing"
    """
    decision = state.get("routing_decision", {})
    d = decision.get("decision")
    if d == "create_new":
        return "create_node"
    if d == "create_subtopic":
        return "create_subtopic"
    return "generate"
