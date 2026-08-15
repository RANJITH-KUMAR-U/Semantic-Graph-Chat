"""
Pydantic request/response schemas (DTOs) shared across API routes.

These are distinct from app.graph.router.RoutingDecision, which is
internal to the routing step. Everything here is part of the public
API contract between the backend and the frontend.
"""
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ── Inbound ────────────────────────────────────────────────────────────


class MessageIn(BaseModel):
    """Payload sent by the client when posting a chat message."""

    content: str = Field(..., description="User's raw message text.", min_length=1)
    force_node_id: Optional[str] = Field(
        default=None,
        description="If set, skip the semantic router and pin to this node.",
    )


class ForceRouteIn(BaseModel):
    """Body for POST /api/nodes/{node_id}/force-route."""

    session_id: str = Field(..., description="Session that owns the node.")


class CreateSessionIn(BaseModel):
    """Optional body for POST /api/sessions (session_id is server-generated if omitted)."""

    session_id: Optional[str] = Field(
        default=None,
        description="Caller-supplied session ID. Generated if omitted.",
    )


# ── Outbound ───────────────────────────────────────────────────────────


# SECURITY-TODO: Validate user authorization before processing reassignments or merges.


class MessageOut(BaseModel):
    """A single message returned from a node's history."""

    message_id: str
    node_id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    confidence: Optional[float] = None
    reasoning: Optional[str] = None


class ReassignMessageIn(BaseModel):
    """Body for POST /api/messages/{message_id}/reassign."""

    session_id: str
    target_node_id: Optional[str] = None
    new_topic_title: Optional[str] = None


class MergeNodesIn(BaseModel):
    """Body for POST /api/nodes/merge."""

    session_id: str
    source_node_id: str
    target_node_id: str


class NodeOut(BaseModel):
    """Summary of a single Topic Node returned to the frontend sidebar."""

    node_id: str
    session_id: str
    title: str
    message_count: int = 0
    created_at: datetime
    last_active_at: datetime
    parent_node_id: Optional[str] = None  # None for root topics
    depth: int = 0                         # 0 = root, 1 = sub-topic
    possible_duplicate_of: Optional[str] = None  # Flagged duplicate node ID
    # Feature Round 3: relatedness graph
    related_node_ids: list[str] = Field(default_factory=list)  # "related" (non-duplicate) nodes
    # In-node bounded memory
    local_summary: str = ""               # compressed digest of archived messages (empty if no compression yet)
    archived_message_count: int = 0       # number of messages moved to archive (not sent to LLM)
    # Document upload: chunk count for this node
    document_chunk_count: int = 0         # number of document chunks indexed in this node



class SessionOut(BaseModel):
    """Full session metadata including its list of topic nodes."""

    session_id: str
    created_at: datetime
    global_summary: str
    nodes: list[NodeOut] = []


# ── Upload DTOs ────────────────────────────────────────────────────────


class UploadStatusOut(BaseModel):
    """Response from POST /api/sessions/{session_id}/upload."""

    upload_id: str
    filename: str
    status: Literal["queued", "chunking", "routing", "indexed", "failed"]
    total_chunks: int = 0
    error: Optional[str] = None
    node_assignments: dict[str, Any] = {}  # { node_id: { title, chunk_count } }


# SECURITY-TODO: Input validation and payload size bounds checking should be strictly enforced here in production.


class WSMessage(BaseModel):
    """
    All messages on the WebSocket use this envelope so the frontend can
    switch on `type` without ad-hoc parsing.

    Types:
        connected       — server confirms the session is ready
        routing         — server is evaluating which node to use
        token           — one streamed token from the generator LLM
        done            — generation complete; carries final routing metadata
        error           — something went wrong; carries error detail
        summary_status  — global summary update status / payload
    """

    type: Literal["connected", "routing", "token", "done", "error", "summary_status", "upload_status"]
    content: Optional[str] = None          # token text or error message
    node_id: Optional[str] = None          # active node after routing
    node_title: Optional[str] = None       # human-readable node label
    parent_node_id: Optional[str] = None   # parent node (for sub-topics)
    node_depth: Optional[int] = None       # 0=root, 1=sub-topic
    reasoning: Optional[str] = None        # router's reasoning string
    confidence: Optional[float] = None     # router confidence score 0-1
    global_summary: Optional[str] = None   # latest global summary
    is_summarizing: Optional[bool] = None  # loading indicator for global summary
    tokens_used: Optional[int] = None      # turn tokens used
    baseline_tokens: Optional[int] = None  # estimated linear chat tokens
    session_id: Optional[str] = None       # echoed back on `connected`
    # Feature Round 3: cross-node reference (Feature 1)
    referenced_node_id: Optional[str] = None   # node referenced by this message
    referenced_node_title: Optional[str] = None  # human-readable title of referenced node
    # Feature Round 3: router model used (Feature 5)
    router_model_used: Optional[str] = None    # which router model was used this turn
    router_latency_ms: Optional[int] = None    # latency of router call in milliseconds
    # Document upload status
    upload_id: Optional[str] = None            # upload ID for tracking
    upload_status: Optional[str] = None        # queued / chunking / routing / indexed / failed
    upload_filename: Optional[str] = None      # filename being processed
    # Source citations (document chunks used in generation)
    source_citations: Optional[list[dict]] = None  # [{source_filename, chunk_id, relevance_score}]

