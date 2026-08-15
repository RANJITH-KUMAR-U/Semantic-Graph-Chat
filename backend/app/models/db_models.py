"""
SQLAlchemy ORM models: DBSession, DBTopicNode, DBMessage.

Matches the entity-relationship diagram in PRD.md section 7:

    SESSION ||--o{ TOPIC_NODE : contains
    TOPIC_NODE ||--o{ MESSAGE : stores
    SESSION ||--|| GLOBAL_SUMMARY : has (stored inline on SESSION)
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


def _now() -> datetime:
    """Return current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


def _uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


class DBSession(Base):
    """
    Top-level chat session.

    One session contains many TopicNodes. The global_summary is a
    compact, LLM-generated digest of activity across all nodes in this
    session, refreshed asynchronously every N turns.
    """

    __tablename__ = "sessions"

    session_id: str = Column(String, primary_key=True, default=_uuid)
    created_at: datetime = Column(DateTime(timezone=True), default=_now, nullable=False)
    global_summary: str = Column(Text, default="", nullable=False)

    # Relationship: one session → many topic nodes
    nodes: list = relationship(
        "DBTopicNode",
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<DBSession id={self.session_id!r}>"


class DBTopicNode(Base):
    """
    An isolated topic within a session.

    Each node maintains its own message history. No message from
    Node A ever appears in the context assembled for Node B.

    In-node bounded memory:
        local_summary stores the LLM-compressed digest of archived messages.
        The live message count (len(messages)) reflects only the rolling
        window; the full history is in archived_message_count.
    """

    __tablename__ = "topic_nodes"

    node_id: str = Column(String, primary_key=True, default=_uuid)
    session_id: str = Column(
        String, ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False
    )
    title: str = Column(String(255), nullable=False)
    # JSON-serialised embedding vector — stored as text for portability.
    # Swap to pgvector's Vector column when running on Postgres + pgvector.
    embedding: str = Column(Text, default="[]", nullable=False)
    created_at: datetime = Column(DateTime(timezone=True), default=_now, nullable=False)
    last_active_at: datetime = Column(DateTime(timezone=True), default=_now, nullable=False)
    # In-node bounded memory
    local_summary: str = Column(Text, default="", nullable=False)

    # Relationships
    session: DBSession = relationship("DBSession", back_populates="nodes")
    messages: list = relationship(
        "DBMessage",
        back_populates="node",
        cascade="all, delete-orphan",
        order_by="DBMessage.created_at",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<DBTopicNode id={self.node_id!r} title={self.title!r}>"


class DBMessage(Base):
    """
    A single chat message stored under a TopicNode.

    role is either 'user' or 'assistant' (OpenAI convention).
    """

    __tablename__ = "messages"

    message_id: str = Column(String, primary_key=True, default=_uuid)
    node_id: str = Column(
        String, ForeignKey("topic_nodes.node_id", ondelete="CASCADE"), nullable=False
    )
    role: str = Column(String(16), nullable=False)  # 'user' | 'assistant'
    content: str = Column(Text, nullable=False)
    created_at: datetime = Column(DateTime(timezone=True), default=_now, nullable=False)

    # Relationship
    node: DBTopicNode = relationship("DBTopicNode", back_populates="messages")

    def __repr__(self) -> str:
        return f"<DBMessage id={self.message_id!r} role={self.role!r}>"
