"""
LangGraph checkpoint persistence.

For the MVP we use LangGraph's built-in MemorySaver so no Postgres
or Redis is required to run the backend. The graph state is kept
in-process and survives as long as the server is alive.

To upgrade to Postgres persistence later:
1. Install `langgraph-checkpoint-postgres`
2. Flip `use_memory_saver = False` in `.env`
3. Uncomment the AsyncPostgresSaver block below.

Reference: https://langchain-ai.github.io/langgraph/reference/checkpoints/
"""
import logging

from langgraph.checkpoint.memory import MemorySaver

from app.core.config import settings

logger = logging.getLogger(__name__)

# Module-level singleton — the same saver is reused across all requests
# so state is shared within a single server process.
_memory_saver: MemorySaver | None = None


def get_checkpointer() -> MemorySaver:
    """
    Return the active LangGraph checkpointer.

    Currently always returns the MemorySaver singleton. In production,
    swap this out for AsyncPostgresSaver when Postgres is available.
    """
    global _memory_saver

    if settings.use_memory_saver:
        if _memory_saver is None:
            _memory_saver = MemorySaver()
            logger.info("LangGraph checkpointer: MemorySaver (in-process).")
        return _memory_saver

    # ── Future: Postgres saver ─────────────────────────────────────────
    # from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    # return AsyncPostgresSaver.from_conn_string(settings.database_url)
    raise NotImplementedError(
        "Postgres checkpointer not yet wired. "
        "Set USE_MEMORY_SAVER=true in .env to use the in-memory saver."
    )
