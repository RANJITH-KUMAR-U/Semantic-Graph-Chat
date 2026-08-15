"""
SQLAlchemy async engine and session factory.

On startup, `init_db()` creates all tables that don't exist yet (safe
to call repeatedly). Import `AsyncSessionLocal` wherever you need a
database session — prefer using it as an async context manager:

    async with AsyncSessionLocal() as session:
        ...

For SQLite (in-memory testing) set DATABASE_URL to:
    sqlite+aiosqlite:///./test.db
"""
import logging

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.models.db_models import Base

logger = logging.getLogger(__name__)

# ── Engine ─────────────────────────────────────────────────────────────
# NullPool is recommended for async engines when using psycopg3.
# For SQLite testing, remove NullPool.
_engine_kwargs: dict = {"echo": settings.debug}

# psycopg3 (psycopg[binary]) uses the "postgresql+psycopg" scheme.
# aiosqlite uses "sqlite+aiosqlite".
if "postgresql" in settings.database_url:
    _engine_kwargs["poolclass"] = NullPool

engine = create_async_engine(settings.database_url, **_engine_kwargs)

# ── Session factory ────────────────────────────────────────────────────
AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def init_db() -> None:
    """
    Create all tables defined in the ORM models.

    Safe to call on every startup — SQLAlchemy only creates tables that
    are missing (it does not drop or alter existing ones).
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialised.")


async def get_db() -> AsyncSession:  # type: ignore[override]
    """
    FastAPI dependency that yields a database session per request.

    Usage in a route:
        async def my_route(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        yield session
