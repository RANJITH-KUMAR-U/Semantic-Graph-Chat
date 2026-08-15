"""
FastAPI application entrypoint.

Wires together:
  - REST routes  (api/routes/chat.py, api/routes/nodes.py)
  - WebSocket    (api/websocket.py)
  - LangGraph    (graph/graph_builder.py — compiled at startup)

Run locally:
    uvicorn app.main:app --reload --port 8000

Health check:
    GET http://localhost:8000/health

Swagger UI:
    http://localhost:8000/docs
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import websocket
from app.api.routes import chat, nodes, upload
from app.core.config import settings
from app.graph.graph_builder import build_graph

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan: run startup logic, then yield, then shutdown.

    Startup:
      - Compile the LangGraph StateGraph (warms the @lru_cache).
      - Log the active model configuration.

    Shutdown:
      - Nothing to tear down for the MemorySaver. Add cleanup here when
        switching to the Postgres checkpointer.
    """
    logger.info("═══ Semantic Graph Chat API starting up ═══")
    logger.info("Router model  : %s", settings.router_model)
    logger.info("Generator model: %s", settings.generator_model)
    logger.info("Checkpointer  : %s", "MemorySaver (in-process)" if settings.use_memory_saver else "Postgres")

    # Pre-compile graph (surfaces import/wiring errors early)
    try:
        build_graph()
        logger.info("LangGraph StateGraph compiled ✓")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to compile graph: %s", exc)
        raise

    yield

    logger.info("═══ Semantic Graph Chat API shutting down ═══")


# ── App factory ────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=(
        "Semantic Graph Chat backend — routes user messages to isolated "
        "Topic Nodes via a fast LLM semantic router, maintaining separate "
        "memory per topic so context never bleeds between threads."
    ),
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────
# In production, restrict origins to your frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────────────────────────────────
app.include_router(chat.router, prefix="/api")
app.include_router(nodes.router, prefix="/api")
app.include_router(upload.router, prefix="/api")
app.include_router(websocket.router)


# ── Health check ───────────────────────────────────────────────────────
@app.get("/health", tags=["meta"])
async def health() -> dict:
    """Lightweight liveness probe — returns 200 when the server is up."""
    return {
        "status": "ok",
        "version": settings.app_version,
        "router_model": settings.router_model,
        "generator_model": settings.generator_model,
    }


@app.get("/", tags=["meta"])
async def root() -> dict:
    """Root redirect hint."""
    return {
        "message": "Semantic Graph Chat API",
        "docs": "/docs",
        "health": "/health",
    }
