# Semantic Graph Chat

An LLM chat architecture that replaces linear chat history with isolated,
auto-routed **Topic Nodes** — so a long, multi-topic conversation never
pollutes itself. Full spec: **[`PRD.md`](./PRD.md)** (read this first — it has
every diagram, the data model, and the API surface).

This repo is a **scaffold**, not a finished app: every file has a docstring
explaining exactly what belongs there. It's built to be handed to an
Antigravity agent (or any coding agent / yourself) to fill in, following the
build order in **[`AGENTS.md`](./AGENTS.md)**.

---

## How to use this repo

1. **Read `PRD.md`** — the architecture, tech stack, diagrams, and roadmap.
2. **Read `AGENTS.md`** — the build order and hard constraints, written for
   an agent (Antigravity reads this file automatically before doing any work
   in this workspace — it's the same convention as `.cursorrules` or
   `CLAUDE.md`, just cross-tool).
3. **Open this folder as a Project in Antigravity** and prompt it with
   something like: *"Follow AGENTS.md and PRD.md — implement the backend
   in the order specified, starting with the LangGraph state and router."*
4. Or build it yourself top-down, module by module — every stub file below
   tells you exactly what it's responsible for.

---

## Folder structure

```
semantic-graph-chat/
├── PRD.md                  # Full product spec — architecture, diagrams, roadmap
├── AGENTS.md                # Standing instructions for Antigravity (or any agent)
├── README.md                # This file
├── .env.example              # All required environment variables, one place
├── docker-compose.yml         # Postgres (pgvector) + Redis + backend, for local dev
├── .gitignore
│
├── backend/                  # FastAPI + LangGraph service
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── app/
│   │   ├── main.py            # FastAPI app entrypoint — wires everything together
│   │   ├── core/
│   │   │   ├── config.py       # All settings (models, DB URL, etc.) in one place
│   │   │   └── security.py     # Auth/session placeholder for pre-launch hardening
│   │   ├── graph/               # <-- the heart of the architecture (PRD section 3)
│   │   │   ├── state.py          # GraphState TypedDict (PRD 6.3)
│   │   │   ├── router.py         # Semantic Router + RoutingDecision schema (PRD 6.1)
│   │   │   ├── nodes.py          # Topic Node creation + isolated context assembly
│   │   │   ├── summarizer.py     # Async global-context summarizer (PRD 6.3)
│   │   │   └── graph_builder.py  # Compiles the LangGraph StateGraph
│   │   ├── api/
│   │   │   ├── websocket.py      # /ws/chat/{session_id} — the main chat loop
│   │   │   └── routes/
│   │   │       ├── chat.py        # Session bootstrap REST endpoints
│   │   │       └── nodes.py       # Node listing / history / manual override
│   │   ├── models/
│   │   │   ├── schemas.py         # Pydantic request/response DTOs
│   │   │   └── db_models.py       # SQLAlchemy ORM models (PRD section 7 ER diagram)
│   │   ├── db/
│   │   │   ├── session.py         # SQLAlchemy engine/session factory
│   │   │   └── checkpointer.py    # LangGraph Postgres checkpoint persistence
│   │   └── services/
│   │       └── llm_service.py     # Wraps the Anthropic client for router + generator
│   └── tests/
│       └── test_router.py         # Proves memory isolation (Month 1 milestone)
│
├── frontend/                 # Next.js 15 (App Router) chat UI
│   ├── package.json
│   ├── next.config.js
│   ├── .env.local.example
│   ├── app/
│   │   ├── layout.tsx           # Root layout, global providers
│   │   └── page.tsx             # Main screen: sidebar graph + chat window
│   ├── components/
│   │   ├── chat/
│   │   │   ├── ChatWindow.tsx     # Message list + input for the active node
│   │   │   └── MessageBubble.tsx  # Single message bubble
│   │   ├── graph-view/
│   │   │   ├── TopicGraphSidebar.tsx  # React Flow map of all topic nodes
│   │   │   └── TopicNodeCard.tsx      # Individual node card in the canvas
│   │   └── ui/                     # shadcn/ui primitives live here
│   ├── lib/
│   │   ├── websocket-client.ts    # Wraps the /ws/chat connection
│   │   └── api-client.ts          # Typed fetch wrappers for the REST routes
│   ├── hooks/
│   │   └── useChatSession.ts       # Session state hook used across components
│   └── public/
│
├── docs/
│   ├── architecture.md         # Deeper design notes beyond the PRD
│   ├── api-spec.md              # Points to FastAPI's auto-generated OpenAPI docs
│   └── diagrams/                # Exported static images, if you need any beyond Mermaid
│
└── scripts/
    ├── setup.sh                # One-shot: env file, Postgres/Redis, deps for both apps
    └── dev.sh                   # Runs backend + frontend concurrently
```

---

## What each top-level piece is for

| Path | Purpose |
|---|---|
| `PRD.md` | Source of truth for *what* to build and *why*. All architecture diagrams live here as Mermaid so they render on GitHub. |
| `AGENTS.md` | Source of truth for *how* an agent should build it — order of operations and hard constraints (e.g. never leak context across nodes). |
| `backend/app/graph/` | The actual Semantic Graph Chat engine — router, isolated node memory, summarizer, and the compiled LangGraph. This is the part worth getting right first; everything else is plumbing around it. |
| `backend/app/api/` | How the outside world (the frontend) talks to the graph — WebSocket for the live chat loop, REST for session/node bookkeeping. |
| `frontend/components/graph-view/` | The visual "mind map" of topics (React Flow), which is what makes this architecture legible to the user instead of just a backend trick. |
| `docker-compose.yml` + `.env.example` | Everything needed to run Postgres (with `pgvector`) and Redis locally without hunting for config values. |
| `scripts/` | Two commands (`setup.sh`, `dev.sh`) so you never have to remember the exact install/run incantations. |

---

## Getting started locally

```bash
cd semantic-graph-chat
cp .env.example .env        # fill in ANTHROPIC_API_KEY at minimum
bash scripts/setup.sh       # installs deps, starts Postgres + Redis
bash scripts/dev.sh         # runs backend on :8000, frontend on :3000
```

Backend health check: `GET http://localhost:8000/health`
Auto-generated API docs (once routes are implemented): `http://localhost:8000/docs`

---

## Status

This is a **scaffold**, matching the "Must Have (MVP)" checklist in
`PRD.md` section 5. Every `.py` and `.ts(x)` file currently contains a
docstring/comment describing its responsibility and a `TODO` / `raise
NotImplementedError` where logic goes — follow the build order in
`AGENTS.md` to fill them in.
