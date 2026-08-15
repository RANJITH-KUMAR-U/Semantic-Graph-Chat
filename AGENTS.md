# AGENTS.md — Standing Instructions for Antigravity Agents

Every agent spawned in this workspace reads this file before starting work.
Full spec lives in `PRD.md` — read that first for architecture and rationale.

## Build Order (follow the roadmap in PRD.md section 9)
1. **Backend core** — implement `backend/app/graph/*` in this order:
   `state.py` → `router.py` → `nodes.py` → `graph_builder.py` → `summarizer.py`.
   Prove memory isolation with `backend/tests/test_router.py` before moving on.
2. **Persistence** — wire `backend/app/db/checkpointer.py` (Postgres) and
   `backend/app/services/llm_service.py`.
3. **API** — implement `backend/app/api/websocket.py` (primary chat loop)
   then `backend/app/api/routes/*.py`.
4. **Frontend** — implement `frontend/hooks/useChatSession.ts` and
   `frontend/lib/*` first, then the components that consume them.

## Non-negotiable constraints
- **Never let one Topic Node's LLM call see another node's messages.**
  `assemble_context()` in `backend/app/graph/nodes.py` is the isolation
  boundary — do not widen it "for convenience."
- **Router output must be structured (Pydantic/tool-calling), never free text.**
  See `RoutingDecision` in `backend/app/graph/router.py`.
- **The global summarizer runs async and must never block the main
  response path** (PRD section 6.3).
- On router low-confidence, default to the currently active node rather
  than creating a new one (PRD section 6.4).

## Coding standards
- Python: PEP 8, type hints everywhere, every function gets a docstring.
- TypeScript: functional components only, no class components.
- No hardcoded secrets — everything sensitive comes from `.env` (see `.env.example`).
- One responsibility per file — do not collapse graph/router/nodes logic
  back into `main.py`.

## Definition of done for the MVP
All "Must Have" checkboxes in `PRD.md` section 5 are checked, and
`backend/tests/test_router.py` demonstrates zero cross-node context leakage.
