#!/usr/bin/env bash
# Runs backend (FastAPI/uvicorn) and frontend (Next.js) concurrently for local dev.
set -e
(cd backend && . .venv/bin/activate && uvicorn app.main:app --reload) &
(cd frontend && npm run dev) &
wait
