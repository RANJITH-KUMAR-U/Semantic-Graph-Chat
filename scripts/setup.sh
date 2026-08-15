#!/usr/bin/env bash
# One-shot local dev setup: copies env template, installs backend +
# frontend deps, and brings up Postgres/Redis via docker-compose.
set -e

cp -n .env.example .env || true

echo "==> Starting Postgres + Redis"
docker compose up -d postgres redis

echo "==> Installing backend deps"
cd backend && python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && cd ..

echo "==> Installing frontend deps"
cd frontend && npm install && cd ..

echo "Setup complete. Run 'scripts/dev.sh' to start both servers."
