"""
Live check: OpenRouter LLM + Render backend health + WebSocket
"""
import asyncio
import os
import time
import urllib.request
import json
import dotenv

dotenv.load_dotenv("c:/Users/ranji/Downloads/EasyChat/semantic-graph-chat/backend/.env")

RENDER_BASE = "https://semantic-graph-chat.onrender.com"

# ── 1. Render health ──────────────────────────────────────────────────────────
def check_render_health():
    print("=" * 55)
    print("1. Render backend health check")
    try:
        req = urllib.request.Request(f"{RENDER_BASE}/health", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read().decode()
            print(f"   ✅ HTTP {r.status} → {body[:120]}")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")

# ── 2. OpenRouter direct LLM ─────────────────────────────────────────────────
async def check_openrouter():
    print("=" * 55)
    print("2. OpenRouter LLM direct call (nvidia/nemotron-3.5-lightning:free)")
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    t = time.time()
    try:
        res = await client.chat.completions.create(
            model="nvidia/nemotron-3.5-lightning:free",
            messages=[{"role": "user", "content": "Say OK in one word."}],
            max_tokens=10,
        )
        elapsed = time.time() - t
        print(f"   [OK] Response in {elapsed:.2f}s -> '{res.choices[0].message.content.strip()}'")
    except Exception as e:
        print(f"   [FAIL] FAILED ({time.time()-t:.2f}s): {e}")

# ── 3. Render session create (REST) ──────────────────────────────────────────
def check_render_session():
    print("=" * 55)
    print("3. Render REST API — POST /api/sessions")
    try:
        data = json.dumps({"name": "nighttest"}).encode()
        req = urllib.request.Request(
            f"{RENDER_BASE}/api/sessions",
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read().decode()
            print(f"   ✅ HTTP {r.status} → {body[:120]}")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")

# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    print(f"\n[LIVE CHECK] Full live check at 12:23 AM IST\n")
    check_render_health()
    await check_openrouter()
    check_render_session()
    print("=" * 55)
    print("Done.")

asyncio.run(main())
