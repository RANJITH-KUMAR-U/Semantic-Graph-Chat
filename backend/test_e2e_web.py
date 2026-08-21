"""
E2E test: Create session -> Connect WebSocket -> Send message -> Check LLM streams back
"""
import asyncio
import json
import urllib.request
import urllib.error

RENDER_BASE = "https://semantic-graph-chat.onrender.com"

def create_session():
    print("[1] Creating session via REST...")
    data = json.dumps({"name": "webtest"}).encode()
    req = urllib.request.Request(
        f"{RENDER_BASE}/api/sessions",
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "testclient/1.0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            body = json.loads(r.read().decode())
            session_id = body["session_id"]
            print(f"   OK  session_id={session_id}")
            return session_id
    except Exception as e:
        print(f"   FAIL: {e}")
        return None

async def test_websocket(session_id):
    import websockets
    ws_url = f"wss://semantic-graph-chat.onrender.com/ws/chat/{session_id}"
    print(f"[2] Connecting WebSocket: {ws_url}")
    try:
        async with websockets.connect(ws_url, open_timeout=20) as ws:
            print("   OK  WebSocket connected")
            # Send a message
            payload = json.dumps({"content": "explain python in 3 sentences", "force_node_id": None})
            await ws.send(payload)
            print("[3] Message sent. Waiting for LLM stream (90s timeout)...")
            chunks = []
            deadline = asyncio.get_event_loop().time() + 90
            async for msg in ws:
                data = json.loads(msg)
                mtype = data.get("type", "")
                # Log ALL event types for debugging
                if mtype == "token":
                    chunks.append(data.get("content", ""))
                    print(".", end="", flush=True)
                elif mtype == "routing":
                    print(f"\n   [routing] {data.get('content','')}")
                elif mtype == "node_created":
                    print(f"\n   [node_created] id={data.get('node_id','?')} label={data.get('label','?')}")
                elif mtype == "node_update":
                    print(f"\n   [node_update] id={data.get('node_id','?')}")
                elif mtype == "stream_start":
                    print(f"\n   [stream_start] model={data.get('model','?')}")
                elif mtype == "stream_end":
                    print(f"\n   [stream_end]")
                    break
                elif mtype == "done":
                    print(f"\n   [done] node_id={data.get('node_id','?')} title={data.get('node_title','?')}")
                    break  # Full response complete — exit cleanly
                elif mtype == "summary_status":
                    gs = data.get("global_summary", "")
                    if gs:
                        print(f"\n   [summary] {gs[:80]}...")
                    else:
                        print(f"\n   [summary] summarizing={data.get('is_summarizing')}")
                elif mtype == "error":
                    print(f"\n   [ERROR] {data.get('content','')}")
                    break
                else:
                    print(f"\n   [UNKNOWN event: {mtype}] {str(data)[:80]}")
                if asyncio.get_event_loop().time() > deadline:
                    print("\n   TIMEOUT (90s) waiting for response")
                    break

            full_response = "".join(chunks)
            print(f"\n[4] Full LLM Response ({len(full_response)} chars):")
            print("   " + full_response[:400])
            if full_response:
                print("\n   VERDICT: WORKING - LLM responded successfully!")
            else:
                print("\n   VERDICT: BROKEN - No tokens received!")
    except Exception as e:
        print(f"   FAIL WebSocket error: {type(e).__name__}: {e}")

async def main():
    print("=" * 55)
    print("E2E Web Test - Vercel + Render + OpenRouter")
    print("=" * 55)

    # First check if websockets package available
    try:
        import websockets
    except ImportError:
        print("Installing websockets...")
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "websockets"], check=True)
        import websockets

    session_id = create_session()
    if not session_id:
        print("ABORT: Could not create session")
        return

    await test_websocket(session_id)
    print("=" * 55)

asyncio.run(main())
