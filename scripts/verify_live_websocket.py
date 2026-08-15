"""
Live end-to-end test against the running WebSocket server on http://localhost:8000
"""
import asyncio
import json
import uuid
import websockets


async def test_live_websocket():
    session_id = f"live_test_{uuid.uuid4().hex[:8]}"
    uri = f"ws://localhost:8000/ws/chat/{session_id}"

    print(f"Connecting to live WebSocket: {uri}...")
    async with websockets.connect(uri) as ws:
        # 1. Expect connected message
        raw_msg = await ws.recv()
        data = json.loads(raw_msg)
        print(f"Received handshake: {data}")
        assert data["type"] == "connected"
        assert data["session_id"] == session_id

        # 2. Send chat prompt
        prompt = "Hello! What is FastAPI and how does it compare to Flask?"
        print(f"Sending prompt: {prompt!r}")
        await ws.send(json.dumps({"content": prompt}))

        # 3. Receive messages until done
        received_tokens = []
        node_id = None
        node_title = None

        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "routing":
                print(f"[WS EVENT: routing] {msg.get('content')}")
                if msg.get("node_id"):
                    node_id = msg.get("node_id")
                    node_title = msg.get("node_title")
            elif msg_type == "token":
                token = msg.get("content", "")
                received_tokens.append(token)
                print(token, end="", flush=True)
            elif msg_type == "done":
                print(f"\n[WS EVENT: done] Finished generation for node: {msg.get('node_id')} ({msg.get('node_title')})")
                break
            elif msg_type == "error":
                print(f"\n[WS ERROR] {msg.get('content')}")
                break

        full_reply = "".join(received_tokens)
        print(f"\nCompleted response length: {len(full_reply)} chars.")
        print(f"Assigned node: {node_id} ('{node_title}')")
        assert len(full_reply) > 0, "No response received!"
        assert node_id is not None, "No topic node assigned!"
        print("LIVE WEBSOCKET TEST PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    asyncio.run(test_live_websocket())
