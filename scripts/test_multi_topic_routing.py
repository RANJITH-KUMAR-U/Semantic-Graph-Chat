"""
Fast multi-topic routing live test:
  1. Sends "explain about tree data structures"
  2. Sends "explain about ai engineering and MLOps"
  3. Sends "explain about doomsday and apocalypse theory"
  Immediately proceeds upon receiving "routing" WS frame!
"""
import asyncio
import json
import uuid
import websockets


async def test_multi_topic():
    session_id = f"fast_multi_topic_{uuid.uuid4().hex[:8]}"
    uri = f"ws://localhost:8000/ws/chat/{session_id}"

    prompts = [
        "explain about tree data structures",
        "explain about ai engineering and MLOps",
        "explain about doomsday and apocalypse theory",
    ]

    created_nodes = []

    async with websockets.connect(uri) as ws:
        init_msg = json.loads(await ws.recv())
        print(f"Connected: {init_msg}")

        for idx, prompt in enumerate(prompts):
            print(f"\n--- Turn {idx + 1}: Sending prompt: {prompt!r} ---")
            await ws.send(json.dumps({"content": prompt}))

            assigned_node_id = None
            assigned_node_title = None

            # Read WS frames until we receive "routing" or "done"
            while True:
                raw = await ws.recv()
                msg = json.loads(raw)
                msg_type = msg.get("type")

                if msg_type == "routing" and msg.get("node_id"):
                    assigned_node_id = msg.get("node_id")
                    assigned_node_title = msg.get("node_title")
                    print(f"  Routing assigned -> {assigned_node_id} ({assigned_node_title!r})")
                    break
                elif msg_type == "done":
                    break
                elif msg_type == "error":
                    print(f"  ERROR: {msg.get('content')}")
                    break

            created_nodes.append((assigned_node_id, assigned_node_title))
            # Give backend 1s pause between turns
            await asyncio.sleep(1)

    print("\n=== MULTI-TOPIC ROUTING SUMMARY ===")
    for i, (nid, ntitle) in enumerate(created_nodes, 1):
        print(f"Turn {i}: Node ID = {nid}, Title = {ntitle!r}")

    unique_nodes = {nid for nid, _ in created_nodes if nid}
    print(f"Total unique nodes created across {len(prompts)} distinct topics: {len(unique_nodes)}")
    assert len(unique_nodes) == 3, f"Expected 3 distinct topic nodes, but got {len(unique_nodes)}!"
    print("SUCCESS: Multi-topic routing created 3 distinct topic nodes!")


if __name__ == "__main__":
    asyncio.run(test_multi_topic())
