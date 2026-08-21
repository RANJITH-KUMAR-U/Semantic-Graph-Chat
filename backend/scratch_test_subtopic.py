import asyncio
import logging
from app.graph.graph_builder import build_graph

logging.basicConfig(level=logging.INFO)

async def test_subtopic_flow():
    graph = build_graph()
    config = {"configurable": {"thread_id": "test_subtopic_thread_1"}}

    # Turn 1: Root Topic Creation
    print("\n--- Turn 1: Root Topic ---")
    st1 = {
        "session_id": "test_subtopic_thread_1",
        "nodes": {},
        "active_node_id": "",
        "current_input": "Explain Convolutional Neural Networks (CNNs) in deep learning",
        "routing_decision": {},
        "force_node_id": None,
        "global_summary": "",
        "turn_count": 0,
        "last_response": "",
    }
    res1 = await graph.ainvoke(st1, config)
    nodes1 = res1.get("nodes", {})
    active_id1 = res1.get("active_node_id")
    print(f"Created Root Node ID: {active_id1}")
    print(f"Active Nodes ({len(nodes1)}):")
    for nid, ndata in nodes1.items():
        print(f"  [{nid}] depth={ndata.get('depth')} title={ndata.get('title')!r}")

    # Turn 2: Sub-Topic Creation under Turn 1's Root Node
    print("\n--- Turn 2: Sub-Topic Query ---")
    st2 = {
        "session_id": "test_subtopic_thread_1",
        "nodes": nodes1,
        "active_node_id": active_id1,
        "current_input": "How does max pooling work in CNN feature extraction?",
        "routing_decision": {},
        "force_node_id": None,
        "global_summary": res1.get("global_summary", ""),
        "turn_count": 1,
        "last_response": "",
    }
    res2 = await graph.ainvoke(st2, config)
    nodes2 = res2.get("nodes", {})
    active_id2 = res2.get("active_node_id")
    print(f"Active Node after Turn 2: {active_id2}")
    print(f"All Nodes after Turn 2 ({len(nodes2)}):")
    for nid, ndata in nodes2.items():
        print(f"  [{nid}] depth={ndata.get('depth')} title={ndata.get('title')!r} parent={ndata.get('parent_node_id')!r}")

    assert len(nodes2) >= 2, "Sub-topic node was not created!"
    sub_node = nodes2.get(active_id2)
    print(f"\nSUCCESS: Sub-topic created cleanly!")
    print(f"  Sub-Topic Title: {sub_node.get('title')!r}")
    print(f"  Parent Root Node: {sub_node.get('parent_node_id')!r}")

if __name__ == "__main__":
    asyncio.run(test_subtopic_flow())
