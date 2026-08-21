import asyncio
import logging
from app.graph.graph_builder import build_graph

logging.basicConfig(level=logging.INFO)

async def test_subtopic_branch_flow():
    graph = build_graph()
    config = {"configurable": {"thread_id": "test_subtopic_branch_thread"}}

    # Turn 1: Root Topic Creation
    print("\n--- Turn 1: Root Topic ---")
    st1 = {
        "session_id": "test_subtopic_branch_thread",
        "nodes": {},
        "active_node_id": "",
        "current_input": "Explain Convolutional Neural Networks (CNNs)",
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
    for nid, ndata in nodes1.items():
        print(f"  [{nid}] depth={ndata.get('depth')} title={ndata.get('title')!r}")

    # Turn 2: Sub-Topic Branch Query
    print("\n--- Turn 2: Sub-Topic Branch Query ---")
    st2 = {
        "session_id": "test_subtopic_branch_thread",
        "nodes": nodes1,
        "active_node_id": active_id1,
        "current_input": "Create a sub-topic focus on comparing Max Pooling vs Average Pooling implementations in PyTorch",
        "routing_decision": {},
        "force_node_id": None,
        "global_summary": res1.get("global_summary", ""),
        "turn_count": 1,
        "last_response": "",
    }
    res2 = await graph.ainvoke(st2, config)
    nodes2 = res2.get("nodes", {})
    active_id2 = res2.get("active_node_id")
    print(f"\nActive Node after Turn 2: {active_id2}")
    print(f"All Nodes after Turn 2 ({len(nodes2)}):")
    for nid, ndata in nodes2.items():
        print(f"  [{nid}] depth={ndata.get('depth')} title={ndata.get('title')!r} parent={ndata.get('parent_node_id')!r}")

if __name__ == "__main__":
    asyncio.run(test_subtopic_branch_flow())
