from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import (
    claim_router, rag_retriever_node,
    evidence_aggregator, model_inference_node, verdict_generator
)

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("claim_router",        claim_router)
    g.add_node("rag_retriever",       rag_retriever_node)
    g.add_node("evidence_aggregator", evidence_aggregator)
    g.add_node("model_inference",     model_inference_node)
    g.add_node("verdict_generator",   verdict_generator)

    g.set_entry_point("claim_router")
    g.add_edge("claim_router",        "rag_retriever")
    g.add_edge("rag_retriever",       "evidence_aggregator")
    g.add_edge("evidence_aggregator", "model_inference")
    g.add_edge("model_inference",     "verdict_generator")
    g.add_edge("verdict_generator",   END)

    return g.compile()

graph = build_graph()
