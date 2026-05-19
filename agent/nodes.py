from agent.state import AgentState
from agent.tools import get_vectorstore
from agent.models.eprvfl_wrapper import EPRVFLWrapper
from agent.models.mistral_wrapper import MistralLoRAWrapper

eprvfl  = EPRVFLWrapper.load()
mistral = MistralLoRAWrapper.load_stub()   # Stub mode for HF Spaces CPU

def claim_router(state: AgentState) -> AgentState:
    state["route"] = "fast"
    state["evidence"] = []
    return state

def rag_retriever_node(state: AgentState) -> AgentState:
    vs = get_vectorstore()
    results = vs.similarity_search_with_score(state["claim"], k=5)
    state["evidence"].extend([
        {"text": doc.page_content, "source": doc.metadata.get("source", "unknown"),
         "label": doc.metadata.get("label", ""), "score": float(score)}
        for doc, score in results
    ])
    return state

def evidence_aggregator(state: AgentState) -> AgentState:
    seen, ranked = set(), []
    for ev in state["evidence"]:
        key = ev["text"][:80]
        if key not in seen:
            seen.add(key)
            ranked.append(ev)
    ranked.sort(key=lambda x: x["score"])
    state["evidence"] = ranked[:8]
    return state

def model_inference_node(state: AgentState) -> AgentState:
    claim = state["claim"]
    evidence_text = " ".join(e["text"] for e in state["evidence"][:3])
    eprvfl_out = eprvfl.predict(claim, evidence_text)
    state["eprvfl_result"] = eprvfl_out
    if eprvfl_out["confidence"] < 0.75:
        state["route"] = "deep"
        mistral_out = mistral.predict(claim, evidence_text)
        state["mistral_result"] = mistral_out
    return state

def verdict_generator(state: AgentState) -> AgentState:
    primary = state.get("mistral_result") or state["eprvfl_result"]
    model_used = "Mistral-7B-LoRA" if state.get("mistral_result") else "EPRVFL"
    if primary.get("stub"):
        model_used += " (stub)"
    state["verdict"] = {
        "label": primary["label"],
        "confidence": primary["confidence"],
        "model_used": model_used,
        "route": state["route"],
        "sources": [e["source"] for e in state["evidence"][:5]],
        "evidence_snippets": [e["text"][:200] for e in state["evidence"][:3]],
    }
    return state
