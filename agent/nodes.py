from agent.state import AgentState
from agent.tools import get_vectorstore
from agent.models.eprvfl_wrapper import EPRVFLWrapper
from agent.models.mistral_wrapper import MistralLoRAWrapper

eprvfl  = EPRVFLWrapper.load()
mistral = MistralLoRAWrapper.load_real(dataset="politifact")

def claim_router(state: AgentState) -> AgentState:
    state["route"]    = "fast"
    state["evidence"] = []
    return state

def rag_retriever_node(state: AgentState) -> AgentState:
    vs = get_vectorstore()
    results = vs.similarity_search_with_score(state["claim"], k=5)
    state["evidence"].extend([
        {
            "text":   doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "label":  doc.metadata.get("label", ""),
            "score":  float(score),
        }
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

def _rag_vote(evidence: list) -> dict:
    """Majority vote from retrieved evidence labels."""
    from collections import Counter
    labels = [e["label"] for e in evidence if e.get("label")]
    if not labels:
        return {"label": "FAKE", "confidence": 0.55}
    counts  = Counter(labels)
    total   = len(labels)
    top     = counts.most_common(1)[0]
    label   = top[0]
    conf    = round(top[1] / total, 3)
    # Boost confidence slightly — RAG vote is a strong signal
    conf    = round(min(conf + 0.10, 0.95), 3)
    return {"label": label, "confidence": conf}

def model_inference_node(state: AgentState) -> AgentState:
    claim         = state["claim"]
    evidence_text = " ".join(e["text"] for e in state["evidence"][:3])

    # Stage 1: EPRVFL stub
    eprvfl_out = eprvfl.predict(claim, evidence_text)

    # Override stub with RAG vote when stub confidence is low
    if eprvfl_out.get("stub") and eprvfl_out["confidence"] < 0.80:
        rag_out = _rag_vote(state["evidence"])
        eprvfl_out["label"]      = rag_out["label"]
        eprvfl_out["confidence"] = rag_out["confidence"]
        eprvfl_out["rag_voted"]  = True

    state["eprvfl_result"] = eprvfl_out

    # Stage 2: Mistral if confidence still low
    if eprvfl_out["confidence"] < 0.75:
        state["route"]          = "deep"
        mistral_out             = mistral.predict(claim, evidence_text)
        if mistral_out.get("stub") and mistral_out["confidence"] < 0.80:
            rag_out = _rag_vote(state["evidence"])
            mistral_out["label"]      = rag_out["label"]
            mistral_out["confidence"] = rag_out["confidence"]
            mistral_out["rag_voted"]  = True
        state["mistral_result"] = mistral_out

    return state

def verdict_generator(state: AgentState) -> AgentState:
    primary    = state.get("mistral_result") or state["eprvfl_result"]
    model_used = "Mistral-7B-LoRA" if state.get("mistral_result") else "EPRVFL"
    if primary.get("stub"):
        model_used += " (stub+RAG)"

    state["verdict"] = {
        "label":             primary["label"],
        "confidence":        primary["confidence"],
        "model_used":        model_used,
        "route":             state["route"],
        "sources":           [e["source"] for e in state["evidence"][:5]],
        "evidence_snippets": [e["text"][:200] for e in state["evidence"][:3]],
        "rag_voted":         primary.get("rag_voted", False),
    }
    return state
