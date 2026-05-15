import sys, pytest
sys.path.insert(0, ".")
from agent.state import AgentState
from agent.nodes import (
    claim_router, evidence_aggregator,
    model_inference_node, verdict_generator
)

def base_state(claim="COVID vaccines cause autism") -> AgentState:
    return {
        "claim": claim, "messages": [], "evidence": [],
        "eprvfl_result": None, "mistral_result": None,
        "verdict": None, "route": "fast",
    }

def test_claim_router_sets_route():
    state = claim_router(base_state())
    assert state["route"] == "fast"
    assert state["evidence"] == []

def test_claim_router_preserves_claim():
    state = claim_router(base_state("test claim"))
    assert state["claim"] == "test claim"

def test_evidence_aggregator_deduplication():
    state = base_state()
    dup = {"text": "Vaccines are safe and effective.", "source": "url1", "score": 0.3, "label": "REAL"}
    state["evidence"] = [dup, dup, dup]
    state = evidence_aggregator(state)
    assert len(state["evidence"]) == 1

def test_evidence_aggregator_sorts_by_score():
    state = base_state()
    state["evidence"] = [
        {"text": "Evidence A", "source": "s1", "score": 0.9, "label": "FAKE"},
        {"text": "Evidence B", "source": "s2", "score": 0.2, "label": "REAL"},
        {"text": "Evidence C", "source": "s3", "score": 0.5, "label": "FAKE"},
    ]
    state = evidence_aggregator(state)
    scores = [e["score"] for e in state["evidence"]]
    assert scores == sorted(scores)

def test_evidence_aggregator_max_8():
    state = base_state()
    state["evidence"] = [
        {"text": f"Evidence {i}", "source": f"s{i}", "score": float(i), "label": "FAKE"}
        for i in range(20)
    ]
    state = evidence_aggregator(state)
    assert len(state["evidence"]) <= 8

def test_model_inference_fast_route():
    state = base_state("hoax crisis actor deep state conspiracy fake")
    state["evidence"] = [{"text": "Some evidence", "source": "s1", "score": 0.3, "label": "FAKE"}]
    state = model_inference_node(state)
    assert state["eprvfl_result"] is not None
    assert state["eprvfl_result"]["label"] in ["FAKE", "REAL"]
    assert 0.0 <= state["eprvfl_result"]["confidence"] <= 1.0

def test_model_inference_deep_route_triggered():
    # Force low confidence by using neutral claim
    state = base_state("something happened somewhere yesterday")
    state["evidence"] = [{"text": "Some evidence", "source": "s1", "score": 0.3, "label": "REAL"}]
    state = model_inference_node(state)
    # If confidence < 0.75, mistral_result should be set
    if state["eprvfl_result"]["confidence"] < 0.75:
        assert state["mistral_result"] is not None
        assert state["route"] == "deep"

def test_verdict_uses_eprvfl_when_confident():
    state = base_state()
    state["eprvfl_result"] = {"label": "FAKE", "confidence": 0.91, "stub": True}
    state["mistral_result"] = None
    state["evidence"] = [{"text": "Evidence.", "source": "http://x.com", "score": 0.3, "label": "FAKE"}]
    state = verdict_generator(state)
    assert state["verdict"]["label"] == "FAKE"
    assert state["verdict"]["confidence"] == 0.91
    assert "EPRVFL" in state["verdict"]["model_used"]

def test_verdict_prefers_mistral_over_eprvfl():
    state = base_state()
    state["eprvfl_result"] = {"label": "FAKE",      "confidence": 0.65, "stub": True}
    state["mistral_result"] = {"label": "UNCERTAIN", "confidence": 0.82, "stub": True}
    state["evidence"] = [{"text": "Evidence.", "source": "http://x.com", "score": 0.3, "label": "REAL"}]
    state = verdict_generator(state)
    assert state["verdict"]["label"] == "UNCERTAIN"
    assert "Mistral" in state["verdict"]["model_used"]

def test_verdict_contains_required_keys():
    state = base_state()
    state["eprvfl_result"] = {"label": "REAL", "confidence": 0.88, "stub": True}
    state["evidence"] = [{"text": "Evidence.", "source": "http://x.com", "score": 0.3, "label": "REAL"}]
    state = verdict_generator(state)
    for key in ["label", "confidence", "model_used", "route", "sources", "evidence_snippets"]:
        assert key in state["verdict"], f"Missing key: {key}"

def test_verdict_sources_list():
    state = base_state()
    state["eprvfl_result"] = {"label": "FAKE", "confidence": 0.80, "stub": True}
    state["evidence"] = [
        {"text": f"Evidence {i}", "source": f"http://src{i}.com", "score": float(i), "label": "FAKE"}
        for i in range(6)
    ]
    state = verdict_generator(state)
    assert len(state["verdict"]["sources"]) <= 5
