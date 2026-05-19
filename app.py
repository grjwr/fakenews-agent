import sys
sys.path.insert(0, "/app")

from agent.graph import graph
from agent.state import AgentState

# Simple test
state: AgentState = {
    "claim": "COVID vaccines contain microchips",
    "messages": [], "evidence": [],
    "eprvfl_result": None, "mistral_result": None,
    "verdict": None, "route": "fast",
}
result = graph.invoke(state)
v = result["verdict"]
print(f"Verdict: {v['label']} ({v['confidence']:.0%})")
