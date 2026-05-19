import sys, os
sys.path.insert(0, "/app")

# Build vector store if missing
if not os.path.exists("data/fakenews_index/index.faiss"):
    print("Building vector store...")
    os.system("python scripts/build_vectorstore.py")

from agent.graph import graph
from agent.state import AgentState

state: AgentState = {
    "claim": "COVID vaccines contain microchips",
    "messages": [], "evidence": [],
    "eprvfl_result": None, "mistral_result": None,
    "verdict": None, "route": "fast",
}
result = graph.invoke(state)
print(f"Verdict: {result['verdict']['label']}")
