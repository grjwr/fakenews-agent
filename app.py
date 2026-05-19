import gradio as gr
from agent.graph import graph
from agent.state import AgentState

def verify_claim(claim):
    if not claim or len(claim) < 10:
        return "Error: Claim too short", "", ""
    
    state: AgentState = {
        "claim": claim, "messages": [], "evidence": [],
        "eprvfl_result": None, "mistral_result": None,
        "verdict": None, "route": "fast",
    }
    try:
        result = graph.invoke(state)
        v = result["verdict"]
        verdict_text = f"{v['label']} ({v['confidence']:.0%}) via {v['model_used']}"
        evidence = "\n".join([f"• {s}" for s in v.get('evidence_snippets', [])])
        sources = ", ".join(v.get('sources', []))
        return verdict_text, evidence, sources
    except Exception as e:
        return f"Error: {e}", "", ""

interface = gr.Interface(
    fn=verify_claim,
    inputs=gr.Textbox(label="Enter a news claim", lines=3),
    outputs=[
        gr.Textbox(label="Verdict"),
        gr.Textbox(label="Evidence"),
        gr.Textbox(label="Sources"),
    ],
    title="Fake News Verifier",
    description="LangGraph agent using EPRVFL + Mistral-7B LoRA + RAG"
)

interface.launch()
