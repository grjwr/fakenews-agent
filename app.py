import os
import sys
sys.path.insert(0, "/app")

import streamlit as st

# Download FAISS index if not present
if not os.path.exists('data/fakenews_index/index.faiss'):
    from huggingface_hub import hf_hub_download
    os.makedirs('data/fakenews_index', exist_ok=True)
    hf_hub_download(repo_id="grjwr/fakenews-agent-data", filename="index.faiss",
                    repo_type="dataset", local_dir='data/fakenews_index')
    hf_hub_download(repo_id="grjwr/fakenews-agent-data", filename="index.pkl",
                    repo_type="dataset", local_dir='data/fakenews_index')

from agent.graph import graph
from agent.state import AgentState

# Header with author details at TOP
st.title("🔍 Fake News Claim Verifier")
st.markdown("""
**Author:** Rajiv Kumar Gurjwar (SVNIT Surat) | 
**Paper:** EPRVFL in Pattern Recognition Letters 2025 | 
[GitHub](https://github.com/grjwr/fakenews-agent) | 
[Scholar Profile](https://scholar.google.com/citations?user=_3_1ExAAAAAJ)
""")
st.markdown("---")
st.caption("EPRVFL + Mistral-7B LoRA + RAG over 108K articles")

# Example claims
st.markdown("#### 💡 Try these examples:")

FAKE_EXAMPLES = [
    "COVID vaccines contain microchips planted by Bill Gates",
    "5G towers are spreading coronavirus to control population",
    "The moon landing was staged by NASA in a Hollywood studio",
    "Drinking bleach cures cancer according to new research",
    "George Soros is funding Antifa to destabilize America",
]

REAL_EXAMPLES = [
    "Scientists confirm climate change is accelerating glacial melting",
    "WHO reports global vaccination rates have improved child mortality",
    "U.S. Federal Reserve raises interest rates to combat inflation",
    "Study shows Mediterranean diet reduces risk of heart disease",
    "NASA confirms water ice found on the Moon's south pole",
]

col1, col2 = st.columns(2)

with col1:
    st.markdown("🔴 **Likely Fake**")
    for example in FAKE_EXAMPLES:
        if st.button(example[:60] + "...", key=f"fake_{example[:20]}"):
            st.session_state.claim = example

with col2:
    st.markdown("🟢 **Likely Real**")
    for example in REAL_EXAMPLES:
        if st.button(example[:60] + "...", key=f"real_{example[:20]}"):
            st.session_state.claim = example

st.markdown("---")

# Text input
claim = st.text_area(
    "Enter a news claim or click an example above:",
    value=st.session_state.get("claim", ""),
    height=100
)

if st.button("🔍 Verify Claim", type="primary"):
    if not claim or len(claim) < 10:
        st.warning("Please enter a longer claim")
    else:
        with st.spinner("Agent is analyzing the claim..."):
            state: AgentState = {
                "claim": claim, "messages": [], "evidence": [],
                "eprvfl_result": None, "mistral_result": None,
                "verdict": None, "route": "fast",
            }
            try:
                result = graph.invoke(state)
                v = result["verdict"]
                color = {"FAKE": "🔴", "REAL": "🟢"}
                icon = color.get(v['label'], "⚪")
                st.subheader(f"{icon} Verdict: **{v['label']}** ({v['confidence']:.0%} confidence)")
                st.metric("Model Used", v['model_used'])
                with st.expander("📄 Evidence Retrieved"):
                    for snip in v.get('evidence_snippets', []):
                        st.write(f"• {snip}")
                with st.expander("🔗 Sources"):
                    for src in v.get('sources', []):
                        st.write(f"- {src}")
            except Exception as e:
                st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.caption("© 2025 Rajiv Kumar Gurjwar | SVNIT Surat | MIT License")
