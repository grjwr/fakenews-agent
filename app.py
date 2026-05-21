import os
import sys
sys.path.insert(0, "/app")

# Download FAISS index if not present
if not os.path.exists('data/fakenews_index/index.faiss'):
    print("Downloading FAISS index from HF Datasets...")
    from huggingface_hub import hf_hub_download
    os.makedirs('data/fakenews_index', exist_ok=True)
    hf_hub_download(repo_id="grjwr/fakenews-agent-data", filename="index.faiss", 
                    repo_type="dataset", local_dir='data/fakenews_index')
    hf_hub_download(repo_id="grjwr/fakenews-agent-data", filename="index.pkl", 
                    repo_type="dataset", local_dir='data/fakenews_index')
    print("Index downloaded.")

import streamlit as st
from agent.graph import graph
from agent.state import AgentState

st.title("🔍 Fake News Claim Verifier")
st.caption("EPRVFL + Mistral-7B LoRA + RAG over 108K articles")

claim = st.text_area("Enter a news claim:", height=100)

if st.button("Verify", type="primary"):
    if not claim or len(claim) < 10:
        st.warning("Please enter a longer claim")
    else:
        with st.spinner("Analyzing..."):
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
                st.subheader(f"{icon} **{v['label']}** ({v['confidence']:.0%})")
                st.metric("Model", v['model_used'])
                
                with st.expander("Evidence"):
                    for snip in v.get('evidence_snippets', []):
                        st.write(f"• {snip}")
                
                with st.expander("Sources"):
                    for src in v.get('sources', []):
                        st.write(f"- {src}")
            except Exception as e:
                st.error(f"Error: {e}")

# Add citation footer
st.markdown("---")
st.caption(
    "**Author:** Rajiv Kumar Gurjwar (SVNIT Surat) | "
    "**Paper:** EPRVFL in Pattern Recognition Letters 2025 | "
    "[GitHub](https://github.com/grjwr/fakenews-agent) | "
    "[Scholar Profile](https://scholar.google.com/citations?user=_3_1ExAAAAAJ)"
)

st.markdown("---")
st.caption(
    "**Author:** Rajiv Kumar Gurjwar (SVNIT Surat) | "
    "[GitHub](https://github.com/grjwr/fakenews-agent) | "
    "[Paper](https://scholar.google.com/citations?user=_3_1ExAAAAAJ)"
)

