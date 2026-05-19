import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Fake News Verifier",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Fake News Claim Verifier")
st.caption(
    "Powered by **EPRVFL** (Pattern Recognition Letters 2025) + "
    "**Mistral-7B LoRA** (stub mode for demo) + **RAG**"
)
st.divider()

claim = st.text_area(
    "Enter a news claim to verify:",
    height=100,
    placeholder="e.g. COVID vaccines contain microchips",
)

col1, col2 = st.columns([1, 4])
with col1:
    verify_btn = st.button("Verify", type="primary", use_container_width=True)
with col2:
    st.caption("Running in stub mode (no GPU needed)")

if verify_btn and claim.strip():
    if len(claim.strip()) < 10:
        st.warning("Please enter a longer claim (at least 10 characters).")
    else:
        with st.spinner("Agent is analyzing..."):
            try:
                from agent.graph import graph
                from agent.state import AgentState
                
                state: AgentState = {
                    "claim": claim.strip(), "messages": [], "evidence": [],
                    "eprvfl_result": None, "mistral_result": None,
                    "verdict": None, "route": "fast",
                }
                result = graph.invoke(state)
                verdict = result["verdict"]
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        label      = verdict["label"]
        confidence = verdict["confidence"]
        model_used = verdict["model_used"]
        
        color_map = {"FAKE": "🔴", "REAL": "🟢"}
        icon = color_map.get(label, "⚪")

        st.divider()
        st.subheader(f"{icon} Verdict: **{label}**")

        m1, m2 = st.columns(2)
        m1.metric("Confidence",  f"{confidence:.1%}")
        m2.metric("Model",       model_used)

        st.divider()
        st.markdown("#### 📄 Evidence")
        snippets = verdict.get("evidence_snippets", [])
        if snippets:
            for i, snip in enumerate(snippets, 1):
                st.markdown(f"**[{i}]** {snip}")

        st.markdown("#### 🔗 Sources")
        sources = list(dict.fromkeys(verdict.get("sources", [])))
        if sources:
            for src in sources:
                st.markdown(f"- {src}")

elif verify_btn and not claim.strip():
    st.warning("Please enter a claim.")
