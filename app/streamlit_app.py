import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Fake News Verifier",
    page_icon="🔍",
    layout="centered"
)

# --- Header ---
st.title("🔍 Fake News Claim Verifier")
st.caption(
    "Powered by **EPRVFL** (Pattern Recognition Letters 2025) + "
    "**Mistral-7B LoRA** + **RAG** over 108K news articles"
)
st.divider()

# --- Input ---
claim = st.text_area(
    "Enter a news claim to verify:",
    height=100,
    placeholder="e.g. COVID vaccines contain microchips inserted by Bill Gates",
)

col1, col2 = st.columns([1, 4])
with col1:
    verify_btn = st.button("Verify", type="primary", use_container_width=True)
with col2:
    st.caption("Response time ~100–500ms on CPU, ~50ms on GPU")

# --- Examples ---
with st.expander("Try an example claim"):
    examples = [
        "COVID vaccines contain microchips inserted by Bill Gates",
        "NASA confirms water ice found on the moon surface",
        "Study published in Nature confirms new cancer treatment works",
        "Deep state conspiracy to cover up alien contact confirmed",
        "Barack Obama was not born in the United States",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state["example_claim"] = ex
            st.rerun()

# Auto-fill example if selected
if "example_claim" in st.session_state:
    claim = st.session_state.pop("example_claim")

# --- Verification ---
if verify_btn and claim.strip():
    if len(claim.strip()) < 10:
        st.warning("Please enter a longer claim (at least 10 characters).")
    else:
        with st.spinner("Agent is searching for evidence..."):
            try:
                resp = requests.post(
                    f"{API_URL}/verify",
                    json={"claim": claim.strip()},
                    timeout=120,
                )
                resp.raise_for_status()
                result = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure the FastAPI server is running on port 8000.")
                st.stop()
            except requests.exceptions.Timeout:
                st.error("Request timed out. The model may be loading — try again.")
                st.stop()
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        # --- Verdict display ---
        label      = result["label"]
        confidence = result["confidence"]
        model_used = result["model_used"]
        route      = result["route"]
        latency    = result["latency_ms"]

        color_map = {"FAKE": "🔴", "REAL": "🟢", "UNCERTAIN": "🟡"}
        icon = color_map.get(label, "⚪")

        st.divider()
        st.subheader(f"{icon} Verdict: **{label}**")

        m1, m2, m3 = st.columns(3)
        m1.metric("Confidence",  f"{confidence:.1%}")
        m2.metric("Model",       model_used.replace(" (stub)", ""))
        m3.metric("Latency",     f"{latency:.0f} ms")

        if route == "deep":
            st.info("🔄 Low initial confidence — escalated to Mistral-7B LoRA for deeper analysis.")

        # --- Evidence ---
        st.divider()
        st.markdown("#### 📄 Evidence Snippets")
        snippets = result.get("evidence_snippets", [])
        if snippets:
            for i, snip in enumerate(snippets, 1):
                st.markdown(f"**[{i}]** {snip}")
        else:
            st.caption("No evidence snippets retrieved.")

        # --- Sources ---
        st.markdown("#### 🔗 Sources")
        sources = list(dict.fromkeys(result.get("sources", [])))  # deduplicate
        if sources:
            source_labels = {
                "politifact": "PolitiFact",
                "gossipcop":  "GossipCop",
                "welfake":    "WELFake",
                "buzzfeed":   "BuzzFeed",
                "liar2":      "LIAR2",
                "fake_real":  "Fake & Real News",
            }
            for src in sources:
                label_name = source_labels.get(src, src)
                st.markdown(f"- 📚 {label_name}")
        else:
            st.caption("No sources found.")

        # --- Raw JSON (for debugging) ---
        with st.expander("Raw API response (JSON)"):
            st.json(result)

elif verify_btn and not claim.strip():
    st.warning("Please enter a claim before clicking Verify.")

# --- Sidebar ---
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "This agent verifies news claims using:\n"
        "- **EPRVFL** — Published model (Pattern Recognition Letters, 2025)\n"
        "- **Mistral-7B LoRA** — Fine-tuned on 6 fake news datasets\n"
        "- **RAG** — 86K indexed news articles via FAISS\n"
        "- **LangGraph** — Agentic pipeline with confidence routing"
    )
    st.divider()
    st.markdown("### API Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"API: {health['status'].upper()}")
        st.caption(f"Vector DB: {health['vector_db']}")
        st.caption(f"GPU: {health['gpu']}")
    except:
        st.error("API offline")

    st.divider()
    st.markdown("### Datasets")
    st.caption("PolitiFact · GossipCop · WELFake · BuzzFeed · LIAR2 · Fake & Real News")
    st.caption("Total: 108,303 articles")
