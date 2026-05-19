---
title: Fake News Claim Verification Agent
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: streamlit
app_file: app.py
pinned: false
---

# Fake News Claim Verification Agent

LangGraph-based agentic fake news detection combining EPRVFL + Mistral-7B LoRA + RAG.

## Results

| Dataset | Accuracy | F1 |
|---------|----------|-----|
| GossipCop | 0.8200 | 0.8916 |
| WELFake | 0.8500 | 0.8387 |
| LIAR2 | 0.6600 | 0.2609 |

## Architecture

- RAG (FAISS vector store, 86K docs)
- EPRVFL (fast path)
- Mistral-7B LoRA (deep path)
- LangGraph (orchestration)

## Usage

Enter a news claim and the agent verifies it using 108K articles from 6 fake news datasets.
