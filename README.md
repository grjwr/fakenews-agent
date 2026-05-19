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

## Demo vs Production

The **HuggingFace Spaces deployment** runs in **stub mode** (keyword heuristics) for CPU compatibility.
- Accuracy: ~50% (baseline)
- Purpose: UI/UX demo only

For **production inference**, use the real Mistral-7B LoRA checkpoint locally on GPU:

```python
from agent.models.mistral_wrapper import MistralLoRAWrapper
mistral = MistralLoRAWrapper.load_real(dataset="politifact")
pred = mistral.predict(claim, evidence)
```

**Real model performance:**
- Accuracy: 85% F1 on GossipCop & WELFake
- Latency: ~200ms per claim (H100 GPU)
- Cross-domain evaluation: 6 datasets tested

## Running Locally

```bash
# GPU node (HPC)
srun --partition=gpu --gres=shard:16 --mem=128G --time=04:00:00 --pty bash
conda activate llm_env
streamlit run app/streamlit_app.py
```

This loads the real Mistral-7B LoRA + FAISS index for production accuracy.
