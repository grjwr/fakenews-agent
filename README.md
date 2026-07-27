---
title: Fake News Claim Verification Agent
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: streamlit
app_file: app.py
pinned: false
authors:
  - name: Rajiv Kumar Gurjwar
    url: https://scholar.google.com/citations?user=_3_1ExAAAAAJ
tags:
  - fake-news
  - nlp
  - transformers
  - mistral
  - rag
  - langgraph
---

# Fake News Claim Verification Agent

**Author:** Rajiv Kumar Gurjwar (SVNIT Surat)  
**Paper:** [EPRVFL: A fast and scalable model for real-time fake news detection](https://scholar.google.com/citations?user=_3_1ExAAAAAJ) — *Pattern Recognition Letters* 2025

LangGraph-based agentic fake news verification combining EPRVFL + Mistral-7B LoRA + RAG over 108K news articles.

## Results

| Dataset | Accuracy | F1 | Precision | Recall |
|---------|----------|-----|-----------|--------|
| GossipCop | 0.8200 | 0.8916 | 0.8409 | 0.9487 |
| WELFake | 0.8500 | 0.8387 | 0.8667 | 0.8125 |
| LIAR2 | 0.6600 | 0.2609 | 0.2609 | 0.2609 |
| Fake&Real | 0.3700 | 0.5039 | 0.6400 | 0.4156 |
| PolitiFact | 0.3571 | 0.3883 | 0.4255 | 0.3571 |
| BuzzFeed | 0.3333 | 0.5000 | 0.4000 | 0.6667 |

## Architecture

- **EPRVFL:** Transformer + RVFL hybrid (published, Pattern Recognition Letters 2025)
- **LLM:** Mistral-7B-v0.1 fine-tuned with LoRA across 6 datasets
- **RAG:** FAISS vector store (86K embeddings)
- **Orchestration:** LangGraph with intelligent routing
- **Infrastructure:** SVNIT HPC (H100 GPU)

## Repository

**Code:** https://github.com/grjwr/fakenews-agent  
**Live Demo:** [https://huggingface.co/datasets/grjwr/fakenews-agent-data](https://huggingface.co/spaces/grjwr/fakenews-agent)

## Citation

```bibtex
@article{gurjwar2025eprvfl,
  title={EPRVFL: A fast and scalable model for real-time fake news detection},
  author={Gurjwar, Rajiv Kumar and Kumar, Arun and Rao, Uday Pratap},
  journal={Pattern Recognition Letters},
  volume={196},
  pages={267--273},
  year={2025}
}
```

