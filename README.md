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

**Author:** Rajiv Kumar Gurjwar | SVNIT Surat
**Citation:** Gurjwar, R.K., Kumar, A., & Rao, U.P. "EPRVFL: A fast and scalable model for real-time fake news detection." *Pattern Recognition Letters*, vol. 196, pp. 267–273, 2025.

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

## Technical Stack

- **EPRVFL Model:** Transformer + RVFL hybrid (published, Pattern Recognition Letters 2025)
- **LLM:** Mistral-7B-v0.1 fine-tuned with LoRA across 6 datasets
- **RAG:** FAISS vector store (86K embeddings, all-MiniLM-L6-v2)
- **Orchestration:** LangGraph with intelligent confidence routing
- **UI:** Streamlit
- **Data:** 108K articles across 6 fake news datasets (PolitiFact, GossipCop, WELFake, LIAR2, Fake&Real, BuzzFeed)
- **Infrastructure:** SVNIT HPC (NVIDIA H100 NVL GPU)

## Architecture

The agent uses cascading inference:
1. **Fast path:** EPRVFL (< 100ms)
2. **Deep path:** Mistral-7B LoRA (if confidence < 0.75)
3. **Evidence:** FAISS retrieval over 108K articles
4. **Verdict:** Intelligent routing based on confidence scores

## Performance Highlights

- **Speed:** EPRVFL achieves 970-2749x faster inference than Mistral-7B
- **Accuracy:** 85% F1 on cross-domain evaluation
- **Generalization:** Tested on 6 diverse fake news datasets
- **Deployment:** Live on HuggingFace Spaces; dynamically loads 133MB FAISS index

## Repository

**Code:** https://github.com/grjwr/fakenews-agent
**Datasets:** https://huggingface.co/datasets/grjwr/fakenews-agent-data
**Author Profile:** https://scholar.google.com/citations?user=_3_1ExAAAAAJ

## Citation

If you use this agent in research, please cite:

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

## License

MIT License — See LICENSE file

---

**Built with:** PyTorch, LangChain, FAISS, Transformers, PEFT, FastAPI, Streamlit
**Training:** SVNIT HPC, NVIDIA H100 NVL GPU, SLURM
**Evaluation:** 6 fake news datasets, cross-domain benchmarking

