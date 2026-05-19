# Fake News Claim Verification Agent (Phase 3)

LangGraph-based agentic fake news detection combining EPRVFL + Mistral-7B LoRA + RAG over 108K news articles.

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

- **RAG** — FAISS vector store (86K training docs, all-MiniLM embeddings)
- **EPRVFL** — Fast path (your published model, stub mode for now)
- **Mistral-7B LoRA** — Deep path (confidence routing if EPRVFL < 0.75)
- **LangGraph** — Agentic orchestration with parallel tool execution

## API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{"claim": "COVID vaccines contain microchips"}'
```

## UI

```bash
streamlit run app/streamlit_app.py --server.port 8501
```

Open http://localhost:8501

## Datasets

6 fake news datasets combined: PolitiFact, GossipCop, WELFake, BuzzFeed, LIAR2, Fake&Real News (108,303 total articles, 86K training split).

## Code

- `agent/graph.py` — LangGraph state graph
- `agent/nodes.py` — all node functions
- `agent/models/` — EPRVFL + Mistral wrappers
- `api/main.py` — FastAPI backend
- `app/streamlit_app.py` — Streamlit UI
- `scripts/` — preprocessing, vector store build, evaluation
