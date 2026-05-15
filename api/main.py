import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import time

from agent.graph import graph
from agent.state import AgentState

app = FastAPI(
    title="Fake News Verification Agent",
    description="Agentic fake news detection using EPRVFL + Mistral-7B LoRA + RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request / Response schemas ---

class ClaimRequest(BaseModel):
    claim: str = Field(..., min_length=10, max_length=1000,
                       example="COVID vaccines contain microchips inserted by Bill Gates")

class VerdictResponse(BaseModel):
    claim:             str
    label:             str           # FAKE | REAL
    confidence:        float
    model_used:        str
    route:             str           # fast | deep
    sources:           List[str]
    evidence_snippets: List[str]
    latency_ms:        float

class HealthResponse(BaseModel):
    status:    str
    vector_db: str
    gpu:       str

# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
def health():
    try:
        from agent.tools import get_vectorstore
        get_vectorstore()
        vdb = "ok"
    except Exception as e:
        vdb = f"error: {e}"

    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-only"
    except:
        gpu = "cpu-only"

    return HealthResponse(status="ok", vector_db=vdb, gpu=gpu)


@app.post("/verify", response_model=VerdictResponse)
def verify_claim(req: ClaimRequest):
    if not req.claim.strip():
        raise HTTPException(status_code=400, detail="Claim cannot be empty.")

    t0 = time.time()

    initial_state: AgentState = {
        "claim":          req.claim.strip(),
        "messages":       [],
        "evidence":       [],
        "eprvfl_result":  None,
        "mistral_result": None,
        "verdict":        None,
        "route":          "fast",
    }

    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    verdict = final_state.get("verdict")
    if not verdict:
        raise HTTPException(status_code=500, detail="Agent returned no verdict.")

    latency = round((time.time() - t0) * 1000, 2)

    return VerdictResponse(
        claim             = req.claim,
        label             = verdict["label"],
        confidence        = verdict["confidence"],
        model_used        = verdict["model_used"],
        route             = verdict["route"],
        sources           = verdict["sources"],
        evidence_snippets = verdict["evidence_snippets"],
        latency_ms        = latency,
    )


@app.get("/")
def root():
    return {
        "message": "Fake News Verification Agent API",
        "docs":    "/docs",
        "health":  "/health",
        "verify":  "POST /verify",
    }
