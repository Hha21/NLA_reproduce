"""FastAPI server for the NLA web demo.

Run from the repo root:
    uvicorn server.main:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000/ in a browser.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from server.inference import NLAInference


ROOT     = Path(__file__).parent.parent
FRONTEND = ROOT / "frontend"

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["nla"] = NLAInference()
    yield
    state.clear()


app = FastAPI(title="NLA demo", lifespan=lifespan)


# --------------------------------------------------------------------- schemas
class TokenizeRequest(BaseModel):
    text: str


class AnalyzeRequest(BaseModel):
    """Either supply pre-tokenised IDs (preferred, used by the chat flow) or
    raw text (used by the standalone tokenise flow)."""
    text:      str | None       = None
    token_ids: list[int] | None = None
    position:  int              = -1


class ChatMessage(BaseModel):
    role:    Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages:       list[ChatMessage]
    max_new_tokens: int   = Field(200, ge=1, le=1024)
    temperature:    float = Field(0.7, ge=0.0, le=2.0)
    top_p:          float = Field(0.9, gt=0.0, le=1.0)


# --------------------------------------------------------------------- routes
@app.get("/api/health")
def health():
    nla = state.get("nla")
    return {
        "status":        "ok" if nla else "loading",
        "model":         "Qwen2.5-0.5B",
        "checkpoint_av": "models/av.pt",
        "checkpoint_ar": "models/ar.pt",
        "fve_baseline":  ("corpus mean" if nla and nla.corpus_mean is not None
                          else "unavailable"),
    }


@app.post("/api/tokenize")
def tokenize(req: TokenizeRequest):
    return {"tokens": state["nla"].tokenize(req.text)}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    nla = state["nla"]
    try:
        if req.token_ids is not None:
            return nla.analyze_ids(req.token_ids, req.position)
        if req.text is not None:
            return nla.analyze_text(req.text, req.position)
        raise ValueError("must provide either 'text' or 'token_ids'")
    except (ValueError, IndexError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        return state["nla"].chat(
            [m.model_dump() for m in req.messages],
            max_new_tokens = req.max_new_tokens,
            temperature    = req.temperature,
            top_p          = req.top_p,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Static frontend served at the root path. Must be mounted last so it does
# not shadow the /api/* routes above.
app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")
