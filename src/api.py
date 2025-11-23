from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import os
from fastapi.encoders import jsonable_encoder

try:
    from src.infer import score_transaction, score_stream, pipeline
except Exception:
    # allow module import even if models not present yet
    score_transaction = None
    score_stream = None
    pipeline = None

app = FastAPI(title="Fraud Detection Scoring API")

# CORS - allow all origins in dev; set ENV var `ALLOWED_ORIGINS` for production
allowed = os.environ.get("ALLOWED_ORIGINS", "*")
origins = [o.strip() for o in allowed.split(",")] if allowed != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple API-key auth (optional). Set API_KEY env var to enable.
API_KEY = os.environ.get("API_KEY")


class TxnModel(BaseModel):
    amount: float
    timestamp: Optional[str] = None
    txn_type: Optional[str] = None
    recipient_account_id: Optional[str] = None
    # allow extra fields
    class Config:
        extra = "allow"


class ScoreRequest(BaseModel):
    txn: Dict[str, Any]
    account_agg: Optional[Dict[str, Any]] = {}
    alpha: Optional[float] = 0.7


class StreamRequest(BaseModel):
    txn: Dict[str, Any]
    account_history: List[Dict[str, Any]]
    alpha: Optional[float] = 0.7
    alert_threshold: Optional[float] = None


def check_api_key(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health():
    return {"status": "ok", "pipeline_loaded": bool(pipeline is not None)}


@app.post("/score")
async def score(req: ScoreRequest, x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)
    if score_transaction is None:
        raise HTTPException(status_code=503, detail="Scoring models not available. Run training pipeline first.")

    # call score_transaction; ensure we pass plain dicts
    try:
        res = score_transaction(req.txn, req.account_agg or {}, alpha=req.alpha)
        return jsonable_encoder(res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score_stream")
async def score_stream_endpoint(req: StreamRequest, x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)
    if score_stream is None:
        raise HTTPException(status_code=503, detail="Scoring models not available. Run training pipeline first.")

    try:
        # call score_stream with DataFrame-like history (list of dicts accepted)
        res = score_stream(req.txn, __import__("pandas").DataFrame(req.account_history), alpha=req.alpha, alert_threshold=req.alert_threshold)
        return jsonable_encoder(res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
def models_info(x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)
    info = {"pipeline": bool(pipeline is not None)}
    try:
        if pipeline is not None:
            info["feature_count"] = len(pipeline.named_steps.get("rf").feature_importances_) if pipeline and pipeline.named_steps.get("rf") is not None else None
    except Exception:
        info["feature_count"] = None
    return info
