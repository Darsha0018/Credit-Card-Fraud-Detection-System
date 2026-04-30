
import sys, os
sys.path.insert(0, ".")

import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.features import add_features, DROP_COLS, cast_cat_to_str

# ─────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="ML-powered fraud scoring for transactions. Returns fraud probability and decision.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # allow Next.js dashboard
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────
# LOAD MODEL ON STARTUP
# ─────────────────────────────────────────────────────────
MODEL_PATH = "models/fraud_xgb.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Run: python src/tune_optuna.py first."
    )

bundle    = joblib.load(MODEL_PATH)
MODEL     = bundle["model"]
THRESHOLD = bundle["threshold"]
PR_AUC    = bundle["pr_auc"]
ROC_AUC   = bundle["roc_auc"]

print(f"✅ Model loaded | PR-AUC={PR_AUC:.4f} | Threshold={THRESHOLD:.2f}")

# ─────────────────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────
class Transaction(BaseModel):
    tx_id:                    str     = Field(...,  example="TX00123456")
    ts:                       str     = Field(...,  example="2024-01-15 14:30:00")
    amount:                   float   = Field(...,  example=850.0)
    merchant_cat:             str     = Field(...,  example="grocery")
    merchant_id_hash:         str     = Field(...,  example="M1234")
    card_id_hash:             str     = Field(...,  example="C56789")
    city:                     str     = Field(...,  example="Mumbai")
    country:                  str     = Field(...,  example="IN")
    device_type:              str     = Field(...,  example="mobile")
    channel:                  str     = Field(...,  example="mobile_app")
    hour:                     int     = Field(...,  example=14)
    dayofweek:                int     = Field(...,  example=0)
    prev_24h_tx_count_card:   float   = Field(...,  example=3.0)
    prev_24h_amt_card:        float   = Field(...,  example=1200.0)
    prev_1h_tx_count_card:    float   = Field(...,  example=1.0)
    velocity_amt_1h:          float   = Field(...,  example=850.0)
    is_international:         bool    = Field(...,  example=False)
    is_night:                 bool    = Field(...,  example=False)

class ScoreResponse(BaseModel):
    tx_id:             str
    fraud_probability: float
    decision:          str       # "ALLOW" or "REVIEW"
    risk_level:        str       # "LOW" / "MEDIUM" / "HIGH"
    threshold_used:    float

# ─────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────
def tx_to_df(tx: Transaction) -> pd.DataFrame:
    d = tx.model_dump()
    d["ts"]       = pd.to_datetime(d["ts"])
    d["is_fraud"] = 0           # placeholder — not known at score time
    return pd.DataFrame([d])

def get_risk_level(prob: float) -> str:
    if prob < 0.20:   return "LOW"
    if prob < 0.50:   return "MEDIUM"
    return "HIGH"

def score_df(df: pd.DataFrame):
    df = cast_cat_to_str(add_features(df))
    X  = df.drop(columns=DROP_COLS)
    return MODEL.predict_proba(X)[:, 1]

# ─────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Check if API is running."""
    return {
        "status":     "ok",
        "timestamp":  datetime.utcnow().isoformat(),
        "model":      "fraud_xgb.joblib",
        "pr_auc":     round(PR_AUC, 4),
        "threshold":  round(THRESHOLD, 2),
    }


@app.get("/model")
def model_info():
    """Get model metadata."""
    return {
        "model_type":  "XGBoost Classifier",
        "version":     "1.0.0",
        "pr_auc":      round(PR_AUC, 4),
        "roc_auc":     round(ROC_AUC, 4),
        "threshold":   round(THRESHOLD, 2),
        "decision_rule": f"REVIEW if prob >= {THRESHOLD:.2f} else ALLOW",
        "top_features": [
            "prev_1h_tx_count_card",
            "velocity_ratio",
            "amt_vs_avg_ratio",
            "prev_24h_tx_count_card",
            "log_amount",
        ],
    }


@app.post("/score", response_model=List[ScoreResponse])
def score_batch(transactions: List[Transaction]):
    """
    Score a batch of transactions.
    Send a list of transaction objects, get back fraud decisions.
    """
    if not transactions:
        raise HTTPException(status_code=400, detail="Empty transaction list.")
    if len(transactions) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 transactions per batch.")

    try:
        df     = pd.concat([tx_to_df(tx) for tx in transactions], ignore_index=True)
        probas = score_df(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")

    responses = []
    for tx, prob in zip(transactions, probas):
        responses.append(ScoreResponse(
            tx_id             = tx.tx_id,
            fraud_probability = round(float(prob), 4),
            decision          = "REVIEW" if prob >= THRESHOLD else "ALLOW",
            risk_level        = get_risk_level(prob),
            threshold_used    = round(THRESHOLD, 2),
        ))
    return responses


@app.post("/stream", response_model=ScoreResponse)
def score_single(tx: Transaction):
    """
    Score a single transaction in real-time.
    Use this endpoint for streaming / webhook integrations.
    """
    try:
        df   = tx_to_df(tx)
        prob = float(score_df(df)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")

    return ScoreResponse(
        tx_id             = tx.tx_id,
        fraud_probability = round(prob, 4),
        decision          = "REVIEW" if prob >= THRESHOLD else "ALLOW",
        risk_level        = get_risk_level(prob),
        threshold_used    = round(THRESHOLD, 2),
    )


# ─────────────────────────────────────────────────────────
# STARTUP MESSAGE
# ─────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_message():
    print("\n" + "="*50)
    print("  Fraud Detection API — RUNNING")
    print("="*50)
    print("  Docs   : http://127.0.0.1:8000/docs")
    print("  Health : http://127.0.0.1:8000/health")
    print("  Score  : POST http://127.0.0.1:8000/score")
    print("  Stream : POST http://127.0.0.1:8000/stream")
    print("="*50 + "\n")