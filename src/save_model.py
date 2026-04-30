

import sys, os, json
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from src.features import add_features, DROP_COLS, cast_cat_to_str

print("=" * 55)
print("  STEP 9: MODEL SAVE & VERIFICATION")
print("=" * 55)

# ─────────────────────────────────────────────────────────
# LOAD & VERIFY
# ─────────────────────────────────────────────────────────
print("\nLoading saved model bundle...")
bundle    = joblib.load("models/fraud_xgb.joblib")
model     = bundle["model"]
TH        = bundle["threshold"]
pr_auc    = bundle["pr_auc"]
roc_auc   = bundle["roc_auc"]
params    = bundle["params"]

print(f"   Model loaded successfully")
print(f"   PR-AUC    : {pr_auc:.4f}")
print(f"   ROC-AUC   : {roc_auc:.4f}")
print(f"   Threshold : {TH:.2f}")

# ─────────────────────────────────────────────────────────
# MOCK PREDICTION — simulates what the API will do
# ─────────────────────────────────────────────────────────
print("\n Running mock predictions...")

mock_transactions = pd.DataFrame([
    {   # NORMAL transaction
        "tx_id": "TEST_001", "ts": "2024-01-15 14:30:00",
        "amount": 850.0, "merchant_cat": "grocery",
        "merchant_id_hash": "M1234", "card_id_hash": "C56789",
        "city": "Mumbai", "country": "IN",
        "device_type": "mobile", "channel": "mobile_app",
        "hour": 14, "dayofweek": 0,
        "prev_24h_tx_count_card": 3.0, "prev_24h_amt_card": 1200.0,
        "prev_1h_tx_count_card": 1.0, "velocity_amt_1h": 850.0,
        "is_international": False, "is_night": False, "is_fraud": 0,
    },
    {   # SUSPICIOUS transaction (high amount, late night, international)
        "tx_id": "TEST_002", "ts": "2024-01-15 02:15:00",
        "amount": 42000.0, "merchant_cat": "jewelry",
        "merchant_id_hash": "M9999", "card_id_hash": "C11111",
        "city": "Delhi", "country": "AE",
        "device_type": "desktop", "channel": "online",
        "hour": 2, "dayofweek": 5,
        "prev_24h_tx_count_card": 18.0, "prev_24h_amt_card": 85000.0,
        "prev_1h_tx_count_card": 7.0, "velocity_amt_1h": 42000.0,
        "is_international": True, "is_night": True, "is_fraud": 1,
    },
    {   # BORDERLINE transaction
        "tx_id": "TEST_003", "ts": "2024-01-15 22:00:00",
        "amount": 5500.0, "merchant_cat": "electronics",
        "merchant_id_hash": "M5678", "card_id_hash": "C22222",
        "city": "Bangalore", "country": "US",
        "device_type": "tablet", "channel": "online",
        "hour": 22, "dayofweek": 6,
        "prev_24h_tx_count_card": 8.0, "prev_24h_amt_card": 12000.0,
        "prev_1h_tx_count_card": 3.0, "velocity_amt_1h": 5500.0,
        "is_international": True, "is_night": False, "is_fraud": 0,
    },
])

mock_transactions["ts"] = pd.to_datetime(mock_transactions["ts"])
mock_feat = cast_cat_to_str(add_features(mock_transactions))
X_mock    = mock_feat.drop(columns=DROP_COLS)
probas    = model.predict_proba(X_mock)[:, 1]
decisions = [" REVIEW (FRAUD)" if p >= TH else " ALLOW (NORMAL)" for p in probas]

print(f"\n   {'Tx':<10} {'Amount':>10} {'Prob':>8} {'Decision'}")
print(f"   {'─'*55}")
for _, row in mock_transactions.iterrows():
    i    = mock_transactions.index.get_loc(_)
    flag = "← flagged correctly!" if (probas[i]>=TH) == bool(row.is_fraud) else "← wrong"
    print(f"   {row.tx_id:<10} ₹{row.amount:>9,.0f} {probas[i]:>7.2%}  {decisions[i]}  {flag}")

# ─────────────────────────────────────────────────────────
# SAVE MODEL CARD
# ─────────────────────────────────────────────────────────
model_card = {
    "model_name":        "Credit Card Fraud Detector",
    "model_type":        "XGBoost Classifier",
    "version":           "1.0.0",
    "trained_on":        "Synthetic transaction data (50,000 rows)",
    "training_period":   "2023-01-01 to 2023-10-21",
    "validation_period": "2023-10-21 to 2023-12-31",
    "saved_at":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "metrics": {
        "pr_auc":        round(pr_auc, 4),
        "roc_auc":       round(roc_auc, 4),
        "threshold":     round(TH, 2),
    },
    "best_hyperparameters": {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in params.items()},
    "features": {
        "total_features_after_ohe": 60,
        "top_shap_drivers": [
            "prev_1h_tx_count_card",
            "velocity_ratio",
            "amt_vs_avg_ratio",
            "prev_24h_tx_count_card",
            "log_amount",
        ]
    },
    "cost_model": {
        "fn_cost_inr": 5000,
        "fp_cost_inr": 50,
        "decision_rule": "REVIEW if prob >= threshold else ALLOW"
    }
}

with open("models/model_card.json", "w") as f:
    json.dump(model_card, f, indent=2)
print(f"\n Saved: models/model_card.json")

# ─────────────────────────────────────────────────────────
# PRINT MODEL CARD
# ─────────────────────────────────────────────────────────
print(f"""
{'='*55}
  MODEL CARD
{'='*55}
  Name      : {model_card['model_name']}
  Type      : {model_card['model_type']}
  Version   : {model_card['version']}
  Saved At  : {model_card['saved_at']}

  METRICS:
    PR-AUC    → {model_card['metrics']['pr_auc']}
    ROC-AUC   → {model_card['metrics']['roc_auc']}
    Threshold → {model_card['metrics']['threshold']}

  TOP FRAUD SIGNALS:
    1. prev_1h_tx_count_card  (high frequency = suspicious)
    2. velocity_ratio         (sudden spend burst)
    3. amt_vs_avg_ratio       (amount vs daily average)
    4. prev_24h_tx_count_card (day-level frequency)
    5. log_amount             (large amounts)

  FILES:
    models/fraud_xgb.joblib   ← trained model pipeline
    models/model_card.json    ← metadata

   Next → python serving/app.py
{'='*55}
""")