

import sys, os
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    confusion_matrix, precision_recall_curve
)
from xgboost import XGBClassifier
import joblib

from src.features import add_features, DROP_COLS, cast_cat_to_str
from src.pipeline import build_preprocessor

os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("=" * 55)
print("  STEP 7: XGBOOST TUNING WITH OPTUNA")
print("=" * 55)

# ─────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────
print("\n Loading data...")
train = pd.read_parquet("data/train.parquet")
valid = pd.read_parquet("data/valid.parquet")

train = cast_cat_to_str(add_features(train))
valid = cast_cat_to_str(add_features(valid))

X_tr = train.drop(columns=DROP_COLS);  y_tr = train["is_fraud"]
X_va = valid.drop(columns=DROP_COLS);  y_va = valid["is_fraud"]

print(f"   Train: {X_tr.shape} | Fraud: {y_tr.sum()} ({y_tr.mean()*100:.2f}%)")
print(f"   Valid: {X_va.shape} | Fraud: {y_va.sum()} ({y_va.mean()*100:.2f}%)")

# ─────────────────────────────────────────────────────────
# CLASS WEIGHT FOR IMBALANCE
# ─────────────────────────────────────────────────────────
# XGBoost uses scale_pos_weight = normal_count / fraud_count
# This tells the model: "treat each fraud as X times more important"
pos_weight = (len(y_tr) - y_tr.sum()) / max(1, y_tr.sum())
print(f"\n scale_pos_weight = {pos_weight:.1f}x  (fraud weighted higher)")

# ─────────────────────────────────────────────────────────
# OPTUNA OBJECTIVE FUNCTION
# ─────────────────────────────────────────────────────────
def objective(trial):
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators", 200, 700),
        max_depth         = trial.suggest_int("max_depth", 3, 8),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        subsample         = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_lambda        = trial.suggest_float("reg_lambda", 0.0, 10.0),
        reg_alpha         = trial.suggest_float("reg_alpha", 0.0, 2.0),
        min_child_weight  = trial.suggest_int("min_child_weight", 1, 10),
        scale_pos_weight  = pos_weight,
        tree_method       = "hist",
        random_state      = 42,
        n_jobs            = -1,
        eval_metric       = "aucpr",
        verbosity         = 0,
    )
    pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("xgb", XGBClassifier(**params))
    ])
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_va)[:, 1]
    return average_precision_score(y_va, proba)   # maximize PR-AUC

# ─────────────────────────────────────────────────────────
# RUN TUNING (30 trials — takes ~2-3 mins)
# ─────────────────────────────────────────────────────────
print("\n🔍 Running Optuna (30 trials) — please wait...")
print("   Each trial trains XGBoost with different hyperparameters")
print("   and picks the one with best PR-AUC on validation set.\n")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=False)

best_params = study.best_params
best_score  = study.best_value

print(f" Tuning complete!")
print(f"   Best PR-AUC     : {best_score:.4f}")
print(f"   Best Parameters :")
for k, v in best_params.items():
    print(f"     {k:<22} = {v}")

# ─────────────────────────────────────────────────────────
# TRAIN FINAL MODEL WITH BEST PARAMS
# ─────────────────────────────────────────────────────────
print("\n🏋️  Training final XGBoost with best params...")

final_params = {**best_params,
                "scale_pos_weight": pos_weight,
                "tree_method": "hist",
                "random_state": 42,
                "n_jobs": -1,
                "eval_metric": "aucpr",
                "verbosity": 0}

final_model = Pipeline([
    ("pre", build_preprocessor()),
    ("xgb", XGBClassifier(**final_params))
])
final_model.fit(X_tr, y_tr)
proba = final_model.predict_proba(X_va)[:, 1]

pr_auc  = average_precision_score(y_va, proba)
roc_auc = roc_auc_score(y_va, proba)
print(f"   Final PR-AUC  : {pr_auc:.4f}")
print(f"   Final ROC-AUC : {roc_auc:.4f}")

# ─────────────────────────────────────────────────────────
# COST-OPTIMAL THRESHOLD SELECTION
# ─────────────────────────────────────────────────────────
# FN = missed fraud  → bank loses ₹5000 per missed fraud
# FP = false alarm   → customer friction costs ₹50
# We find the threshold that MINIMIZES total cost
print("\n Finding cost-optimal decision threshold...")

FN_COST = 5000   # cost of missing a fraud (₹)
FP_COST = 50     # cost of false alarm (₹)

best_t, best_cost = 0.5, float("inf")
threshold_results = []

for t in np.linspace(0.01, 0.99, 99):
    yhat = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_va, yhat).ravel()
    cost = fn * FN_COST + fp * FP_COST
    threshold_results.append({"threshold": t, "cost": cost,
                               "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                               "recall": tp/(tp+fn+1e-9),
                               "precision": tp/(tp+fp+1e-9)})
    if cost < best_cost:
        best_cost = cost
        best_t    = t

t_df = pd.DataFrame(threshold_results)
opt  = t_df[t_df.threshold == best_t].iloc[0]

print(f"   Chosen threshold : {best_t:.2f}")
print(f"   Total cost       : ₹{int(best_cost):,}")
print(f"   TP={int(opt.tp)}  FP={int(opt.fp)}  FN={int(opt.fn)}  TN={int(opt.tn)}")
print(f"   Recall           : {opt.recall:.4f}")
print(f"   Precision        : {opt.precision:.4f}")

# ─────────────────────────────────────────────────────────
# CHART 11 — Optuna Trial History
# ─────────────────────────────────────────────────────────
trial_values = [t.value for t in study.trials]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Chart 11: Optuna Tuning Results", fontsize=14, fontweight="bold")

axes[0].plot(trial_values, color="#3498db", alpha=0.6, marker="o", markersize=3)
axes[0].axhline(best_score, color="#e74c3c", linestyle="--",
                label=f"Best = {best_score:.4f}")
axes[0].set_xlabel("Trial Number")
axes[0].set_ylabel("PR-AUC")
axes[0].set_title("PR-AUC per Trial")
axes[0].legend()

axes[1].plot(t_df.threshold, t_df.cost, color="#e74c3c", linewidth=2)
axes[1].axvline(best_t, color="#2ecc71", linestyle="--",
                label=f"Optimal threshold = {best_t:.2f}")
axes[1].set_xlabel("Decision Threshold")
axes[1].set_ylabel("Total Cost (₹)")
axes[1].set_title("Cost vs Threshold\n(FN=₹5000, FP=₹50)")
axes[1].legend()

plt.tight_layout()
plt.savefig("outputs/chart11_optuna_tuning.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n   Saved: outputs/chart11_optuna_tuning.png")

# ─────────────────────────────────────────────────────────
# CHART 12 — PR Curve of Final XGBoost
# ─────────────────────────────────────────────────────────
prec_arr, rec_arr, thr_arr = precision_recall_curve(y_va, proba)

fig, ax = plt.subplots(figsize=(9, 6))
fig.suptitle("Chart 12: Final XGBoost — Precision-Recall Curve", fontsize=14, fontweight="bold")
ax.plot(rec_arr, prec_arr, color="#e74c3c", linewidth=2.5,
        label=f"XGBoost (PR-AUC = {pr_auc:.4f})")
ax.scatter([opt.recall], [opt.precision], color="#2ecc71", s=120, zorder=5,
           label=f"Chosen threshold ({best_t:.2f})")
ax.axhline(y_va.mean(), linestyle="--", color="gray",
           label=f"Random baseline ({y_va.mean()*100:.1f}%)")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Green dot = cost-optimal operating point")
ax.legend(); ax.set_xlim([0,1]); ax.set_ylim([0,1])
plt.tight_layout()
plt.savefig("outputs/chart12_xgb_pr_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: outputs/chart12_xgb_pr_curve.png")

# ─────────────────────────────────────────────────────────
# SAVE MODEL BUNDLE
# ─────────────────────────────────────────────────────────
bundle = {
    "model":      final_model,
    "threshold":  float(best_t),
    "pr_auc":     float(pr_auc),
    "roc_auc":    float(roc_auc),
    "params":     best_params,
}
joblib.dump(bundle, "models/fraud_xgb.joblib")
print(f"\n Saved: models/fraud_xgb.joblib")

print(f"""
{'='*55}
  TUNING COMPLETE
{'='*55}
  PR-AUC        : {pr_auc:.4f}
  ROC-AUC       : {roc_auc:.4f}
  Threshold     : {best_t:.2f}
  Recall        : {opt.recall:.4f}
  Precision     : {opt.precision:.4f}
  Total Cost    : ₹{int(best_cost):,}

   Next → python src/evaluate.py
{'='*55}
""")