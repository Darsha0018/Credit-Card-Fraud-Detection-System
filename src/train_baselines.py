
import sys, os
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, confusion_matrix,
    classification_report, f1_score
)

from src.features import add_features, DROP_COLS, cast_cat_to_str
from src.pipeline import build_preprocessor

os.makedirs("outputs", exist_ok=True)

print("=" * 55)
print("  STEP 6: BASELINE MODEL TRAINING")
print("=" * 55)

# ─────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────
print("\n Loading train/valid splits...")
train = pd.read_parquet("data/train.parquet")
valid = pd.read_parquet("data/valid.parquet")

train = cast_cat_to_str(add_features(train))
valid = cast_cat_to_str(add_features(valid))

X_tr = train.drop(columns=DROP_COLS);  y_tr = train["is_fraud"]
X_va = valid.drop(columns=DROP_COLS);  y_va = valid["is_fraud"]

print(f"   Train: {X_tr.shape}  |  Fraud: {y_tr.sum()} ({y_tr.mean()*100:.2f}%)")
print(f"   Valid: {X_va.shape}  |  Fraud: {y_va.sum()} ({y_va.mean()*100:.2f}%)")

# ─────────────────────────────────────────────────────────
# DEFINE MODELS
# ─────────────────────────────────────────────────────────
models = {
    "LogisticRegression": Pipeline([
        ("pre", build_preprocessor()),
        ("clf", LogisticRegression(
            max_iter=500,
            class_weight="balanced",   # handles imbalance
            C=0.1,
            solver="lbfgs",
            random_state=42
        ))
    ]),
    "RandomForest": Pipeline([
        ("pre", build_preprocessor()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced",   # handles imbalance
            random_state=42,
            n_jobs=-1
        ))
    ]),
}

# ─────────────────────────────────────────────────────────
# TRAIN & EVALUATE
# ─────────────────────────────────────────────────────────
results = {}

for name, model in models.items():
    print(f"\n Training: {name}...")
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_va)[:, 1]
    preds = (proba >= 0.5).astype(int)

    pr_auc  = average_precision_score(y_va, proba)
    roc_auc = roc_auc_score(y_va, proba)
    f1      = f1_score(y_va, preds)
    cm      = confusion_matrix(y_va, preds)
    tn, fp, fn, tp = cm.ravel()
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    results[name] = {
        "model": model, "proba": proba,
        "pr_auc": pr_auc, "roc_auc": roc_auc,
        "f1": f1, "recall": recall, "precision": precision,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "cm": cm
    }

    print(f"   PR-AUC    : {pr_auc:.4f}")
    print(f"   ROC-AUC   : {roc_auc:.4f}")
    print(f"   F1-Score  : {f1:.4f}")
    print(f"   Recall    : {recall:.4f}  (caught {tp}/{tp+fn} frauds)")
    print(f"   Precision : {precision:.4f}")
    print(f"   TP={tp}  FP={fp}  FN={fn}  TN={tn}")

# ─────────────────────────────────────────────────────────
# CHART 8 — Confusion Matrices Side by Side
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Chart 8: Confusion Matrices — Baseline Models", fontsize=14, fontweight="bold")

for ax, (name, r) in zip(axes, results.items()):
    labels_cm = [["TN\n(Correct Normal)", "FP\n(False Alarm)"],
                 ["FN\n(Missed Fraud!)", "TP\n(Caught Fraud)"]]
    annot = np.array([[f"{r['tn']}\n{labels_cm[0][0]}", f"{r['fp']}\n{labels_cm[0][1]}"],
                      [f"{r['fn']}\n{labels_cm[1][0]}", f"{r['tp']}\n{labels_cm[1][1]}"]])
    sns.heatmap(r["cm"], annot=annot, fmt="", cmap="Blues",
                xticklabels=["Predicted Normal","Predicted Fraud"],
                yticklabels=["Actual Normal","Actual Fraud"], ax=ax,
                linewidths=1, linecolor="white", cbar=False)
    ax.set_title(f"{name}\nPR-AUC={r['pr_auc']:.3f}  Recall={r['recall']:.3f}")

plt.tight_layout()
plt.savefig("outputs/chart8_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n    Saved: outputs/chart8_confusion_matrices.png")

# ─────────────────────────────────────────────────────────
# CHART 9 — Precision-Recall Curves
# ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
fig.suptitle("Chart 9: Precision-Recall Curves", fontsize=14, fontweight="bold")

colors = ["#3498db", "#e74c3c"]
for (name, r), color in zip(results.items(), colors):
    prec, rec, _ = precision_recall_curve(y_va, r["proba"])
    ax.plot(rec, prec, color=color, linewidth=2,
            label=f"{name}  (PR-AUC={r['pr_auc']:.3f})")

baseline = y_va.mean()
ax.axhline(baseline, linestyle="--", color="gray",
           label=f"Random baseline ({baseline*100:.1f}%)")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Higher area = better model at catching fraud")
ax.legend(loc="upper right")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig("outputs/chart9_pr_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: outputs/chart9_pr_curves.png")

# ─────────────────────────────────────────────────────────
# CHART 10 — Model Comparison Bar Chart
# ─────────────────────────────────────────────────────────
metrics = ["pr_auc", "roc_auc", "f1", "recall", "precision"]
labels  = ["PR-AUC", "ROC-AUC", "F1", "Recall", "Precision"]
x = np.arange(len(metrics)); width = 0.3

fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle("Chart 10: Model Comparison", fontsize=14, fontweight="bold")

for i, (name, r) in enumerate(results.items()):
    vals = [r[m] for m in metrics]
    bars = ax.bar(x + i*width, vals, width, label=name, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{v:.3f}", ha="center", fontsize=8)

ax.set_xticks(x + width/2); ax.set_xticklabels(labels)
ax.set_ylabel("Score"); ax.set_ylim([0, 1.1])
ax.legend(); ax.axhline(0.9, linestyle=":", color="gray", alpha=0.5)
plt.tight_layout()
plt.savefig("outputs/chart10_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: outputs/chart10_model_comparison.png")

# ─────────────────────────────────────────────────────────
# WINNER SELECTION
# ─────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["pr_auc"])
print(f"""
{'='*55}
  BASELINE RESULTS SUMMARY
{'='*55}
  {'Model':<22} {'PR-AUC':>8} {'ROC-AUC':>9} {'Recall':>8}
  {'-'*50}""")
for name, r in results.items():
    star = " ← BEST" if name == best_name else ""
    print(f"  {name:<22} {r['pr_auc']:>8.4f} {r['roc_auc']:>9.4f} {r['recall']:>8.4f}{star}")
print(f"""
  Winner: {best_name}
  Next step: XGBoost + Optuna tuning will improve this further.
   Next → python src/tune_optuna.py
{'='*55}
""")