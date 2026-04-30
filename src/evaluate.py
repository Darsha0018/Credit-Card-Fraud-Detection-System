

import sys, os
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import joblib

from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, roc_curve,
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score
)

from src.features import add_features, DROP_COLS, cast_cat_to_str, NUM_COLS, CAT_COLS

os.makedirs("outputs", exist_ok=True)

print("=" * 55)
print("  STEP 8: FULL EVALUATION & EXPLAINABILITY")
print("=" * 55)

# ─────────────────────────────────────────────────────────
# LOAD MODEL + DATA
# ─────────────────────────────────────────────────────────
print("\n Loading model and validation data...")
bundle = joblib.load("models/fraud_xgb.joblib")
model  = bundle["model"]
TH     = bundle["threshold"]
print(f"   Model loaded   |  Decision threshold: {TH:.2f}")

valid = pd.read_parquet("data/valid.parquet")
valid = cast_cat_to_str(add_features(valid))
X_va  = valid.drop(columns=DROP_COLS)
y_va  = valid["is_fraud"]

# ─────────────────────────────────────────────────────────
# PREDICTIONS
# ─────────────────────────────────────────────────────────
proba  = model.predict_proba(X_va)[:, 1]
preds  = (proba >= TH).astype(int)

tn, fp, fn, tp = confusion_matrix(y_va, preds).ravel()
precision  = precision_score(y_va, preds, zero_division=0)
recall     = recall_score(y_va, preds, zero_division=0)
f1         = f1_score(y_va, preds, zero_division=0)
pr_auc     = average_precision_score(y_va, proba)
roc_auc    = roc_auc_score(y_va, proba)

print(f"""
 EVALUATION RESULTS (threshold = {TH:.2f})
{'─'*45}
  PR-AUC        : {pr_auc:.4f}   ← PRIMARY metric
  ROC-AUC       : {roc_auc:.4f}
  F1-Score      : {f1:.4f}
  Recall        : {recall:.4f}   ({tp} of {tp+fn} frauds caught)
  Precision     : {precision:.4f}   ({fp} false alarms)
{'─'*45}
  True Positives  (caught fraud)  : {tp}
  False Positives (false alarms)  : {fp}
  False Negatives (missed fraud!) : {fn}   ← minimize this
  True Negatives  (correct allow) : {tn}
{'─'*45}""")

print("\nFull Classification Report:")
print(classification_report(y_va, preds, target_names=["Normal","Fraud"]))

# ─────────────────────────────────────────────────────────
# CHART 13 — Confusion Matrix (Final Model)
# ─────────────────────────────────────────────────────────
cm = confusion_matrix(y_va, preds)
fig, ax = plt.subplots(figsize=(7, 6))
fig.suptitle("Chart 13: Final XGBoost — Confusion Matrix", fontsize=14, fontweight="bold")

labels = np.array([[f"TN\n{tn}\nCorrect Normal", f"FP\n{fp}\nFalse Alarm"],
                   [f"FN\n{fn}\nMissed Fraud!", f"TP\n{tp}\nCaught Fraud"]])
sns.heatmap(cm, annot=labels, fmt="", cmap="Blues",
            xticklabels=["Predicted Normal","Predicted Fraud"],
            yticklabels=["Actual Normal","Actual Fraud"],
            linewidths=2, linecolor="white", cbar=False, ax=ax,
            annot_kws={"size": 12})
ax.set_title(f"PR-AUC={pr_auc:.4f}  |  Recall={recall:.4f}  |  Precision={precision:.4f}")
plt.tight_layout()
plt.savefig("outputs/chart13_final_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: outputs/chart13_final_confusion_matrix.png")

# ─────────────────────────────────────────────────────────
# CHART 14 — PR Curve + ROC Curve Side by Side
# ─────────────────────────────────────────────────────────
prec_arr, rec_arr, _  = precision_recall_curve(y_va, proba)
fpr_arr, tpr_arr, _   = roc_curve(y_va, proba)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Chart 14: PR Curve & ROC Curve — Final XGBoost", fontsize=14, fontweight="bold")

# PR Curve
axes[0].plot(rec_arr, prec_arr, color="#e74c3c", linewidth=2.5,
             label=f"XGBoost (PR-AUC={pr_auc:.4f})")
axes[0].scatter([recall], [precision], s=120, color="#2ecc71", zorder=5,
                label=f"Operating point (t={TH:.2f})")
axes[0].axhline(y_va.mean(), linestyle="--", color="gray",
                label=f"Random ({y_va.mean()*100:.1f}%)")
axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
axes[0].set_title("Precision-Recall Curve"); axes[0].legend()
axes[0].set_xlim([0,1]); axes[0].set_ylim([0,1])

# ROC Curve
axes[1].plot(fpr_arr, tpr_arr, color="#3498db", linewidth=2.5,
             label=f"XGBoost (ROC-AUC={roc_auc:.4f})")
axes[1].plot([0,1],[0,1], linestyle="--", color="gray", label="Random")
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve"); axes[1].legend()
axes[1].set_xlim([0,1]); axes[1].set_ylim([0,1])

plt.tight_layout()
plt.savefig("outputs/chart14_pr_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print(" Saved: outputs/chart14_pr_roc_curves.png")

# ─────────────────────────────────────────────────────────
# CHART 15 — SHAP Feature Importance
# ─────────────────────────────────────────────────────────
print("\n Computing SHAP feature importance...")
try:
    import shap
    xgb_model   = model.named_steps["xgb"]
    preprocessor = model.named_steps["pre"]
    X_va_trans  = preprocessor.transform(X_va)

    # get feature names after OHE
    ohe_names = (preprocessor
                 .named_transformers_["cat"]
                 .named_steps["encoder"]
                 .get_feature_names_out(CAT_COLS))
    all_feat_names = list(NUM_COLS) + list(ohe_names)

    # use small sample for speed
    sample_idx  = np.random.choice(len(X_va_trans), size=min(500, len(X_va_trans)), replace=False)
    X_sample    = X_va_trans[sample_idx]

    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)

    # mean absolute SHAP per feature
    mean_shap  = np.abs(shap_values).mean(axis=0)
    feat_imp   = pd.Series(mean_shap, index=all_feat_names).sort_values(ascending=False)
    top15      = feat_imp.head(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("Chart 15: SHAP Feature Importance — Top 15 Drivers", fontsize=14, fontweight="bold")
    colors = ["#e74c3c" if i < 5 else "#3498db" if i < 10 else "#95a5a6"
              for i in range(len(top15))]
    bars = ax.barh(top15.index[::-1], top15.values[::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Mean |SHAP Value| (impact on fraud prediction)")
    ax.set_title("Red = highest impact features")
    for bar, val in zip(bars, top15.values[::-1]):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("outputs/chart15_shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: outputs/chart15_shap_importance.png")
    print(f"\n   Top 5 fraud drivers:")
    for feat, val in top15.head(5).items():
        print(f"     {feat:<30} SHAP={val:.4f}")

except ImportError:
    print("   SHAP not installed. Run: pip install shap")
    print("      Skipping SHAP chart, continuing...")

# ─────────────────────────────────────────────────────────
# CHART 16 — Fraud Score Distribution
# ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Chart 16: Fraud Probability Score Distribution", fontsize=14, fontweight="bold")

ax.hist(proba[y_va==0], bins=60, alpha=0.6, color="#2ecc71",
        label="Normal transactions", density=True)
ax.hist(proba[y_va==1], bins=60, alpha=0.7, color="#e74c3c",
        label="Fraud transactions",  density=True)
ax.axvline(TH, color="black", linestyle="--", linewidth=2,
           label=f"Decision threshold ({TH:.2f})")
ax.set_xlabel("Predicted Fraud Probability")
ax.set_ylabel("Density")
ax.set_title("Good model: fraud scores cluster near 1, normal near 0")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/chart16_score_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: outputs/chart16_score_distribution.png")

# ─────────────────────────────────────────────────────────
# SAVE PREDICTIONS CSV
# ─────────────────────────────────────────────────────────
results_df = valid[["ts","amount","merchant_cat","channel","is_fraud"]].copy()
results_df["fraud_probability"] = np.round(proba, 4)
results_df["decision"]          = np.where(preds == 1, "REVIEW", "ALLOW")
results_df["correct"]           = (preds == y_va.values).astype(int)
results_df.to_csv("outputs/predictions.csv", index=False)
print("   Saved: outputs/predictions.csv")

print(f"""
{'='*55}
  EVALUATION COMPLETE
{'='*55}
  PR-AUC   : {pr_auc:.4f}
  ROC-AUC  : {roc_auc:.4f}
  Recall   : {recall:.4f}  ← caught {tp}/{tp+fn} frauds
  Precision: {precision:.4f}  ← {fp} false alarms

  Charts saved in: outputs/
   Next → python src/save_model.py
{'='*55}
""")