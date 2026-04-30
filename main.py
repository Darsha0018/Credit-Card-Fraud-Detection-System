
import subprocess, sys, os

def run(cmd, label):
    print(f"\n{'='*55}")
    print(f"  RUNNING: {label}")
    print(f"  CMD    : {cmd}")
    print(f"{'='*55}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n FAILED at: {label}")
        print(f"   Fix the error above and re-run: {cmd}")
        sys.exit(1)
    print(f" Done: {label}")

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  CREDIT CARD FRAUD DETECTION — FULL PIPELINE")
    print("="*55)

    run("python data\\generate_data.py",       "Step 1 — Generate Dataset")
    run("python notebooks\\01_ingest.py",      "Step 2 — Data Ingestion")
    run("python notebooks\\02_eda.py",         "Step 3 — EDA + Charts")
    run("python src\\train_baselines.py",      "Step 4 — Baseline Models")
    run("python src\\tune_optuna.py",          "Step 5 — XGBoost Tuning")
    run("python src\\evaluate.py",             "Step 6 — Evaluation + SHAP")
    run("python src\\save_model.py",           "Step 7 — Save & Verify Model")

    print(f"""
{'='*55}
   FULL PIPELINE COMPLETE
{'='*55}
  Dataset  → data/transactions.csv
  Model    → models/fraud_xgb.joblib
  Charts   → outputs/ (16 PNG files)
  Metadata → models/model_card.json

  To start the API:
    uvicorn serving.app:app --reload --port 8000

  Then open:
    http://127.0.0.1:8000/docs
{'='*55}
""")