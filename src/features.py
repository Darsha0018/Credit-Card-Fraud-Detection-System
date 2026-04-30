"""
features.py
-----------
Step 4: Feature Engineering
- Creates new meaningful features from raw data
- Velocity ratios, log transforms, weekend flags
- Rare merchant category detection

Used by: train_baselines.py, tune_optuna.py, evaluate.py
"""

import pandas as pd
import numpy as np


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw transaction DataFrame and returns enriched version
    with engineered features. Does NOT modify the original df.
    """
    df = df.copy()

    # ── 1. LOG AMOUNT ────────────────────────────────────────────
    # Fraud amounts are very high → log squishes the scale
    # so model can learn the pattern more easily
    df["log_amount"] = np.log1p(df["amount"])

    # ── 2. AVERAGE SPEND IN LAST 24 HOURS ────────────────────────
    # If someone usually spends ₹500/day but this tx is ₹8000 → suspicious
    df["avg_tx_amt_24h"] = (
        df["prev_24h_amt_card"] / (df["prev_24h_tx_count_card"] + 1e-3)
    )

    # ── 3. VELOCITY RATIO ─────────────────────────────────────────
    # How does this hour's spending compare to the daily average?
    # A very HIGH ratio = sudden burst = fraud signal
    df["velocity_ratio"] = df["velocity_amt_1h"] / (df["avg_tx_amt_24h"] + 1e-3)

    # ── 4. AMOUNT vs AVERAGE RATIO ────────────────────────────────
    # Current transaction amount vs typical daily amount
    df["amt_vs_avg_ratio"] = df["amount"] / (df["avg_tx_amt_24h"] + 1e-3)

    # ── 5. WEEKEND FLAG ───────────────────────────────────────────
    # Fraudsters often strike on weekends (less monitoring)
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # ── 6. RARE MERCHANT CATEGORY ────────────────────────────────
    # Categories with very few transactions may indicate mule merchants
    cat_counts = df["merchant_cat"].value_counts()
    df["merchant_cat_rare"] = (
        df["merchant_cat"].map(lambda x: int(cat_counts.get(x, 0) < 50))
    )

    # ── 7. HIGH FREQUENCY FLAG ───────────────────────────────────
    # More than 5 transactions in the last hour is unusual
    df["high_freq_1h"] = (df["prev_1h_tx_count_card"] > 5).astype(int)

    # ── 8. NIGHT + INTERNATIONAL COMBO ───────────────────────────
    # Fraudsters often make international purchases late at night
    df["night_intl_combo"] = (
        df["is_night"].astype(int) * df["is_international"].astype(int)
    )

    return df


# ─────────────────────────────────────────────────────────
# COLUMN LISTS  (imported by pipeline.py)
# ─────────────────────────────────────────────────────────

# Numerical columns for StandardScaler
NUM_COLS = [
    "amount", "log_amount",
    "prev_24h_tx_count_card", "prev_24h_amt_card",
    "prev_1h_tx_count_card", "velocity_amt_1h",
    "avg_tx_amt_24h", "velocity_ratio", "amt_vs_avg_ratio",
    "hour", "dayofweek",
]

# Categorical columns for OneHotEncoder
CAT_COLS = [
    "merchant_cat", "city", "country", "device_type", "channel",
    "is_international", "is_night", "is_weekend",
    "merchant_cat_rare", "high_freq_1h", "night_intl_combo",
]

# Columns to DROP before training (IDs, timestamps, target)
DROP_COLS = ["tx_id", "ts", "merchant_id_hash", "card_id_hash", "is_fraud"]


# ─────────────────────────────────────────────────────────
# QUICK TEST  (run: python src/features.py)
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, ".")

    df = pd.read_parquet("data/transactions.parquet")
    df_feat = add_features(df)

    new_cols = ["log_amount","avg_tx_amt_24h","velocity_ratio",
                "amt_vs_avg_ratio","is_weekend","merchant_cat_rare",
                "high_freq_1h","night_intl_combo"]

    print("=" * 50)
    print("  STEP 4: FEATURE ENGINEERING")
    print("=" * 50)
    print(f"\nOriginal columns : {df.shape[1]}")
    print(f"Enriched columns : {df_feat.shape[1]}")
    print(f"\nNew features added:")
    for c in new_cols:
        print(f"   {c:<25}  sample → {df_feat[c].head(3).tolist()}")

    print(f"\nFeature stats (fraud vs normal):")
    for c in ["log_amount", "velocity_ratio", "amt_vs_avg_ratio"]:
        fraud_mean  = df_feat[df_feat.is_fraud==1][c].mean()
        normal_mean = df_feat[df_feat.is_fraud==0][c].mean()
        print(f"   {c:<25}  fraud={fraud_mean:.2f}  normal={normal_mean:.2f}")

    print(f"\nFeature engineering working! Total features: {df_feat.shape[1]}")
    print("   Next → python src/pipeline.py")


def cast_cat_to_str(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas category columns to plain strings for sklearn compatibility."""
    df = df.copy()
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].astype(str)
    for col in df.select_dtypes(include="bool").columns:
        df[col] = df[col].astype(int)
    return df