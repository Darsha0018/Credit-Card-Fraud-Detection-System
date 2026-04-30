
import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────
# 1. SCHEMA DEFINITION
# ─────────────────────────────────────────
SCHEMA = {
    "tx_id":                   "string",
    "amount":                  "float64",
    "merchant_cat":            "category",
    "merchant_id_hash":        "string",
    "card_id_hash":            "string",
    "city":                    "category",
    "country":                 "category",
    "device_type":             "category",
    "channel":                 "category",
    "hour":                    "int64",
    "dayofweek":               "int64",
    "prev_24h_tx_count_card":  "float64",
    "prev_24h_amt_card":       "float64",
    "prev_1h_tx_count_card":   "float64",
    "velocity_amt_1h":         "float64",
    "is_international":        "bool",
    "is_night":                "bool",
    "is_fraud":                "int64",
}

# ─────────────────────────────────────────
# 2. LOAD CSV
# ─────────────────────────────────────────
print("=" * 50)
print("  STEP 2: DATA INGESTION")
print("=" * 50)

csv_path = "data/transactions.csv"

if not os.path.exists(csv_path):
    print("ERROR: data/transactions.csv not found!")
    print("   Please run: python data/generate_data.py first")
    exit(1)

print(f"\n Loading: {csv_path}")
df = pd.read_csv(csv_path)
print(f"   Raw shape: {df.shape}  (rows, columns)")

# ─────────────────────────────────────────
# 3. PARSE TIMESTAMP SEPARATELY
# ─────────────────────────────────────────
df["ts"] = pd.to_datetime(df["ts"])
print(f"   Date range: {df['ts'].min().date()}  →  {df['ts'].max().date()}")

# ─────────────────────────────────────────
# 4. ENFORCE SCHEMA
# ─────────────────────────────────────────
print("\n Enforcing data types...")
df = df.astype(SCHEMA)

# Sort chronologically (important for time-aware train/valid split later)
df = df.sort_values("ts").reset_index(drop=True)
print("   Sorted by timestamp ")

# ─────────────────────────────────────────
# 5. DATA QUALITY CHECKS
# ─────────────────────────────────────────
print("\n Data Quality Report:")
print(f"   Total rows        : {len(df):,}")
print(f"   Total columns     : {df.shape[1]}")
print(f"   Missing values    : {df.isnull().sum().sum()}")
print(f"   Duplicate tx_ids  : {df['tx_id'].duplicated().sum()}")
print(f"   Fraud count       : {df['is_fraud'].sum():,}  ({df['is_fraud'].mean()*100:.2f}%)")
print(f"   Normal count      : {(df['is_fraud']==0).sum():,}")
print(f"\n   Amount stats:")
print(f"     Min    : ₹{df['amount'].min():,.2f}")
print(f"     Max    : ₹{df['amount'].max():,.2f}")
print(f"     Mean   : ₹{df['amount'].mean():,.2f}")
print(f"     Median : ₹{df['amount'].median():,.2f}")

print(f"\n   Column dtypes:")
for col, dtype in df.dtypes.items():
    print(f"     {col:<30} {dtype}")

# ─────────────────────────────────────────
# 6. CATEGORY VALUE COUNTS
# ─────────────────────────────────────────
print(f"\nCategory Distributions:")
for col in ["merchant_cat", "channel", "device_type"]:
    print(f"\n   {col}:")
    print(df[col].value_counts().to_string(header=False))

# ─────────────────────────────────────────
# 7. SAVE AS PARQUET
# ─────────────────────────────────────────
parquet_path = "data/transactions.parquet"
df.to_parquet(parquet_path, index=False)
parquet_size = os.path.getsize(parquet_path) / 1024
csv_size     = os.path.getsize(csv_path) / 1024

print(f"\nSaved: {parquet_path}")
print(f"   CSV size     : {csv_size:,.1f} KB")
print(f"   Parquet size : {parquet_size:,.1f} KB  (compressed )")

# ─────────────────────────────────────────
# 8. PREVIEW
# ─────────────────────────────────────────
print(f"\n First 3 rows preview:")
print(df[["tx_id","ts","amount","merchant_cat","country","is_fraud"]].head(3).to_string(index=False))

print("\nIngestion complete! Next → python notebooks/02_eda.py")