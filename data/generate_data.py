import pandas as pd
import numpy as np

np.random.seed(42)
N = 50_000
FRAUD_RATE = 0.015

n_fraud  = int(N * FRAUD_RATE)
n_normal = N - n_fraud

merchant_cats = ["grocery","electronics","travel","restaurant","clothing",
                 "fuel","pharmacy","entertainment","jewelry","online_gaming"]
cities    = ["Mumbai","Delhi","Bangalore","Chennai","Pune",
             "Hyderabad","Kolkata","Ahmedabad","Jaipur","Lucknow"]
countries = ["IN","US","GB","AE","SG","AU","DE","FR","JP","CA"]
devices   = ["mobile","desktop","tablet","pos_terminal"]
channels  = ["online","in_store","atm","mobile_app"]

def make_transactions(n, is_fraud=False):
    rng = np.random.default_rng(0 if not is_fraud else 1)

    if is_fraud:
        # Fraud — high amounts but with OVERLAP with normal (realistic noise)
        amount     = rng.exponential(scale=4000, size=n).clip(200, 80_000)
        # Mix late-night AND daytime fraud (not just 0-3am)
        hour       = np.where(rng.random(n) < 0.5,
                               rng.choice([0,1,2,3,22,23], size=n),
                               rng.integers(7, 23, size=n))
        country    = rng.choice(countries, size=n)
        is_intl    = rng.choice([True,False], size=n, p=[0.55, 0.45])
        prev24_cnt = rng.integers(2, 20, size=n).astype(float)
        prev24_amt = amount * rng.uniform(1.5, 6, size=n)
        prev1h_cnt = rng.integers(1, 10, size=n).astype(float)
        vel_amt    = amount * rng.uniform(1.2, 5, size=n)
        merchant   = rng.choice(["jewelry","electronics","online_gaming",
                                  "travel","clothing","restaurant"], size=n)
    else:
        # Normal — but some high-value legitimate transactions too
        amount     = rng.exponential(scale=1500, size=n).clip(10, 60_000)
        hour       = rng.integers(7, 23, size=n)
        p_country  = np.array([0.70,0.04,0.04,0.04,0.04,0.03,0.03,0.03,0.03,0.02])
        country    = rng.choice(countries, size=n, p=p_country)
        is_intl    = rng.choice([True,False], size=n, p=[0.10, 0.90])
        prev24_cnt = rng.integers(0, 10, size=n).astype(float)
        prev24_amt = amount * rng.uniform(0.5, 4, size=n)
        prev1h_cnt = rng.integers(0, 4, size=n).astype(float)
        vel_amt    = amount * rng.uniform(0.3, 2, size=n)
        merchant   = rng.choice(merchant_cats, size=n)

    dayofweek = rng.integers(0, 7, size=n)
    city      = rng.choice(cities, size=n)
    device    = rng.choice(devices, size=n)
    channel   = rng.choice(channels, size=n)

    base_ts = pd.Timestamp("2023-01-01")
    seconds = np.sort(rng.integers(0, 365*24*3600, size=n))
    ts = [base_ts + pd.Timedelta(seconds=int(s)) for s in seconds]

    df = pd.DataFrame({
        "tx_id":                  [f"TX{i:08d}" for i in range(n)],
        "ts":                     ts,
        "amount":                 np.round(amount, 2),
        "merchant_cat":           merchant,
        "merchant_id_hash":       [f"M{rng.integers(1000, 9999)}" for _ in range(n)],
        "card_id_hash":           [f"C{rng.integers(10000, 99999)}" for _ in range(n)],
        "city":                   city,
        "country":                country,
        "device_type":            device,
        "channel":                channel,
        "hour":                   hour,
        "dayofweek":              dayofweek,
        "prev_24h_tx_count_card": np.round(prev24_cnt, 1),
        "prev_24h_amt_card":      np.round(prev24_amt, 2),
        "prev_1h_tx_count_card":  np.round(prev1h_cnt, 1),
        "velocity_amt_1h":        np.round(vel_amt, 2),
        "is_international":       is_intl,
        "is_night":               hour < 6,
        "is_fraud":               int(is_fraud),
    })
    return df

normal_df = make_transactions(n_normal, is_fraud=False)
fraud_df  = make_transactions(n_fraud,  is_fraud=True)

df = pd.concat([normal_df, fraud_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("data/transactions.csv", index=False)
print(f"Dataset saved → data/transactions.csv")
print(f"   Total rows : {len(df):,}")
print(f"   Fraud rows : {df.is_fraud.sum():,}  ({df.is_fraud.mean()*100:.2f}%)")
print(f"   Normal rows: {(df.is_fraud==0).sum():,}")