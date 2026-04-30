
import sys
sys.path.insert(0, ".")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.features import NUM_COLS, CAT_COLS


def build_preprocessor() -> ColumnTransformer:
    """
    Returns a fresh ColumnTransformer preprocessor.
    Call this function each time you need a new pipeline
    (avoids state sharing between scripts).
    """

    # ── NUMERICAL PIPELINE ───────────────────────────────────────
    # Step 1: Fill any missing values with column median
    # Step 2: Scale to mean=0, std=1 so all features are comparable
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    # ── CATEGORICAL PIPELINE ─────────────────────────────────────
    # Step 1: Fill missing with most frequent value
    # Step 2: One-Hot Encode → turns "mobile", "desktop" into 0/1 columns
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # ── COLUMN TRANSFORMER ───────────────────────────────────────
    # Applies the right pipeline to the right columns
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, NUM_COLS),
        ("cat", cat_pipeline, CAT_COLS),
    ], remainder="drop")    # drops any column not listed above

    return preprocessor


# ─────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd
    from src.features import add_features, DROP_COLS, cast_cat_to_str

    print("=" * 50)
    print("  STEP 5: PREPROCESSING PIPELINE")
    print("=" * 50)

    train = pd.read_parquet("data/train.parquet")
    train = add_features(train)

    train = cast_cat_to_str(train)
    X = train.drop(columns=DROP_COLS)
    y = train["is_fraud"]

    pre = build_preprocessor()
    X_transformed = pre.fit_transform(X, y)

    print(f"\nInput  shape  : {X.shape}")
    print(f"Output shape  : {X_transformed.shape}  (after OHE expansion)")
    print(f"Target shape  : {y.shape}")
    print(f"Fraud rate    : {y.mean()*100:.2f}%")

    # Show feature names after transformation
    ohe_features = pre.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(CAT_COLS)
    all_features  = list(NUM_COLS) + list(ohe_features)
    print(f"\nTotal features after preprocessing: {len(all_features)}")
    print(f"  Numerical : {len(NUM_COLS)}")
    print(f"  Categorical (OHE expanded): {len(ohe_features)}")
    print(f"\nSample OHE features: {list(ohe_features[:8])}")
    print("\nPipeline working! Next → python src/train_baselines.py")