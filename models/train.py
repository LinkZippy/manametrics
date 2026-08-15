import os
import warnings
import numpy as np
import pandas as pd
import joblib
import snowflake.connector
from dotenv import load_dotenv
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

# ---------------------------------------------------------------------------
# Feature list — keep this in sync with:
#   1. spark/transform.py  (columns written to Snowflake)
#   2. dashboard/inventory.py  (columns selected for prediction)
#
# Why these features:
#   PRINTING_AGE_DAYS   - older cards are rarer/harder to reprint
#   REPRINT_COUNT       - more reprints = more supply = lower price
#   IS_RESERVED_LIST    - WotC can't reprint → scarcity floor
#   EDHREC_RANK         - Commander demand proxy (nulls = no rank, left as-is;
#                         XGBoost/LightGBM handle missing values natively)
#   RARITY_RANK         - 0=common → 3=mythic; controls print run size
#   IS_MASTERS_SET      - limited-print reprint product → holds value
#   IS_DRAFT_INNOVATION - Horizons-style sets inject staples into eternal formats
#   LEGAL_*             - each legal format = another demand pool bidding on the card
#   IS_*                (card type flags) - different types have different demand curves
# ---------------------------------------------------------------------------
FEATURES = [
    "PRINTING_AGE_DAYS",
    "REPRINT_COUNT",
    "IS_RESERVED_LIST",
    "EDHREC_RANK",
    "RARITY_RANK",
    "IS_MASTERS_SET",
    "IS_DRAFT_INNOVATION",
    "LEGAL_MODERN",
    "LEGAL_LEGACY",
    "LEGAL_VINTAGE",
    "LEGAL_COMMANDER",
    "IS_CREATURE",
    "IS_INSTANT",
    "IS_SORCERY",
    "IS_ENCHANTMENT",
    "IS_ARTIFACT",
    "IS_PLANESWALKER",
]

TARGET = "LOG_PRICE_USD"

# ---------------------------------------------------------------------------
# Pull curated features from Snowflake.
#
# ORDER BY original_release_date is critical: TimeSeriesSplit needs rows
# sorted chronologically so that "future" data never leaks into training
# folds. Shuffling prices from 2024 into a training fold that predicts 2019
# would be data leakage — the model would see future market conditions.
# ---------------------------------------------------------------------------
conn = snowflake.connector.connect(
    account=os.environ["SNOWFLAKE_ACCOUNT"],
    user=os.environ["SNOWFLAKE_USER"],
    password=os.environ["SNOWFLAKE_PASSWORD"],
    warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
    database=os.environ["SNOWFLAKE_DATABASE"],
    schema=os.environ["SNOWFLAKE_SCHEMA"],
    role=os.environ["SNOWFLAKE_ROLE"],
)

query_cols = ", ".join([TARGET, "ORIGINAL_RELEASE_DATE"] + FEATURES)
df = pd.read_sql(f"""
    SELECT {query_cols}
    FROM card_price_features
    ORDER BY original_release_date
""", conn)
conn.close()

print(f"Loaded {len(df)} rows from Snowflake")
print(df.dtypes)
print(df.head())

# Snowflake returns column names uppercase — already accounted for in FEATURES.
X = df[FEATURES]
y = df[TARGET]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Null counts per feature:\n{X.isnull().sum()}")

# ---------------------------------------------------------------------------
# Cross-validation with TimeSeriesSplit.
#
# Why TimeSeriesSplit instead of random KFold?
# Card price data is temporal. If we randomly shuffle and split:
#   - A training fold might include a card's 2024 price
#   - The test fold asks us to "predict" its 2021 price
# That's backwards in time, which is unrealistic and inflates performance.
#
# TimeSeriesSplit keeps time order: train on early dates, test on later ones.
# n_splits=5 gives 5 progressively larger training windows.
# ---------------------------------------------------------------------------
tscv = TimeSeriesSplit(n_splits=5)

xgb_rmse_scores = []
lgb_rmse_scores = []

xgb_params = {
    "objective": "reg:squarederror",
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "enable_categorical": False,
}

lgb_params = {
    "objective": "regression",
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "verbose": -1,
}

print("\n=== TimeSeriesSplit Cross-Validation ===")
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # XGBoost
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_val)
    xgb_rmse = root_mean_squared_error(y_val, xgb_preds)
    xgb_rmse_scores.append(xgb_rmse)

    # LightGBM
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train, y_train)
    lgb_preds = lgb_model.predict(X_val)
    lgb_rmse = root_mean_squared_error(y_val, lgb_preds)
    lgb_rmse_scores.append(lgb_rmse)

    print(f"  Fold {fold+1}  |  train={len(train_idx):>6}  val={len(val_idx):>6}  "
          f"XGB RMSE={xgb_rmse:.4f}  LGB RMSE={lgb_rmse:.4f}")

print(f"\nXGBoost  Mean RMSE: {np.mean(xgb_rmse_scores):.4f} ± {np.std(xgb_rmse_scores):.4f}")
print(f"LightGBM Mean RMSE: {np.mean(lgb_rmse_scores):.4f} ± {np.std(lgb_rmse_scores):.4f}")

# ---------------------------------------------------------------------------
# Final model: retrain XGBoost on the full dataset.
#
# The CV above told us how well the model generalises across time.
# Now we train on ALL data so the model can use every price signal available
# before we freeze it for the dashboard.
# ---------------------------------------------------------------------------
print("\n=== Training final XGBoost on full data ===")
final_model = xgb.XGBRegressor(**xgb_params)
final_model.fit(X, y)

# ---------------------------------------------------------------------------
# Feature importance — useful sanity check.
# A well-specified model should put RARITY_RANK, REPRINT_COUNT, and the
# format legality flags near the top. If PRINTING_AGE_DAYS dominates
# everything else, the new features may not be loading correctly.
# ---------------------------------------------------------------------------
importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": final_model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\nFeature importances (XGBoost):")
print(importance.to_string(index=False))

# ---------------------------------------------------------------------------
# Sanity-check on a known expensive card: Black Lotus (oracle_id).
# We create a synthetic row with expected feature values and check that
# the model predicts something astronomically high.
# ---------------------------------------------------------------------------
print("\n--- Sanity check: synthetic Black Lotus row ---")
black_lotus_features = {
    "PRINTING_AGE_DAYS":  int((pd.Timestamp.now() - pd.Timestamp("1993-08-05")).days),
    "REPRINT_COUNT":      1,     # has never been reprinted
    "IS_RESERVED_LIST":   1,
    "EDHREC_RANK":        np.nan,  # not Commander-legal
    "RARITY_RANK":        2,     # rare in Alpha
    "IS_MASTERS_SET":     0,
    "IS_DRAFT_INNOVATION":0,
    "LEGAL_MODERN":       0,
    "LEGAL_LEGACY":       0,
    "LEGAL_VINTAGE":      1,     # restricted in Vintage
    "LEGAL_COMMANDER":    0,
    "IS_CREATURE":        0,
    "IS_INSTANT":         0,
    "IS_SORCERY":         0,
    "IS_ENCHANTMENT":     0,
    "IS_ARTIFACT":        1,
    "IS_PLANESWALKER":    0,
}
lotus_X = pd.DataFrame([black_lotus_features])[FEATURES]
lotus_pred_log = final_model.predict(lotus_X)[0]
lotus_pred_usd = np.expm1(lotus_pred_log)
print(f"  Predicted price: ${lotus_pred_usd:,.2f}  (log target = {lotus_pred_log:.4f})")

# ---------------------------------------------------------------------------
# Persist model and feature list.
#
# Both files are gitignored (*.joblib).
# The dashboard loads them with joblib.load() at startup.
# Saving FEATURES alongside the model ensures the dashboard always uses
# exactly the same columns the model was trained on — no silent mismatch.
# ---------------------------------------------------------------------------
models_dir = os.path.dirname(os.path.abspath(__file__))
model_path    = os.path.join(models_dir, "xgb_model.joblib")
features_path = os.path.join(models_dir, "features.joblib")

joblib.dump(final_model, model_path)
joblib.dump(FEATURES, features_path)

print(f"\nSaved model   → {model_path}")
print(f"Saved features → {features_path}")
print("Done.")
