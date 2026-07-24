import os
import warnings
import numpy as np
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb
import lightgbm as lgb

# load_dotenv reads the .env file and injects its key=value pairs into
# os.environ so we can use os.environ["SNOWFLAKE_ACCOUNT"] etc. below.
# We build the path relative to this file's location so the script works
# regardless of which directory you run it from.
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

# ---------------------------------------------------------------------------
# Pull curated features from Snowflake
#
# We query directly from Snowflake rather than reading the local Parquet
# because Snowflake is the single source of truth for the curated layer.
# ORDER BY original_release_date is critical — it ensures that when we later
# split the data into train/test folds, earlier rows are always "older" cards.
# If we didn't sort here, time-series cross-validation would be meaningless.
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

df = pd.read_sql("""
    SELECT
        original_release_date,
        printing_age_days,
        is_reserved_list,
        edhrec_rank,
        log_price_usd
    FROM card_price_features
    ORDER BY original_release_date
""", conn)
conn.close()

print(f"Loaded {len(df)} rows")

# ---------------------------------------------------------------------------
# Parse and prepare features
#
# Snowflake returned ORIGINAL_RELEASE_DATE as a string. We convert it to
# an actual date so we can use it for splitting (though we don't pass it
# directly as a model feature — it's collinear with printing_age_days).
#
# is_reserved_list came back as bool — XGBoost/LightGBM accept booleans
# but converting to int (0/1) is safer and more explicit.
#
# edhrec_rank stays as float64 with NaN for unranked cards. Both XGBoost
# and LightGBM handle NaN natively by learning which branch is better for
# missing values — we do NOT impute a fake rank like 999999.
# ---------------------------------------------------------------------------
df["ORIGINAL_RELEASE_DATE"] = pd.to_datetime(df["ORIGINAL_RELEASE_DATE"])
df["IS_RESERVED_LIST"] = df["IS_RESERVED_LIST"].astype(int)

# These are the features we pass to the model. We exclude ORIGINAL_RELEASE_DATE
# because it's perfectly correlated with PRINTING_AGE_DAYS (one is derivable
# from the other), and having both would be redundant.
FEATURES = ["PRINTING_AGE_DAYS", "IS_RESERVED_LIST", "EDHREC_RANK"]
TARGET = "LOG_PRICE_USD"

X = df[FEATURES].values
y = df[TARGET].values

# ---------------------------------------------------------------------------
# Time-series cross-validation
#
# TimeSeriesSplit(n_splits=5) divides the data into 5 folds where each
# training set only contains data BEFORE the test set in time order.
# Because we sorted by original_release_date when querying, row index
# directly corresponds to release order — earlier rows = older cards.
#
# Fold structure (conceptually):
#   Fold 1: train on rows 0–15k,  test on rows 15k–30k
#   Fold 2: train on rows 0–30k,  test on rows 30k–45k
#   Fold 3: train on rows 0–45k,  test on rows 45k–60k
#   ... and so on. Training set always grows; test set is always future data.
#
# This mimics real-world deployment: you train on all history up to today,
# then predict prices for cards printed in the future.
# ---------------------------------------------------------------------------
tscv = TimeSeriesSplit(n_splits=5)

print("\n--- XGBoost ---")
xgb_rmses = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = xgb.XGBRegressor(
        n_estimators=500,       # number of trees to build
        learning_rate=0.05,     # how much each tree corrects the previous ones
                                # (lower = slower but more robust)
        max_depth=6,            # how deep each tree grows — controls overfitting
        subsample=0.8,          # use 80% of rows per tree (reduces variance)
        colsample_bytree=0.8,   # use 80% of features per tree (reduces variance)
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    # RMSE on log_price_usd — not on raw price. This means an RMSE of 0.5
    # represents error in log space. To interpret: e^0.5 ≈ 1.65, so we're
    # off by a factor of ~1.65x on average (not $0.50).
    rmse = root_mean_squared_error(y_test, preds)
    xgb_rmses.append(rmse)
    print(f"  Fold {fold+1}: RMSE = {rmse:.4f}")

print(f"  Mean RMSE: {np.mean(xgb_rmses):.4f} ± {np.std(xgb_rmses):.4f}")

print("\n--- LightGBM ---")
lgb_rmses = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # LightGBM is generally faster than XGBoost and often matches or beats
    # it on tabular data. It grows trees leaf-wise (best-first) rather than
    # level-wise, which can find better splits but risks overfitting on small
    # datasets. With 78k rows we're fine.
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,           # suppress LightGBM's verbose output
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    lgb_rmses.append(rmse)
    print(f"  Fold {fold+1}: RMSE = {rmse:.4f}")

print(f"  Mean RMSE: {np.mean(lgb_rmses):.4f} ± {np.std(lgb_rmses):.4f}")

# ---------------------------------------------------------------------------
# Retrain the best model on ALL data and save it
#
# Cross-validation told us how well the model generalizes — now we retrain
# on the full dataset so the saved model has seen as much history as possible.
#
# We use XGBoost here because it had a lower mean RMSE across folds.
# The same hyperparameters are kept so the saved model behaves consistently
# with what we evaluated above.
#
# joblib is the standard way to serialize sklearn-compatible models to disk.
# It's faster and more memory-efficient than Python's built-in pickle for
# large numpy arrays (which is what model weights are stored as internally).
# ---------------------------------------------------------------------------
import joblib

print("\n--- Retraining XGBoost on full dataset ---")
final_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
)
final_model.fit(X, y)

# Save to models/ directory alongside this script.
# The FEATURES list is saved separately so the dashboard knows exactly which
# columns to pass in — if we ever add new features, the model file and
# feature list stay in sync.
model_dir = os.path.dirname(os.path.abspath(__file__))
joblib.dump(final_model, os.path.join(model_dir, "xgb_model.joblib"))
joblib.dump(FEATURES, os.path.join(model_dir, "features.joblib"))

print(f"Saved model  → models/xgb_model.joblib")
print(f"Saved features → models/features.joblib")
