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


load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

# ---------------------------------------------------------------------------
# Pull curated features from Snowflake
#
# We query directly from Snowflake rather than reading the local Parquet
# because Snowflake is the single source of truth for the curated layer.
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
print(df.dtypes)
print(df.head())
