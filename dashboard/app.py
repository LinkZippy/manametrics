import os
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import snowflake.connector
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="MTG Price Predictor", layout="wide")
st.title("MTG Card Price Predictor")
st.caption("Trained on Scryfall data · XGBoost · Features: age, Reserved List, EDHREC rank")

# ---------------------------------------------------------------------------
# Load model and feature list
#
# @st.cache_resource tells Streamlit to load this once and reuse it across
# all user interactions. Without caching, the model would reload from disk
# every time someone types in the search box — wasteful and slow.
# cache_resource is for objects that shouldn't be serialized (models, DB
# connections). cache_data is for plain data like DataFrames.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    model = joblib.load(os.path.join(model_dir, "xgb_model.joblib"))
    features = joblib.load(os.path.join(model_dir, "features.joblib"))
    return model, features


@st.cache_data
def load_data():
    # Suppress the SQLAlchemy warning — we know the raw connector works fine
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
                name,
                set_code,
                original_release_date,
                printing_age_days,
                is_reserved_list,
                edhrec_rank,
                price_usd,
                log_price_usd
            FROM card_price_features
            ORDER BY name
        """, conn)
        conn.close()

    df["IS_RESERVED_LIST"] = df["IS_RESERVED_LIST"].astype(int)
    return df


model, FEATURES = load_model()
df = load_data()

# ---------------------------------------------------------------------------
# Run predictions on all cards
#
# We predict on the whole dataset upfront so the search is instant.
# model.predict() expects a 2D array with columns matching FEATURES exactly.
# The output is log_price_usd — we convert back to dollars with expm1(),
# which is the inverse of log1p() used during feature engineering.
# expm1(x) = e^x - 1, which exactly undoes log1p(x) = ln(x + 1).
# ---------------------------------------------------------------------------
X_all = df[FEATURES].values
df["PREDICTED_LOG_PRICE"] = model.predict(X_all)
df["PREDICTED_PRICE_USD"] = np.expm1(df["PREDICTED_LOG_PRICE"])
df["ACTUAL_PRICE_USD"] = df["PRICE_USD"]

# ---------------------------------------------------------------------------
# Search interface
#
# Streamlit reruns the entire script top-to-bottom on every interaction.
# The search box below triggers a rerun each time the user types, but
# because load_model() and load_data() are cached, they don't re-execute.
# ---------------------------------------------------------------------------
st.subheader("Search for a card")
query = st.text_input("Card name (partial match works)", placeholder="e.g. Black Lotus")

if query:
    # Case-insensitive partial match
    mask = df["NAME"].str.contains(query, case=False, na=False)
    results = df[mask].copy()

    if results.empty:
        st.warning("No cards found matching that name.")
    else:
        st.write(f"Found **{len(results)}** matching cards")

        # Show the key columns in a clean table
        display = results[[
            "NAME", "SET_CODE", "ORIGINAL_RELEASE_DATE",
            "PRINTING_AGE_DAYS", "IS_RESERVED_LIST", "EDHREC_RANK",
            "ACTUAL_PRICE_USD", "PREDICTED_PRICE_USD"
        ]].copy()

        display.columns = [
            "Name", "Set", "Original Release",
            "Age (days)", "Reserved List", "EDHREC Rank",
            "Actual Price ($)", "Predicted Price ($)"
        ]

        # Round prices to 2 decimal places for readability
        display["Actual Price ($)"] = display["Actual Price ($)"].round(2)
        display["Predicted Price ($)"] = display["Predicted Price ($)"].round(2)

        st.dataframe(display, use_container_width=True)

# ---------------------------------------------------------------------------
# Overall model performance chart
#
# A scatter plot of actual vs predicted prices across all cards.
# A perfect model would have all points on the diagonal line y=x.
# Deviation from the diagonal reveals where the model struggles.
# We sample 2000 points so the chart renders quickly.
# ---------------------------------------------------------------------------
st.subheader("Model performance: actual vs predicted price")
st.caption("Each dot is one card. A perfect model would follow the diagonal.")

sample = df.sample(n=min(2000, len(df)), random_state=42)

chart_data = pd.DataFrame({
    "actual_usd": sample["ACTUAL_PRICE_USD"].clip(upper=100),
    "predicted_usd": sample["PREDICTED_PRICE_USD"].clip(upper=100),
})

# Note: st.scatter_chart column names must be plain strings — special characters
# like $ and () cause silent rendering failures where it plots the index instead.
st.scatter_chart(chart_data, x="actual_usd", y="predicted_usd", height=400)

st.caption("Prices clipped at $100 for readability — a small number of cards are much more expensive.")
