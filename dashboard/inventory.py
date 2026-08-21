"""
MTG Inventory Dashboard
========================
Run with:
    streamlit run ~/mtg-price-predictor/dashboard/inventory.py
"""

import io
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import snowflake.connector
import streamlit as st
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
GAIN_SELL_PCT  = 35
HOLD_MIN_DAYS  = 180
LOSING_FLOOR   = 0.85
UNDERVAL_RATIO = 1.20
EDHREC_POPULAR = 2000
EDHREC_CHEAP   = 15.0

# ---------------------------------------------------------------------------
# Page config + custom CSS — dark futuristic theme
# ---------------------------------------------------------------------------
st.set_page_config(page_title="MANAMETRICS", layout="wide")

st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #080c14;
    color: #c8d8e8;
    font-family: 'JetBrains Mono', 'Courier New', monospace;
}
[data-testid="stHeader"] { background: transparent; }

/* ── Sidebar / file uploader area ── */
[data-testid="stFileUploader"] {
    background: #0d1525;
    border: 1px solid #1e3a5f;
    border-radius: 4px;
    padding: 12px;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #0d1525;
    border: 1px solid #1a3050;
    border-radius: 4px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] {
    color: #4a90b8 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    color: #e8f4ff !important;
    font-size: 1.4rem !important;
    font-weight: 700;
}
[data-testid="stMetricDelta"] svg { display: none; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
    background: transparent;
    color: #4a90b8;
    border: none;
    border-bottom: 2px solid transparent;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 10px 20px;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00c8ff;
    border-bottom: 2px solid #00c8ff;
    background: rgba(0,200,255,0.05);
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1a3050;
    border-radius: 4px;
}

/* ── Headings ── */
h1 { color: #00c8ff !important; letter-spacing: 0.15em; font-size: 1.6rem !important; }
h2 { color: #4a90b8 !important; font-size: 1rem !important; letter-spacing: 0.1em;
     border-left: 3px solid #00c8ff; padding-left: 10px; margin-top: 24px; }
h3 { color: #7ab8d8 !important; font-size: 0.85rem !important; letter-spacing: 0.08em; }

/* ── Caption / info ── */
[data-testid="stCaptionContainer"] { color: #3a6080 !important; font-size: 0.7rem; }

/* ── Divider ── */
hr { border-color: #1a3050; }

/* ── Selectbox / checkbox ── */
[data-testid="stSelectbox"] > div,
[data-testid="stCheckbox"] { color: #4a90b8; }

/* ── Warning / info banners ── */
[data-testid="stAlert"] {
    background: #0d1e30 !important;
    border: 1px solid #1e4060 !important;
    color: #7ab8d8 !important;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# MANAMETRICS")
st.markdown(
    "<p style='color:#3a6080;font-size:0.72rem;letter-spacing:0.08em;margin-top:-12px'>"
    "LIVE PRICES // SNOWFLAKE · DAILY AIRFLOW PIPELINE · XGBOOST v1"
    "</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Buddhist Era → CE date conversion
# ---------------------------------------------------------------------------
def be_to_ce(date_str: str) -> str | None:
    try:
        y, m, d = str(date_str).strip().split("-")
        return f"{int(y) - 543}-{m}-{d}"
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Parse inventory CSV
# ---------------------------------------------------------------------------
@st.cache_data
def parse_inventory(raw_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw_bytes))
    df.columns = [c.strip() for c in df.columns]
    df = df[df["Card Name"].notna()].copy()
    df["Date CE"] = pd.to_datetime(df["Date Bought"].apply(be_to_ce), errors="coerce")
    df["Price Bought (USD)"] = pd.to_numeric(df["Price Bought (USD)"], errors="coerce")
    today = pd.Timestamp.today().normalize()
    df["Days Held"] = (today - df["Date CE"]).dt.days
    df["_name_key"] = df["Card Name"].str.strip().str.lower()
    df["_set_key"]  = df["Set Code"].str.strip().str.lower()
    df["is_foil"]   = df["Printing"].str.strip().str.lower() == "foil"
    # Collector number is optional — present in exports from Moxfield/TCGPlayer etc.
    if "Card Number" in df.columns:
        df["_collector_key"] = df["Card Number"].astype(str).str.strip().str.lower()
    else:
        df["_collector_key"] = None
    return df

# ---------------------------------------------------------------------------
# Load Snowflake cards
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_snowflake_cards() -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conn = snowflake.connector.connect(
            account   = os.environ["SNOWFLAKE_ACCOUNT"],
            user      = os.environ["SNOWFLAKE_USER"],
            password  = os.environ["SNOWFLAKE_PASSWORD"],
            warehouse = os.environ["SNOWFLAKE_WAREHOUSE"],
            database  = os.environ["SNOWFLAKE_DATABASE"],
            schema    = os.environ["SNOWFLAKE_SCHEMA"],
            role      = os.environ["SNOWFLAKE_ROLE"],
        )
        df = pd.read_sql(
            """
            SELECT name, set_code, collector_number, price_usd, price_usd_foil, loaded_at,
                   printing_age_days, reprint_count, is_reserved_list, edhrec_rank,
                   rarity_rank, is_masters_set, is_draft_innovation,
                   legal_modern, legal_legacy, legal_vintage, legal_commander,
                   is_creature, is_instant, is_sorcery, is_enchantment,
                   is_artifact, is_planeswalker
            FROM card_price_features
            ORDER BY price_usd DESC
            """,
            conn,
        )
        conn.close()
    df["IS_RESERVED_LIST"] = df["IS_RESERVED_LIST"].astype(int)
    df["_name_key"]      = df["NAME"].str.strip().str.lower()
    df["_set_key"]       = df["SET_CODE"].str.strip().str.lower()
    df["_collector_key"] = df["COLLECTOR_NUMBER"].str.strip().str.lower()
    # Pre-compute a numeric sort key so standard-frame arts (lower numbers)
    # take priority over extended/alternate-art treatments when deduplicating.
    df["_cn_sort"] = pd.to_numeric(df["COLLECTOR_NUMBER"], errors="coerce").fillna(9999)
    return df

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    return (
        joblib.load(os.path.join(model_dir, "xgb_model.joblib")),
        joblib.load(os.path.join(model_dir, "features.joblib")),
    )

# ---------------------------------------------------------------------------
# Attach live prices from Snowflake
# ---------------------------------------------------------------------------
def attach_live_prices(inv: pd.DataFrame, sf: pd.DataFrame) -> pd.DataFrame:
    inv = inv.copy()
    # Sort by _cn_sort ascending so the standard-frame printing (lower collector
    # number) wins when multiple rows share the same (name, set_code).
    # Extended-art and alternate-art treatments carry a premium we don't want
    # to accidentally assign to a card the user owns in the regular frame.
    sf_dedup = (
        sf.sort_values(["_name_key", "_set_key", "_cn_sort"])
          .drop_duplicates(subset=["_name_key", "_set_key"])
    )
    nonfoil_lkp = sf_dedup.set_index(["_name_key", "_set_key"])["PRICE_USD"].to_dict()
    foil_lkp    = sf_dedup.set_index(["_name_key", "_set_key"])["PRICE_USD_FOIL"].to_dict()
    lat_lkp     = sf_dedup.set_index(["_name_key", "_set_key"])["LOADED_AT"].to_dict()

    def live(row):
        key = (row["_name_key"], row["_set_key"])
        # If we have an exact collector number match, prefer it
        if row.get("_collector_key") is not None:
            sf_exact = sf[
                (sf["_name_key"] == row["_name_key"]) &
                (sf["_set_key"]  == row["_set_key"])  &
                (sf["_collector_key"] == row["_collector_key"])
            ]
            if not sf_exact.empty:
                r = sf_exact.iloc[0]
                return float(r["PRICE_USD_FOIL"]) if row["is_foil"] else float(r["PRICE_USD"])
        # Fall back to standard-frame dedup
        return (foil_lkp.get(key) or nonfoil_lkp.get(key)) if row["is_foil"] else nonfoil_lkp.get(key)

    inv["Live ($)"]     = inv.apply(live, axis=1)
    inv["Price Source"] = inv.apply(
        lambda r: "LIVE" if (r["_name_key"], r["_set_key"]) in nonfoil_lkp else "STALE", axis=1
    )
    inv["Data As Of"] = inv.apply(
        lambda r: lat_lkp.get((r["_name_key"], r["_set_key"])), axis=1
    )
    return inv

# ---------------------------------------------------------------------------
# Compute P&L and sell signals
# ---------------------------------------------------------------------------
def compute_signals(inv: pd.DataFrame) -> pd.DataFrame:
    inv = inv.copy()
    inv["PnL_usd"] = inv["Live ($)"] - inv["Price Bought (USD)"]
    inv["PnL_pct"] = inv["PnL_usd"] / inv["Price Bought (USD)"] * 100
    inv["sell_flag"] = ""
    inv.loc[inv["PnL_pct"] >= GAIN_SELL_PCT, "sell_flag"] = "Strong gain"
    mask_stale = (
        (inv["Days Held"] >= HOLD_MIN_DAYS)
        & (inv["Live ($)"] < inv["Price Bought (USD)"] * LOSING_FLOOR)
    )
    inv.loc[mask_stale & (inv["sell_flag"] == ""), "sell_flag"] = "Long hold, losing"
    return inv

# ===========================================================================
# App entry point
# ===========================================================================
uploaded = st.file_uploader("LOAD INVENTORY CSV", type="csv", label_visibility="collapsed")
if uploaded is None:
    st.markdown(
        "<div style='border:1px dashed #1a3050;border-radius:4px;padding:40px;"
        "text-align:center;color:#2a5070;font-size:0.8rem;letter-spacing:0.1em'>"
        "DROP INVENTORY CSV TO INITIALISE</div>",
        unsafe_allow_html=True,
    )
    st.stop()

inv_raw = parse_inventory(uploaded.read())

with st.spinner("CONNECTING TO SNOWFLAKE..."):
    sf = load_snowflake_cards()

xgb_model, FEATURES = load_model()
inv = attach_live_prices(inv_raw, sf)
inv = compute_signals(inv)

# ---------------------------------------------------------------------------
# Data freshness badge
# ---------------------------------------------------------------------------
sf_loaded_at = sf["LOADED_AT"].max()
st.markdown(
    f"<p style='color:#1e4060;font-size:0.68rem;letter-spacing:0.06em;margin-bottom:20px'>"
    f"DATA SYNC // {sf_loaded_at} UTC</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Portfolio metrics
# ---------------------------------------------------------------------------
total_live   = inv["Live ($)"].sum()
total_bought = inv["Price Bought (USD)"].sum()
total_pnl    = total_live - total_bought
pnl_pct      = total_pnl / total_bought * 100 if total_bought else 0
n_sell       = (inv["sell_flag"] != "").sum()
n_hold       = (inv["sell_flag"] == "").sum()
n_foil       = inv["is_foil"].sum()
n_miss       = (inv["Price Source"] == "STALE").sum()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("LIVE VALUE",      f"${total_live:,.2f}")
c2.metric("COST BASIS",      f"${total_bought:,.2f}")
c3.metric("UNREALISED P&L",  f"${total_pnl:+,.2f}", delta=f"{pnl_pct:+.1f}%")
c4.metric("SELL SIGNALS",    str(n_sell))
c5.metric("HOLDING",         str(n_hold))
c6.metric("FOIL CARDS",      str(n_foil))

if n_miss:
    st.markdown(
        f"<p style='color:#2a5070;font-size:0.7rem;margin-top:8px'>"
        f"WARNING // {n_miss} card(s) not resolved in Snowflake — "
        f"promos or foil-only prints excluded by transform pipeline</p>",
        unsafe_allow_html=True,
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def make_display_row(row) -> dict:
    live    = row["Live ($)"]
    is_foil = row["is_foil"]

    if is_foil or pd.isna(live):
        predicted = "N/A"
    else:
        sf_match = sf[(sf["_name_key"] == row["_name_key"]) & (sf["_set_key"] == row["_set_key"])]
        if not sf_match.empty:
            # Pick exact collector number if available, else standard frame (lowest number)
            if row.get("_collector_key") is not None:
                exact = sf_match[sf_match["_collector_key"] == row["_collector_key"]]
                sf_row = exact if not exact.empty else sf_match.sort_values("_cn_sort").head(1)
            else:
                sf_row = sf_match.sort_values("_cn_sort").head(1)
        else:
            sf_row = sf_match
        predicted = (
            f"${np.expm1(xgb_model.predict(sf_row[FEATURES].values)[0]):.2f}"
            if not sf_row.empty else "N/A"
        )

    pnl_usd = row["PnL_usd"]
    pnl_pct = row["PnL_pct"]
    return {
        "CARD":       row["Card Name"],
        "SET":        row["Set Code"],
        "CONDITION":  row["Condition"],
        "PRINTING":   row["Printing"],
        "COST ($)":   f"${row['Price Bought (USD)']:.2f}" if pd.notna(row["Price Bought (USD)"]) else "—",
        "LIVE ($)":   f"${live:.2f}" if pd.notna(live) else "—",
        "MODEL ($)":  predicted,
        "P&L ($)":    f"${pnl_usd:+.2f}" if pd.notna(pnl_usd) else "—",
        "P&L (%)":    f"{pnl_pct:+.1f}%" if pd.notna(pnl_pct) else "—",
        "DAYS":       int(row["Days Held"]) if pd.notna(row["Days Held"]) else "—",
        "SOURCE":     row["Price Source"],
    }

def colour_pnl(col):
    out = []
    for v in col:
        try:
            n = float(str(v).replace("$", "").replace("%", "").replace("+", ""))
            out.append("color: #00e676" if n > 0 else "color: #ff1744" if n < 0 else "")
        except Exception:
            out.append("color: #2a5070")
    return out

def render_table(subset: pd.DataFrame):
    rows = [make_display_row(r) for _, r in subset.iterrows()]
    g = pd.DataFrame(rows)
    st.dataframe(
        g.style.apply(colour_pnl, subset=["P&L ($)", "P&L (%)"]),
        use_container_width=True, hide_index=True,
    )

# ===========================================================================
# Tabs
# ===========================================================================
tab_sell, tab_hold, tab_buy = st.tabs([
    "SELL SIGNALS",
    "HOLD",
    "BUY OPPORTUNITIES",
])

# ===========================================================================
# SELL TAB
# ===========================================================================
with tab_sell:
    sell_df = inv[inv["sell_flag"] != ""]
    if sell_df.empty:
        st.markdown(
            "<p style='color:#1a5030;font-size:0.8rem;letter-spacing:0.08em'>"
            "NO ACTIVE SELL SIGNALS</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"**{len(sell_df)} CARD(S) FLAGGED**")
        for flag_label, group in sell_df.groupby("sell_flag", sort=False):
            if flag_label == "Strong gain":
                st.markdown("## STRONG GAIN — CONSIDER EXIT")
                st.caption(f"Unrealised P&L >= {GAIN_SELL_PCT}%. Solid return — locking in is worth considering.")
            else:
                st.markdown("## LONG HOLD / DECLINING VALUE")
                st.caption(
                    f"Held >{HOLD_MIN_DAYS} days and down >{(1 - LOSING_FLOOR)*100:.0f}% from cost basis. "
                    "Consider cutting the position."
                )
            render_table(group)

# ===========================================================================
# HOLD TAB
# ===========================================================================
with tab_hold:
    hold_df = inv[inv["sell_flag"] == ""].sort_values("Live ($)", ascending=False)
    st.markdown(f"**{len(hold_df)} CARD(S) — HOLD**")
    render_table(hold_df)

# ===========================================================================
# BUY TAB
# ===========================================================================
with tab_buy:
    st.markdown(
        "<p style='color:#2a5070;font-size:0.72rem;letter-spacing:0.06em'>"
        "CARDS NOT IN INVENTORY // RANKED BY SIGNAL STRENGTH</p>",
        unsafe_allow_html=True,
    )

    # Pre-compute predictions
    sf_pred = sf[sf["PRICE_USD"] > 0].copy()
    sf_pred["predicted_usd"] = np.expm1(xgb_model.predict(sf_pred[FEATURES].values))
    sf_pred["upside_ratio"]  = sf_pred["predicted_usd"] / sf_pred["PRICE_USD"]

    # Exclude owned cards
    owned     = set(inv["_name_key"])
    not_owned = sf_pred[~sf_pred["_name_key"].isin(owned)].copy()

    # Deduplicate: one row per card name, keep cheapest printing
    not_owned = (
        not_owned.sort_values("PRICE_USD")
                 .drop_duplicates(subset=["_name_key"])
    )

    # Shared filter: predicted value >= $5
    min_pred_filter = st.checkbox("Only show predicted value >= $5", value=False)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---- Signal 1: ML undervalued ----
    st.markdown("## MODEL — UNDERVALUED")
    st.caption(
        f"XGBoost predicts >= {(UNDERVAL_RATIO - 1)*100:.0f}% above live market price. "
        "Nonfoil only — model not trained on foil premiums. Cheapest printing shown. Top 20."
    )

    model_picks = not_owned[not_owned["upside_ratio"] >= UNDERVAL_RATIO].sort_values(
        "upside_ratio", ascending=False
    )
    if min_pred_filter:
        model_picks = model_picks[model_picks["predicted_usd"] >= 5]
    model_picks = model_picks.head(20)

    if model_picks.empty:
        st.caption("NO SIGNALS MATCH CURRENT FILTERS")
    else:
        d = model_picks[["NAME", "SET_CODE", "PRICE_USD", "predicted_usd",
                          "upside_ratio", "IS_RESERVED_LIST", "EDHREC_RANK"]].copy()
        d.columns = ["CARD", "SET", "LIVE ($)", "MODEL ($)", "UPSIDE", "RESERVED", "EDHREC RANK"]
        d["LIVE ($)"]  = d["LIVE ($)"].round(2)
        d["MODEL ($)"] = d["MODEL ($)"].round(2)
        d["UPSIDE"]    = d["UPSIDE"].map("{:.2f}x".format)
        d["RESERVED"]  = d["RESERVED"].map({1: "YES", 0: ""})
        st.dataframe(d, use_container_width=True, hide_index=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---- Signal 2: Reserved List ----
    st.markdown("## RESERVED LIST — BARGAINS")
    st.caption(
        "Cannot be reprinted. Fixed supply + growing demand = hard price floor. "
        "Cheapest unique card shown."
    )

    rl_picks = not_owned[not_owned["IS_RESERVED_LIST"] == 1].sort_values("PRICE_USD")
    if min_pred_filter:
        rl_picks = rl_picks[rl_picks["predicted_usd"] >= 5]
    rl_picks = rl_picks.head(20)

    if rl_picks.empty:
        st.caption("NO SIGNALS MATCH CURRENT FILTERS")
    else:
        d = rl_picks[["NAME", "SET_CODE", "PRICE_USD", "predicted_usd", "EDHREC_RANK"]].copy()
        d.columns = ["CARD", "SET", "LIVE ($)", "MODEL ($)", "EDHREC RANK"]
        d["LIVE ($)"]  = d["LIVE ($)"].round(2)
        d["MODEL ($)"] = d["MODEL ($)"].round(2)
        st.dataframe(d, use_container_width=True, hide_index=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---- Signal 3: High EDH demand, low price ----
    st.markdown("## HIGH COMMANDER DEMAND — LOW PRICE")
    st.caption(
        f"EDHREC rank < {EDHREC_POPULAR:,} and live price <= ${EDHREC_CHEAP}. "
        "High playability relative to current price. Cheapest unique card shown."
    )

    edh_picks = not_owned[
        (not_owned["EDHREC_RANK"] < EDHREC_POPULAR) &
        (not_owned["PRICE_USD"] <= EDHREC_CHEAP)
    ].sort_values("EDHREC_RANK")
    if min_pred_filter:
        edh_picks = edh_picks[edh_picks["predicted_usd"] >= 5]
    edh_picks = edh_picks.head(20)

    if edh_picks.empty:
        st.caption("NO SIGNALS MATCH CURRENT FILTERS")
    else:
        d = edh_picks[["NAME", "SET_CODE", "PRICE_USD", "predicted_usd",
                        "EDHREC_RANK", "IS_RESERVED_LIST"]].copy()
        d.columns = ["CARD", "SET", "LIVE ($)", "MODEL ($)", "EDHREC RANK", "RESERVED"]
        d["LIVE ($)"]  = d["LIVE ($)"].round(2)
        d["MODEL ($)"] = d["MODEL ($)"].round(2)
        d["RESERVED"]  = d["RESERVED"].map({1: "YES", 0: ""})
        st.dataframe(d, use_container_width=True, hide_index=True)
