# ManaMetrics

An end-to-end data engineering portfolio project that ingests Magic: The Gathering card data daily, engineers price-relevant features, trains an XGBoost model, and surfaces buy/sell signals through a live Streamlit dashboard.

**Stack:** Apache Airflow, PySpark, AWS S3, Snowflake, XGBoost/LightGBM, Streamlit

---

## Architecture

```
Scryfall API
    │
    ▼
Airflow DAG: scryfall_ingest_dag   (@daily)
    │  Two-step: query bulk-data endpoint → stream download to S3
    ▼
S3 (raw layer)
    raw/scryfall/dt=YYYY-MM-DD/all_cards.jsonl.gz
    │
    ▼  (triggered automatically on ingest success)
Airflow DAG: mtg_transform_dag
    │  Download from S3 → PySpark feature engineering → Snowflake + S3 (curated)
    ▼
Snowflake: mtg_db.curated.card_price_features
    │
    ├──▶ models/train.py          XGBoost training on curated features
    │
    └──▶ dashboard/inventory.py   Live Streamlit dashboard
```

---

## Local Environment

| Component | Detail |
|---|---|
| Python venv | `~/airflow-learning/venv` |
| Airflow home | `~/airflow-learning/airflow_home` (set via `AIRFLOW_HOME`) |
| Airflow version | 3.3.0, run with `airflow standalone` |
| Java | 17 (required for PySpark) |
| AWS region | `ap-southeast-2` (Sydney) |
| S3 bucket | `mtg-price-predictor-phoom` |

**Always prefix Airflow commands with the home variable:**
```bash
AIRFLOW_HOME=~/airflow-learning/airflow_home airflow <command>
```

**Start Airflow:**
```bash
AIRFLOW_HOME=~/airflow-learning/airflow_home airflow standalone
```
The admin password prints to stdout on startup. The UI runs at `localhost:8080`.

---

## Environment Variables

Create a `.env` file at the project root (gitignored):

```
SNOWFLAKE_ACCOUNT=<account_identifier>
SNOWFLAKE_USER=<username>
SNOWFLAKE_PASSWORD=<password>
SNOWFLAKE_WAREHOUSE=mtg_wh
SNOWFLAKE_DATABASE=mtg_db
SNOWFLAKE_SCHEMA=curated
SNOWFLAKE_ROLE=ACCOUNTADMIN
```

The Snowflake account identifier comes from your Snowsight URL:
`https://<account_identifier>.snowflakecomputing.com`

---

## Snowflake Setup

Run `snowflake/setup.sql` in Snowsight once to create the warehouse, database, schema, and table. This file is the permanent schema source of truth — Snowsight worksheets are scratch space only.

**Important:** `write_pandas` with `overwrite=True` truncates the existing table but does not recreate it. When adding new columns, run the `ALTER TABLE` block at the bottom of `setup.sql` before re-running the pipeline.

---

## Pipeline

### DAG 1 — `scryfall_ingest_dag` (daily)

**File:** `dags/scryfall_ingest_dag.py`

**Tasks:**

**`get_bulk_data`** — Queries `https://api.scryfall.com/bulk-data` to get the current download URL. Scryfall separates the metadata endpoint from the actual file so the download URL changes with each daily snapshot. The relevant field is `jsonl_download_uri` on the `all_cards` entry.
Both Scryfall requests check the HTTP status before using the response, so API or download failures fail the task instead of being mistaken for valid data.

> **Why a User-Agent header?** Scryfall rejects requests from the default `python-requests` User-Agent with a 400 error. A descriptive custom header is required.

**`download_and_upload`** — Streams the bulk file directly to S3 without loading it into memory. Key detail: `response.raw.decode_content = True` must be set before passing the stream to the S3 hook, otherwise urllib3 bypasses gzip decompression and the uploaded bytes are corrupt. The file lands at `raw/scryfall/dt=YYYY-MM-DD/all_cards.jsonl.gz`. The `logical_date` context variable (not `datetime.now()`) is used for partitioning so reruns are idempotent.

> **Format note:** Scryfall originally served a single JSON array; they later switched to gzip-compressed JSONL (one card object per line). The S3 key uses `.jsonl.gz` so PySpark auto-detects compression from the extension.

**`trigger_transform`** — Fires `mtg_transform_dag` using `TriggerDagRunOperator` (import path in Airflow 3.x: `airflow.providers.standard.operators.trigger_dagrun`).

---

### DAG 2 — `mtg_transform_dag` (triggered)

**File:** `dags/mtg_transform_dag.py`

**Tasks:**

**`download_from_s3`** — Downloads the JSONL file for the run's logical date from S3 to `/tmp/`. Returns the local path for the next task.

**`run_transform`** — Bash task that activates the venv and runs `spark/transform.py <local_path>`. The script path is resolved relative to the DAG file, so the task works from this repository's actual checkout path. Using a bash task (rather than a Python task) keeps PySpark's JVM lifecycle separate from Airflow's worker process.

**`upload_parquet_to_s3`** — Syncs the curated Parquet output from `/tmp/card_price_features/` to `s3://mtg-price-predictor-phoom/curated/card_price_features/`.

---

### Transform — `spark/transform.py`

Reads the JSONL file with PySpark in local mode. The Scryfall bulk file contains one row per printing of every card ever released — roughly 530k+ rows before filtering. The pipeline filters and engineers features down to ~79k rows.

**Filtering (applied in order):**

| Step | Filter | Reason |
|---|---|---|
| A | `promo == False AND nonfoil == True` | Promo and foil-only cards are priced by collector scarcity and print run exclusivity, not by playability signals. Including them adds noise to the nonfoil price model. |
| A2 | `type_line NOT CONTAINS "Basic Land"` | Basic lands have unlimited print runs. Price variation is purely cosmetic (art, artist, set symbol). |
| B | `prices.usd IS NOT NULL` | Rows with no nonfoil price have no training target. |

**Feature engineering:**

All oracle-level lookups (original release date, reprint count) are computed from the **full unfiltered** DataFrame before joins. This prevents bias — e.g. a card whose first printing was a promo (filtered out) would otherwise show an artificially young age.

| Feature | Source | Logic |
|---|---|---|
| `price_usd` | `prices.usd` cast to double | Model target (pre-log) |
| `log_price_usd` | `log1p(price_usd)` | Target used in training. Card prices are heavily right-skewed — a few cards cost thousands while most cost under $1. Log transform corrects this so the model isn't dominated by outliers. |
| `price_usd_foil` | `prices.usd_foil` | Stored for dashboard display only; not used in model. |
| `printing_age_days` | `datediff(today, min(released_at) per oracle_id)` | Age since the card's **first ever printing**, not this specific printing. Older cards are harder to reprint and generally hold more value. |
| `reprint_count` | `countDistinct(set) per oracle_id` | Number of distinct sets containing this card. More reprints = more supply = lower price. A card printed once in 1993 (e.g. Black Lotus) vs. reprinted 30+ times (e.g. Sol Ring) have fundamentally different supply curves. |
| `is_reserved_list` | `reserved` boolean | WotC has contractually committed to never reprint Reserved List cards. This creates a hard price floor — fixed supply with growing demand. |
| `edhrec_rank` | `edhrec_rank` | Commander format playability rank. Lower = more played. Nulls are left as nulls (XGBoost handles missing values natively via tree splits). |
| `rarity_rank` | `rarity` mapped to 0–3 | common=0, uncommon=1, rare=2, mythic=3. Rarity directly controls print run size and therefore supply. Ordinal encoding lets the tree model use a single numeric split rather than four one-hot columns. |
| `is_masters_set` | `set_type == "masters"` | Masters sets (Modern Masters, Ultimate Masters, etc.) are limited-print reprint products. High demand + limited supply means cards retain value. |
| `is_draft_innovation` | `set_type == "draft_innovation"` | "Horizons" style sets (Modern Horizons 1/2/3) inject powerful cards directly into eternal formats, creating outsized competitive demand. |
| `legal_modern` | `legalities.modern IN ("legal", "restricted")` | Each legal format represents an independent pool of players competing for the card. A card legal in Modern, Legacy, and Commander has three demand pools bidding on it simultaneously. |
| `legal_legacy` | `legalities.legacy IN ("legal", "restricted")` | Legacy players tend to spend more per card than Modern players. |
| `legal_vintage` | `legalities.vintage IN ("legal", "restricted")` | Smallest but highest-spending player base. Restricted cards are still in demand. |
| `legal_commander` | `legalities.commander IN ("legal", "restricted")` | Largest casual format by volume. Strong Commander demand alone can sustain a card's price even if it's banned in competitive formats. |
| `is_creature` / `is_instant` / `is_sorcery` / `is_enchantment` / `is_artifact` / `is_planeswalker` | `type_line.contains(type)` | Different card types have different demand curves across formats. Binary flags rather than a single category because cards can have multiple types (e.g. "Artifact Creature"). |

**Output:** Written as Parquet to `/tmp/card_price_features/` and loaded into Snowflake via `write_pandas(overwrite=True)`, which truncates the table and bulk-loads fresh data.

---

## Model — `models/train.py`

### Features (17 total)

`PRINTING_AGE_DAYS`, `REPRINT_COUNT`, `IS_RESERVED_LIST`, `EDHREC_RANK`, `RARITY_RANK`, `IS_MASTERS_SET`, `IS_DRAFT_INNOVATION`, `LEGAL_MODERN`, `LEGAL_LEGACY`, `LEGAL_VINTAGE`, `LEGAL_COMMANDER`, `IS_CREATURE`, `IS_INSTANT`, `IS_SORCERY`, `IS_ENCHANTMENT`, `IS_ARTIFACT`, `IS_PLANESWALKER`

### Cross-validation

`TimeSeriesSplit(n_splits=5)` with rows sorted by `original_release_date`.

**Why not random KFold?** Card price data is temporal. With random splitting, a model could train on a card's 2024 price and be asked to "predict" its 2021 price — that's backwards in time and constitutes data leakage. TimeSeriesSplit enforces that training always uses earlier release dates than the validation set, which mirrors real-world deployment.

### Results (as of August 2026)

```
Fold 1  |  train= 13220  val= 13218  XGB RMSE=1.7604  LGB RMSE=1.7837
Fold 2  |  train= 26438  val= 13218  XGB RMSE=0.6919  LGB RMSE=0.6556
Fold 3  |  train= 39656  val= 13218  XGB RMSE=0.4791  LGB RMSE=0.4885
Fold 4  |  train= 52874  val= 13218  XGB RMSE=0.4995  LGB RMSE=0.5050
Fold 5  |  train= 66092  val= 13218  XGB RMSE=0.7419  LGB RMSE=0.7252

XGBoost  Mean RMSE: 0.8345 ± 0.4743
LightGBM Mean RMSE: 0.8316 ± 0.4844
```

Fold 1 RMSE is high because it trains only on Alpha/Beta era cards and validates on a very different market period. Folds 3–5 reflect the model's real-world performance on the modern card pool.

### Feature Importance (XGBoost, final model)

```
IS_RESERVED_LIST     38.2%   ← dominant signal; hard reprint ban = price floor
RARITY_RANK          16.1%   ← print run size proxy; biggest new feature
EDHREC_RANK           7.6%   ← playability/demand signal
PRINTING_AGE_DAYS     7.4%   ← supply scarcity over time
LEGAL_MODERN          4.9%   ← competitive demand pool
REPRINT_COUNT         4.0%   ← cumulative supply signal
LEGAL_VINTAGE         3.4%
LEGAL_LEGACY          3.1%
LEGAL_COMMANDER       2.8%
IS_CREATURE           2.5%
IS_MASTERS_SET        2.3%
...
```

The final XGBoost model is retrained on the full dataset and saved to `models/xgb_model.joblib`. The feature list is saved separately to `models/features.joblib` so the dashboard always loads exactly the columns the model was trained on.

### Retrain

```bash
cd ~/mtg-price-predictor
source ~/airflow-learning/venv/bin/activate
python models/train.py
```

---

## Dashboard — `dashboard/inventory.py`

**Run:**
```bash
streamlit run ~/mtg-price-predictor/dashboard/inventory.py
```

Upload your inventory CSV (columns: `Quantity`, `Card Name`, `Set Code`, `Printing`, `Date Bought`, `Price Bought (USD)`). Dates in the CSV use Buddhist Era — the dashboard converts automatically (CE = BE − 543).

**Live prices** are pulled from Snowflake's `card_price_features` table (updated daily by the pipeline), not from the static CSV. Foil cards use `PRICE_USD_FOIL`; nonfoil cards use `PRICE_USD`.

**Tabs:**

- **SELL SIGNALS** — Cards with unrealised P&L ≥ 35% ("Strong gain") or held >180 days and down >15% from cost ("Long hold, losing")
- **HOLD** — Everything else in the inventory
- **BUY OPPORTUNITIES** — Three signals on cards you don't own:
  - *Model undervalued:* XGBoost predicts ≥ 20% above live market price
  - *Reserved List bargains:* WotC can't reprint; fixed supply with growing demand
  - *High Commander demand, low price:* EDHREC rank < 2,000 and price ≤ $15

All buy signals deduplicate by card name and keep the cheapest printing. Model predictions show "N/A" for foil cards since the model was trained on nonfoil prices only.

---

## Running the Full Pipeline

```bash
# 1. Start Airflow
AIRFLOW_HOME=~/airflow-learning/airflow_home airflow standalone

# 2. Trigger ingest (in a second terminal)
source ~/airflow-learning/venv/bin/activate
AIRFLOW_HOME=~/airflow-learning/airflow_home airflow dags trigger scryfall_ingest_dag --logical-date $(date +%Y-%m-%d)

# 3. Monitor in UI at localhost:8080
#    scryfall_ingest_dag → mtg_transform_dag fires automatically on success

# 4. Retrain model after transform completes
cd ~/mtg-price-predictor
python models/train.py

# 5. Run dashboard
streamlit run dashboard/inventory.py
```

---

## Repo Structure

```
mtg-price-predictor/
├── dags/
│   ├── scryfall_ingest_dag.py   # Daily ingest: Scryfall API → S3
│   └── mtg_transform_dag.py     # Transform: S3 → PySpark → Snowflake
├── spark/
│   └── transform.py             # PySpark feature engineering script
├── models/
│   ├── train.py                 # XGBoost/LightGBM training
│   ├── xgb_model.joblib         # Saved model (gitignored)
│   └── features.joblib          # Saved feature list (gitignored)
├── snowflake/
│   └── setup.sql                # Schema source of truth
├── dashboard/
│   ├── inventory.py             # Inventory tracker (this project)
│   └── app.py                   # Model exploration dashboard
├── .env                         # Snowflake credentials (gitignored)
└── README.md
```

---

## Key Bugs Fixed During Development

**Scryfall API changes**
- `download_uri` field renamed to `jsonl_download_uri`
- Format changed from a single JSON array to gzip-compressed JSONL (`.jsonl.gz`)
- Spark's `option("multiline", "true")` was removed — JSONL is one object per line by default, and multiline mode would try to parse the entire file as one JSON document

**Gzip streaming**
- `response.raw.decode_content = True` must be set before the stream is passed to `S3Hook.load_file_obj()`, otherwise urllib3 skips gzip decompression and the uploaded file contains raw compressed bytes
- The S3 key uses the `.jsonl.gz` extension so Spark auto-detects compression without extra configuration

**Airflow 3.x import path**
- `TriggerDagRunOperator` moved to `airflow.providers.standard.operators.trigger_dagrun`

**Task wiring bug**
- Task dependency calls (`upload = download_and_upload(...)`, `upload >> trigger_transform`) must be in the DAG body, not inside a task function body. Airflow parses DAG files at import time; dependencies inside a function body are never seen by the scheduler.

**Snowflake schema migration**
- `write_pandas(overwrite=True)` truncates the existing table but does not recreate it. Adding new columns requires an explicit `ALTER TABLE` before re-running the pipeline — otherwise the COPY INTO statement references columns that don't exist in the table yet.

**Variable reassignment in PySpark**
- Inserting a new filter step after other DataFrames have already branched off an earlier variable doesn't retroactively apply the filter to those branches. All derived DataFrames must be created after the filter step, not before it.

**Temporal data leakage**
- Random KFold cross-validation on price data leaks future prices into training. `TimeSeriesSplit` sorted by `original_release_date` is used instead to ensure the model only ever trains on information available at the time of prediction.
