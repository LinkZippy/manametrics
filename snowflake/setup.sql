-- Warehouse:
CREATE WAREHOUSE IF NOT EXISTS mtg_wh
WITH WAREHOUSE_SIZE = "SMALL"
AUTO_SUSPEND = 60
AUTO_RESUME = TRUE
;

--Database:
CREATE DATABASE IF NOT EXISTS mtg_db
;

--Schema:
CREATE SCHEMA IF NOT EXISTS mtg_db.curated
;

--Table:
CREATE TABLE IF NOT EXISTS mtg_db.curated.card_price_features (
    oracle_id STRING,
    name STRING,
    set_code STRING,
    released_at DATE,
    original_release_date DATE,
    printing_age_days INTEGER,
    is_reserved_list BOOLEAN,
    edhrec_rank INTEGER,
    price_usd FLOAT,
    log_price_usd FLOAT,
    price_usd_foil FLOAT,      -- foil price, stored for dashboard display only (not used in model)
    loaded_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
)
;
