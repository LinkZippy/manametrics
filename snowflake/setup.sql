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

-- Table (source of truth for schema — run this in Snowsight on fresh setup):
CREATE TABLE IF NOT EXISTS mtg_db.curated.card_price_features (
    -- Identity
    oracle_id             STRING,
    name                  STRING,
    set_code              STRING,        -- renamed from `set` (SQL reserved word)
    collector_number      STRING,        -- e.g. "316" or "406★"; needed for exact printing match
    released_at           DATE,
    original_release_date DATE,

    -- Supply / age signals
    printing_age_days     INTEGER,       -- days since card's FIRST printing ever
    reprint_count         INTEGER,       -- # distinct sets containing this oracle_id

    -- Playability signals
    is_reserved_list      BOOLEAN,
    edhrec_rank           INTEGER,       -- NULL = not ranked; lower = more played

    -- Rarity / print run
    rarity_rank           INTEGER,       -- 0=common 1=uncommon 2=rare 3=mythic

    -- Set type
    is_masters_set        INTEGER,       -- 1 if Masters-series set (limited reprint run)
    is_draft_innovation   INTEGER,       -- 1 if Horizons-style direct-to-format set

    -- Format legality (1 = legal or restricted, 0 = not_legal or banned)
    legal_modern          INTEGER,
    legal_legacy          INTEGER,
    legal_vintage         INTEGER,
    legal_commander       INTEGER,

    -- Card type flags (a card can be multiple types, e.g. Artifact Creature)
    is_creature           INTEGER,
    is_instant            INTEGER,
    is_sorcery            INTEGER,
    is_enchantment        INTEGER,
    is_artifact           INTEGER,
    is_planeswalker       INTEGER,

    -- Price targets
    price_usd             FLOAT,         -- nonfoil market price (model target)
    log_price_usd         FLOAT,         -- log1p(price_usd) — actual training label
    price_usd_foil        FLOAT,         -- foil price, dashboard display only

    loaded_at             TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
)
;

-- ============================================================
-- ALTER TABLE statements to run when adding new columns to an
-- EXISTING table (write_pandas overwrite=True truncates but
-- does NOT recreate the schema — you must add columns first).
-- Run these in Snowsight if upgrading from the old schema.
-- ============================================================
--
-- ALTER TABLE mtg_db.curated.card_price_features
--     ADD collector_number STRING,
--         reprint_count INTEGER,
--         rarity_rank INTEGER,
--         is_masters_set INTEGER,
--         is_draft_innovation INTEGER,
--         legal_modern INTEGER,
--         legal_legacy INTEGER,
--         legal_vintage INTEGER,
--         legal_commander INTEGER,
--         is_creature INTEGER,
--         is_instant INTEGER,
--         is_sorcery INTEGER,
--         is_enchantment INTEGER,
--         is_artifact INTEGER,
--         is_planeswalker INTEGER;
