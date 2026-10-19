from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import sys

if len(sys.argv) < 2:
    raise ValueError("Usage: transform.py <input_json_path>")

input_path = sys.argv[1]

spark = SparkSession.builder.appName("mtg_transform").master("local[*]").getOrCreate()

# Scryfall now serves JSONL (one card object per line) — no multiline option needed.
# multiline=True was for reading a single JSON array spanning many lines.
# JSONL is Spark's default JSON format: each line is an independent JSON record.
df = spark.read.json(input_path)

# ---------------------------------------------------------------------------
# STEP A: Filter out promos and foil-only printings.
#
# Promos and foils are driven by collector scarcity / print run size, not
# by the playability + mechanical signals we are modelling on.
# "nonfoil == True" means this printing has a real non-foil version.
# ---------------------------------------------------------------------------
df_filtered = df.filter((F.col("promo") == False) & (F.col("nonfoil") == True))
print(f"After dropping promos and foil-only printings: {df_filtered.count()} rows")

# ---------------------------------------------------------------------------
# STEP A2: Drop basic lands.
#
# Unlimited print runs make price purely cosmetic / art-driven noise.
# ---------------------------------------------------------------------------
df_filtered = df_filtered.filter(~F.col("type_line").contains("Basic Land"))
print(f"After dropping basic lands: {df_filtered.count()} rows")


# ---------------------------------------------------------------------------
# STEP B: Extract and clean the price target.
#
# prices.usd  = nonfoil market price (our model target)
# prices.usd_foil = foil price, stored alongside for the dashboard only.
# We drop rows with no nonfoil price — no target to train on.
# ---------------------------------------------------------------------------
df_priced = (
    df_filtered
    .withColumn("price_usd",      F.col("prices.usd").cast("double"))
    .withColumn("price_usd_foil", F.col("prices.usd_foil").cast("double"))
    .filter(F.col("price_usd").isNotNull())
)
print(f"After requiring a non-null usd price: {df_priced.count()} rows")


# ---------------------------------------------------------------------------
# STEP C: Log-transform the target.
#
# Card prices are heavily right-skewed. log1p(x) = ln(x+1) corrects that
# so the model isn't dominated by a handful of expensive outliers.
# ---------------------------------------------------------------------------
df_target = df_priced.withColumn("log_price_usd", F.log1p(F.col("price_usd")))


# ---------------------------------------------------------------------------
# STEP D: Printing age from the card's ORIGINAL release date.
#
# We compute this from the FULL unfiltered df so that a card whose first
# printing was a promo (excluded by Step A) still gets the correct age.
# oracle_id is stable across all printings of the same card.
# ---------------------------------------------------------------------------
original_release_lookup = (
    df.withColumn("released_date", F.to_date(F.col("released_at")))
      .groupBy("oracle_id")
      .agg(F.min("released_date").alias("original_release_date"))
)

df_age = df_target.join(original_release_lookup, on="oracle_id", how="left")
df_age = df_age.withColumn(
    "printing_age_days", F.datediff(F.current_date(), F.col("original_release_date"))
)

null_dates = df_age.filter(F.col("original_release_date").isNull()).count()
print(f"Rows with no resolvable original_release_date: {null_dates}")


# ---------------------------------------------------------------------------
# STEP D2: Reprint count.
#
# How many distinct sets contain a printing of this oracle_id?
# More reprints = more supply = generally lower price.
# We compute from the FULL unfiltered df so we count every printing ever
# released, not just the ones that passed our promo/foil filter.
#
# This is a key signal the original model was missing: a card printed once
# in 1994 (e.g. Black Lotus) vs a card reprinted 30+ times (e.g. Sol Ring)
# have fundamentally different supply curves.
# ---------------------------------------------------------------------------
reprint_lookup = (
    df.groupBy("oracle_id")
      .agg(F.countDistinct("set").alias("reprint_count"))
)

df_reprints = df_age.join(reprint_lookup, on="oracle_id", how="left")
print(f"Reprint count attached. Sample distinct counts: {df_reprints.select('reprint_count').distinct().count()} unique values")


# ---------------------------------------------------------------------------
# STEP E: Reserved List flag.
# ---------------------------------------------------------------------------
df_reserved = df_reprints.withColumn("is_reserved_list", F.col("reserved"))

reserved_count = df_reserved.filter(F.col("is_reserved_list") == True).count()
print(f"Rows on the Reserved List: {reserved_count}")


# ---------------------------------------------------------------------------
# STEP F: EDHREC rank as a playability/popularity signal.
#
# Lower rank = more commonly played in Commander decks.
# Null means "not ranked" — we leave nulls as nulls so XGBoost/LightGBM
# can handle missing values natively at training time.
# ---------------------------------------------------------------------------
df_edhrec = df_reserved.withColumn("edhrec_rank", F.col("edhrec_rank"))

null_rank_count = df_edhrec.filter(F.col("edhrec_rank").isNull()).count()
print(f"Rows with no EDHREC rank: {null_rank_count}")


# ---------------------------------------------------------------------------
# STEP G: Rarity rank (ordinal encoding).
#
# Rarity directly controls print run size and therefore supply.
# common=0, uncommon=1, rare=2, mythic=3.
# "special" and "bonus" cards (e.g. timeshifted, masterpiece) are treated
# as rare since their scarcity is closest to that tier.
#
# Why ordinal instead of one-hot? Tree models like XGBoost split on
# numeric thresholds, so ordinal encoding lets the model discover that
# mythic > rare > uncommon > common without needing four separate columns.
# ---------------------------------------------------------------------------
df_rarity = df_edhrec.withColumn(
    "rarity_rank",
    F.when(F.col("rarity") == "common",   0)
     .when(F.col("rarity") == "uncommon", 1)
     .when(F.col("rarity") == "rare",     2)
     .when(F.col("rarity") == "mythic",   3)
     .otherwise(2)   # special / bonus / token → treat as rare
)


# ---------------------------------------------------------------------------
# STEP H: Set type flags.
#
# The type of set a card is printed in is a strong proxy for print run
# size and therefore supply:
#
#   is_masters_set:      Masters sets (e.g. Modern Masters, Ultimate Masters)
#                        are limited print runs of reprinted staples.
#                        High demand + low supply → cards hold value well here.
#
#   is_draft_innovation: "Horizons" style sets (Modern Horizons 1/2/3) that
#                        inject powerful cards directly into non-rotating formats.
#                        Cards here often have outsized competitive demand.
#
# All other set types (expansion, core, commander, etc.) are captured by
# the absence of these flags — the model learns the baseline from there.
# ---------------------------------------------------------------------------
df_settype = (
    df_rarity
    .withColumn(
        "is_masters_set",
        F.when(F.col("set_type") == "masters", 1).otherwise(0)
    )
    .withColumn(
        "is_draft_innovation",
        F.when(F.col("set_type") == "draft_innovation", 1).otherwise(0)
    )
)


# ---------------------------------------------------------------------------
# STEP I: Format legality flags.
#
# Format legality determines the player pool that demands the card:
#   legal_modern    → Modern players (~the largest competitive format)
#   legal_legacy    → Legacy players (smaller but willing to spend more)
#   legal_vintage   → Vintage players (smallest pool, highest spending)
#   legal_commander → Commander players (largest casual format by volume)
#
# A card legal in multiple formats has multiple demand pools bidding for it.
# "restricted" counts as legal here — restricted cards are still in demand.
#
# Scryfall's legalities field is a struct: legalities.modern = "legal" |
# "not_legal" | "banned" | "restricted".
# ---------------------------------------------------------------------------
df_legality = df_settype
for fmt in ["modern", "legacy", "vintage", "commander"]:
    df_legality = df_legality.withColumn(
        f"legal_{fmt}",
        F.when(F.col(f"legalities.{fmt}").isin("legal", "restricted"), 1).otherwise(0)
    )


# ---------------------------------------------------------------------------
# STEP J: Card type flags.
#
# Different card types have different demand curves:
#   - Instants/sorceries: high demand in spell-heavy formats
#   - Creatures: high volume, generally lower individual price
#   - Artifacts: cross-format appeal (any deck can run them)
#   - Enchantments: popular in Commander, price driven by Commander demand
#   - Planeswalkers: historically expensive, limited print slots per set
#
# We use binary flags (0/1) via type_line.contains() because a card can
# belong to multiple types (e.g. "Artifact Creature").
# ---------------------------------------------------------------------------
df_cardtype = df_legality
for card_type in ["Creature", "Instant", "Sorcery", "Enchantment", "Artifact", "Planeswalker"]:
    df_cardtype = df_cardtype.withColumn(
        f"is_{card_type.lower()}",
        F.when(F.col("type_line").contains(card_type), 1).otherwise(0)
    )


# ---------------------------------------------------------------------------
# STEP K: Final select + write to Snowflake.
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
import os
from snowflake.connector.pandas_tools import write_pandas
import snowflake.connector

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

df_final = df_cardtype.select(
    F.col("oracle_id"),
    F.col("name"),
    F.col("set").alias("set_code"),
    F.col("collector_number"),
    F.col("released_at"),
    F.col("original_release_date").cast("string"),
    F.col("printing_age_days"),
    F.col("reprint_count"),
    F.col("is_reserved_list"),
    F.col("edhrec_rank"),
    F.col("rarity_rank"),
    F.col("is_masters_set"),
    F.col("is_draft_innovation"),
    F.col("legal_modern"),
    F.col("legal_legacy"),
    F.col("legal_vintage"),
    F.col("legal_commander"),
    F.col("is_creature"),
    F.col("is_instant"),
    F.col("is_sorcery"),
    F.col("is_enchantment"),
    F.col("is_artifact"),
    F.col("is_planeswalker"),
    F.col("price_usd"),
    F.col("log_price_usd"),
    F.col("price_usd_foil"),
)

# Preview
df_final.select(
    "name", "set_code", "rarity_rank", "reprint_count",
    "legal_modern", "legal_commander", "is_artifact",
    "price_usd"
).show(10, truncate=False)

# Write curated Parquet for S3/Databricks
df_final.write.mode("overwrite").parquet("/tmp/card_price_features")
print("Parquet written to /tmp/card_price_features")

pdf = df_final.toPandas()
pdf.columns = [c.upper() for c in pdf.columns]

conn = snowflake.connector.connect(
    account=os.environ["SNOWFLAKE_ACCOUNT"],
    user=os.environ["SNOWFLAKE_USER"],
    password=os.environ["SNOWFLAKE_PASSWORD"],
    warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
    database=os.environ["SNOWFLAKE_DATABASE"],
    schema=os.environ["SNOWFLAKE_SCHEMA"],
    role=os.environ["SNOWFLAKE_ROLE"],
)

success, nchunks, nrows, _ = write_pandas(
    conn=conn,
    df=pdf,
    table_name="CARD_PRICE_FEATURES",
    database=os.environ["SNOWFLAKE_DATABASE"].upper(),
    schema=os.environ["SNOWFLAKE_SCHEMA"].upper(),
    overwrite=True,
)

print(f"Snowflake write: success={success}, chunks={nchunks}, rows={nrows}")
conn.close()
spark.stop()
