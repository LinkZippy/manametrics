from datetime import datetime
from pathlib import Path
import shlex

from airflow.sdk import task, dag


@dag(
    dag_id="mtg_transform_dag",
    start_date=datetime(2026, 1, 1),
    schedule=None,  # triggered by scryfall_ingest_dag, not on a fixed schedule
    catchup=False,
)
def mtg_transform_dag():

    @task
    def download_from_s3(logical_date=None) -> str:
        """Download the raw Scryfall JSON for this run's date from S3 to /tmp/."""
        from airflow.providers.amazon.aws.hooks.s3 import S3Hook
        import os

        run_date = logical_date.strftime("%Y-%m-%d")
        s3_key = f"raw/scryfall/dt={run_date}/all_cards.jsonl.gz"
        local_path = f"/tmp/all_cards_{run_date}.jsonl.gz"

        hook = S3Hook(aws_conn_id="aws_default")
        tmp_path = hook.download_file(
            key=s3_key,
            bucket_name="mtg-price-predictor-phoom",
            local_path="/tmp",
        )
        os.rename(tmp_path, local_path)
        print(f"Downloaded s3://mtg-price-predictor-phoom/{s3_key} → {local_path}")
        return local_path

    @task.bash
    def run_transform(input_path: str) -> str:
        """Run PySpark transform. Writes Parquet to /tmp/card_price_features and loads Snowflake."""
        transform_script = Path(__file__).resolve().parents[1] / "spark" / "transform.py"
        return (
            f"source ~/airflow-learning/venv/bin/activate && "
            f"python3 {shlex.quote(str(transform_script))} {shlex.quote(input_path)}"
        )

    @task.bash
    def upload_parquet_to_s3(run_transform_done: str) -> str:
        """Sync curated Parquet from /tmp to S3 curated layer."""
        return (
            "aws s3 sync /tmp/card_price_features/ "
            "s3://mtg-price-predictor-phoom/curated/card_price_features/ "
            "--exclude '*' --include '*.parquet'"
        )

    local_path = download_from_s3()
    transform_done = run_transform(local_path)
    upload_parquet_to_s3(transform_done)


mtg_transform_dag()
