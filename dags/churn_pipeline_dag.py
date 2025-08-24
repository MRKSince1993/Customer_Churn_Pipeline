# prefect_churn_pipeline.py

from prefect import flow, task
import subprocess
import os
from datetime import datetime

# ----------------------------
# Project root (Windows path)
# ----------------------------
PROJECT_ROOT = r"C:\Users\mrakkuma\Documents\vamsi\Assignment\customer_churn_pipeline"

# ----------------------------
# Helper task to run shell commands
# ----------------------------
@task(retries=2, retry_delay_seconds=60)
def run_command(command: str):
    print(f"[{datetime.now()}] Running: {command}")
    subprocess.run(command, shell=True, check=True)
    print(f"[{datetime.now()}] Done: {command}")

# ----------------------------
# 1️⃣ Ingest CSV
# ----------------------------
@task
def ingest_csv():
    # run_command(f"python {PROJECT_ROOT}/ingestion/ingest_csv.py")
    run_command("python -m ingestion.ingest_csv")

# ----------------------------
# 2️⃣ Ingest HF data
# ----------------------------
@task
def ingest_hf():
    # run_command(f"python {PROJECT_ROOT}/ingestion/ingest_hf_data.py")
    run_command("python -m ingestion.ingest_hf_data")

# ----------------------------
# 3️⃣ Store raw data
# ----------------------------
@task
def store_raw_data():
    run_command( "python -m ingestion.storage.raw_storage_manager"
        # f"python {PROJECT_ROOT}/ingestion/storage/raw_storage_manager.py "
        # f"--source_dir {PROJECT_ROOT}/temp_ingestion "
        # f"--base_dir {PROJECT_ROOT}/data/bronze"
    )

# ----------------------------
# 4️⃣ Validate data
# ----------------------------
@task
def validate_data():
    # run_command(f"python {PROJECT_ROOT}/validation/validate_data.py")
    run_command("python -m validation.validate_data")


# ----------------------------
# 5️⃣ Prepare / clean data (execute notebook)
# ----------------------------
@task
def prepare_data():
    run_command(
        f"python -m nbconvert --to script --execute {PROJECT_ROOT}/preparation/eda_and_cleaning.ipynb"
    )


# ----------------------------
# 6️⃣ Transform features
# ----------------------------
@task
def transform_features():
    # run_command(f"python {PROJECT_ROOT}/transformation/transform.py")
    run_command("python -m transformation.transform")

# ----------------------------
# 7️⃣ Materialize features
# ----------------------------
from datetime import datetime, timezone
from prefect import task

@task
def materialize_features():
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    run_command(
        f"cd {PROJECT_ROOT}/feature_repo/feature_repo && "
        f"feast apply && "
        f"feast materialize-incremental {ts}"
    )


# ----------------------------
# 8️⃣ Version datasets with DVC
# ----------------------------
@task
def version_data():
    run_command(
        f"cd {PROJECT_ROOT} && "
        f"dvc add data/silver/features.db && "
        f"git add data/silver/features.db.dvc && "
        f'git commit -m "data: Daily update via Prefect" && '
        f"dvc push"
    )

# ----------------------------
# 9️⃣ Train model
# ----------------------------
@task
def train_model():
    run_command(f"python -m modeling.train")

# ----------------------------
# Prefect Flow
# ----------------------------
@flow(name="Customer Churn End-to-End Pipeline")
def churn_pipeline():
    # Sequential execution, respecting dependencies
    ingest_csv()
    ingest_hf()
    store_raw_data()
    validate_data()
    prepare_data()
    transform_features()
    materialize_features()
    version_data()
    train_model()

# ----------------------------
# Run the flow
# ----------------------------
if __name__ == "__main__":
    churn_pipeline()
