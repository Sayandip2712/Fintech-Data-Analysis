"""
ETL Pipeline: Extracts raw data from data/raw/, performs Pandera schema validation,
aggregates transactions, cleans features, and persists to CSV and DuckDB.
"""

from pathlib import Path
import pandas as pd
from src.config import (
    RAW_CUSTOMER_PROFILES_PATH,
    RAW_DIGITAL_INTERACTIONS_PATH,
    RAW_TRANSACTION_LEDGER_PATH,
    MASTER_DATASET_PATH,
    DB_PATH
)
from src.logger import setup_logger
from src.schemas import (
    RawCustomerProfilesSchema,
    RawDigitalInteractionsSchema,
    RawTransactionLedgerSchema,
    MasterDatasetSchema
)

logger = setup_logger(__name__)


def run_etl(output_csv=MASTER_DATASET_PATH, db_path=DB_PATH):
    """Executes the extraction, validation, transformation, aggregation, and loading of raw fintech data."""
    logger.info("Starting ETL process...")
    logger.info("Reading raw tables: %s, %s, %s", RAW_CUSTOMER_PROFILES_PATH.name, RAW_DIGITAL_INTERACTIONS_PATH.name, RAW_TRANSACTION_LEDGER_PATH.name)

    df_cp = pd.read_csv(RAW_CUSTOMER_PROFILES_PATH)
    df_di = pd.read_csv(RAW_DIGITAL_INTERACTIONS_PATH)
    df_t = pd.read_csv(RAW_TRANSACTION_LEDGER_PATH)

    # Standardize column naming to lowercase for uniform processing
    df_cp.columns = df_cp.columns.str.lower()
    df_di.columns = df_di.columns.str.lower()
    df_t.columns = df_t.columns.str.lower()

    # Preprocess & cast datetime types
    df_cp['account_open_date'] = pd.to_datetime(df_cp['account_open_date'])
    df_di['interaction_date'] = pd.to_datetime(df_di['interaction_date'])
    df_t['timestamp'] = pd.to_datetime(df_t['timestamp'])

    # Clean NPS score
    df_di['nps_score'] = pd.to_numeric(df_di['nps_score'], errors='coerce')

    # Standardize transaction amount column if needed
    if 'amount' in df_t.columns and 'amount_inr' not in df_t.columns:
        df_t = df_t.rename(columns={'amount': 'amount_inr'})

    # Validate raw data contracts with Pandera
    logger.info("Validating raw data against Pandera contracts...")
    df_cp = RawCustomerProfilesSchema.validate(df_cp)
    df_di = RawDigitalInteractionsSchema.validate(df_di)
    df_t = RawTransactionLedgerSchema.validate(df_t)
    logger.info("Raw table schemas successfully verified.")

    # Aggregate transactions per customer
    amount_col = 'amount_inr'
    t_agg = df_t.groupby('customer_id').agg(
        txn_count=('customer_id', 'count'),
        txn_sum=(amount_col, 'sum'),
        txn_avg=(amount_col, 'mean'),
        txn_min=(amount_col, 'min'),
        txn_max=(amount_col, 'max')
    ).reset_index()

    # Merge into Master Dataset
    master = df_cp.merge(df_di, on='customer_id', how='inner')
    master = master.merge(t_agg, on='customer_id', how='inner')

    # Validate master dataset contract
    logger.info("Validating master dataset against Pandera schema...")
    master = MasterDatasetSchema.validate(master)

    # Ensure output directories exist and persist CSV
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output_csv, index=False)
    logger.info("Master dataset saved successfully: %s (%s rows, %s cols)", output_csv, f"{len(master):,}", len(master.columns))

    # Persist to DuckDB
    try:
        import duckdb
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(str(db_path))
        conn.register('master_df', master)
        conn.execute("CREATE OR REPLACE TABLE master_dataset AS SELECT * FROM master_df")
        conn.close()
        logger.info("Master dataset written to DuckDB: %s", db_path)
    except ImportError:
        logger.warning("DuckDB not installed; skipping database write.")

    logger.info("ETL pipeline execution finished successfully.")
    return master


if __name__ == "__main__":
    run_etl()
