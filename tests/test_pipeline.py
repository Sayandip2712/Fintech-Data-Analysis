"""
Comprehensive Test Suite for the Fintech Analytics & Churn Prediction Platform.
Tests feature derivations, clustering, ML training, real-time inference,
unseen edge cases, SHAP attribution, and Pandera data validation contracts.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import (
    resolve_master_dataset_path,
    load_master_dataset,
    aggregate_customer_data,
    engineer_features,
    perform_clustering,
    train_churn_models,
    predict_single_customer
)
from src.config import (
    MASTER_DATASET_PATH,
    MODEL_FEATURE_COLS,
    RAW_CUSTOMER_PROFILES_PATH,
    RAW_DIGITAL_INTERACTIONS_PATH,
    RAW_TRANSACTION_LEDGER_PATH
)
from src.schemas import CustomerFeatureSchema

try:
    from pandera.errors import SchemaError
except ImportError:
    import pandera as pa
    SchemaError = pa.errors.SchemaError


def test_resolve_master_dataset_path():
    """Verify master dataset path resolution."""
    path = resolve_master_dataset_path()
    assert path.exists(), f"Resolved path {path} does not exist."
    assert path.name == "master_dataset.csv"


def test_load_master_dataset():
    """Verify master dataset loading and datetime conversion."""
    df = load_master_dataset()
    assert not df.empty, "Master dataset is empty."
    assert pd.api.types.is_datetime64_any_dtype(df['account_open_date']), "account_open_date should be datetime."
    assert pd.api.types.is_datetime64_any_dtype(df['interaction_date']), "interaction_date should be datetime."


def test_customer_aggregation_and_features():
    """Verify aggregation to customer level and feature engineering derivations."""
    df = load_master_dataset()
    customer_df = aggregate_customer_data(df)

    # Assert unique customers
    assert customer_df['customer_id'].is_unique, "Customer IDs must be unique after aggregation."

    # Feature engineering
    featured_df = engineer_features(customer_df)

    required_cols = [
        'recency_days', 'tenure_days', 'tenure_years', 'txn_frequency',
        'ticket_rate', 'interaction_frequency', 'churn_risk', 'is_churned',
        'RFM_Score', 'RFM_Segment', 'NPS_Segment'
    ]
    for col in required_cols:
        assert col in featured_df.columns, f"Missing engineered column: {col}"

    # Recency and tenure sanity checks
    assert (featured_df['recency_days'] >= 0).all(), "Recency days must be non-negative."
    assert (featured_df['tenure_days'] >= 0).all(), "Tenure days must be non-negative."
    assert featured_df['is_churned'].isin([0, 1]).all(), "is_churned must be binary {0, 1}."


def test_clustering_and_models():
    """Verify unsupervised clustering and supervised model output bounds."""
    df = load_master_dataset()
    customer_df = aggregate_customer_data(df)
    customer_df = engineer_features(customer_df)
    customer_df, cluster_artifacts = perform_clustering(customer_df)

    assert 'Cluster' in customer_df.columns
    assert 'PCA1' in customer_df.columns and 'PCA2' in customer_df.columns
    assert len(customer_df['Cluster'].unique()) == 5

    customer_df, model_artifacts = train_churn_models(customer_df)

    assert 'rf' in model_artifacts
    assert 'gb' in model_artifacts
    assert 'lr' in model_artifacts
    assert 'metrics' in model_artifacts

    # Check metric ranges
    metrics = model_artifacts['metrics']
    assert 0.5 <= metrics['rf_auc'] <= 1.0, f"RF AUC out of expected range: {metrics['rf_auc']}"
    assert 0.0 <= metrics['rf_prauc'] <= 1.0, f"RF PR-AUC out of expected range: {metrics['rf_prauc']}"

    # Check probabilities bounded in [0, 1]
    probs = customer_df['Churn_Probability']
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities must be in [0, 1]."


def test_predict_single_customer():
    """Verify real-time prediction for standard individual customer profile."""
    df = load_master_dataset()
    customer_df = aggregate_customer_data(df)
    customer_df = engineer_features(customer_df)
    _, model_artifacts = train_churn_models(customer_df)

    sample_customer = {
        'age': 35,
        'account_type': 'Savings',
        'location_tier': 'Tier 1 (Metro)',
        'primary_platform': 'Mobile App',
        'tenure_years': 2.5,
        'avg_nps': 3.0,
        'ticket_count': 4,
        'interaction_count': 12,
        'total_txn_count': 45,
        'total_txn_sum': 65000.0,
        'avg_txn_amount': 1444.4,
        'min_txn': 50,
        'max_txn': 5000
    }

    result = predict_single_customer(sample_customer, model_artifacts)
    assert 'churn_probability' in result
    assert 0.0 <= result['churn_probability'] <= 1.0
    assert result['risk_level'] in [
        "Critical Risk", "High Risk", "Medium Risk", "Low Risk (Healthy)"
    ]
    assert len(result['recommended_action']) > 0


def test_predict_single_customer_unseen_categories():
    """Verify robust handling of novel/unseen categorical values without throwing KeyError/ValueError."""
    df = load_master_dataset()
    customer_df = aggregate_customer_data(df)
    customer_df = engineer_features(customer_df)
    _, model_artifacts = train_churn_models(customer_df)

    unseen_customer = {
        'age': 42,
        'account_type': 'Cryptocurrency_Vault',  # Unseen account type
        'location_tier': 'Tier 99 (Mars Base)',  # Unseen location tier
        'primary_platform': 'Smartwatch Banking', # Unseen platform
        'tenure_years': 1.5,
        'avg_nps': 5.0,
        'ticket_count': 1,
        'interaction_count': 10,
        'total_txn_count': 20,
        'total_txn_sum': 25000.0
    }

    result = predict_single_customer(unseen_customer, model_artifacts)
    assert 'churn_probability' in result
    assert 0.0 <= result['churn_probability'] <= 1.0
    assert result['risk_level'] is not None


def test_predict_single_customer_shap_attribution():
    """Verify that predict_single_customer returns valid SHAP feature attributions."""
    df = load_master_dataset()
    customer_df = aggregate_customer_data(df)
    customer_df = engineer_features(customer_df)
    _, model_artifacts = train_churn_models(customer_df)

    customer = {
        'age': 55,
        'account_type': 'Current',
        'location_tier': 'Tier 1 (Metro)',
        'primary_platform': 'NetBanking Portal',
        'tenure_years': 4.0,
        'avg_nps': 2.0,
        'ticket_count': 5,
        'interaction_count': 8,
        'total_txn_count': 12,
        'total_txn_sum': 15000.0
    }

    result = predict_single_customer(customer, model_artifacts)
    assert 'shap_contrib' in result
    shap_df = result['shap_contrib']
    assert shap_df is not None
    assert 'Feature' in shap_df.columns and 'SHAP_Impact' in shap_df.columns
    assert len(shap_df) == len(MODEL_FEATURE_COLS)


def test_zero_tenure_and_edge_values():
    """Verify that zero tenure or zero transactions do not trigger ZeroDivisionError."""
    df = load_master_dataset()
    customer_df = aggregate_customer_data(df)
    customer_df = engineer_features(customer_df)
    _, model_artifacts = train_churn_models(customer_df)

    zero_edge_customer = {
        'age': 20,
        'account_type': 'Jan Dhan',
        'location_tier': 'Tier 4 (Rural)',
        'primary_platform': 'WhatsApp Banking',
        'tenure_years': 0.0,
        'avg_nps': 7.0,
        'ticket_count': 0,
        'interaction_count': 0,
        'total_txn_count': 0,
        'total_txn_sum': 0.0,
        'avg_txn_amount': 0.0
    }

    result = predict_single_customer(zero_edge_customer, model_artifacts)
    assert 'churn_probability' in result
    assert 0.0 <= result['churn_probability'] <= 1.0


def test_pandera_schema_validation():
    """Verify that Pandera validates valid datasets and catches intentionally corrupted records."""
    df = load_master_dataset()
    customer_df = aggregate_customer_data(df)
    featured_df = engineer_features(customer_df)

    # Valid schema check
    validated_df = CustomerFeatureSchema.validate(featured_df)
    assert len(validated_df) == len(featured_df)

    # Corrupt test 1: Negative recency should fail
    corrupt_df = featured_df.copy()
    corrupt_df.loc[0, 'recency_days'] = -50
    with pytest.raises(SchemaError):
        CustomerFeatureSchema.validate(corrupt_df)

    # Corrupt test 2: Invalid is_churned value outside {0, 1} should fail
    corrupt_df2 = featured_df.copy()
    corrupt_df2.loc[0, 'is_churned'] = 99
    with pytest.raises(SchemaError):
        CustomerFeatureSchema.validate(corrupt_df2)


def test_etl_referential_integrity():
    """Verify referential integrity between customer profiles and interaction/transaction tables."""
    df_cp = pd.read_csv(RAW_CUSTOMER_PROFILES_PATH)
    df_di = pd.read_csv(RAW_DIGITAL_INTERACTIONS_PATH)
    df_t = pd.read_csv(RAW_TRANSACTION_LEDGER_PATH)

    cp_ids = set(df_cp['Customer_ID'].str.strip())
    di_ids = set(df_di['Customer_ID'].str.strip())
    t_ids = set(df_t['Customer_ID'].str.strip())

    # Every interaction customer should have a profile
    orphaned_di = di_ids - cp_ids
    assert len(orphaned_di) == 0, f"Found {len(orphaned_di)} orphaned interaction records."

    # Every transaction customer should have a profile
    orphaned_t = t_ids - cp_ids
    assert len(orphaned_t) == 0, f"Found {len(orphaned_t)} orphaned transaction records."
