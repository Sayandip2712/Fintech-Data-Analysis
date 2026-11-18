"""
Data validation schemas and contracts using Pandera.
Defines schemas for raw data ingestion, master dataset consolidation,
and engineered customer features.
"""

try:
    import pandera.pandas as pa
except ImportError:
    import pandera as pa
from pandera import Column, Check


# 1. Raw Customer Profiles Schema
RawCustomerProfilesSchema = pa.DataFrameSchema(
    columns={
        "customer_id": Column(str, nullable=False),
        "age": Column(int, Check.in_range(18, 120), nullable=False),
        "account_type": Column(str, Check.isin(["Savings", "Jan Dhan", "Salary", "Current"]), nullable=False),
        "location_tier": Column(str, Check.isin(["Tier 1 (Metro)", "Tier 2 (Urban)", "Tier 3 (Semi-Urban)", "Tier 4 (Rural)"]), nullable=False),
        "account_open_date": Column(pa.DateTime, nullable=False),
    },
    coerce=True,
    strict=False
)

# 2. Raw Digital Interactions Schema
RawDigitalInteractionsSchema = pa.DataFrameSchema(
    columns={
        "interaction_id": Column(str, nullable=False),
        "customer_id": Column(str, nullable=False),
        "interaction_date": Column(pa.DateTime, nullable=False),
        "platform_logged": Column(str, nullable=False),
        "service_ticket_raised": Column(str, Check.isin(["Yes", "No"]), nullable=False),
        "nps_score": Column(float, Check.in_range(0.0, 10.0), nullable=True),
    },
    coerce=True,
    strict=False
)

# 3. Raw Transaction Ledger Schema
RawTransactionLedgerSchema = pa.DataFrameSchema(
    columns={
        "transaction_id": Column(str, nullable=False),
        "customer_id": Column(str, nullable=False),
        "timestamp": Column(pa.DateTime, nullable=False),
        "amount_inr": Column(float, Check.greater_than_or_equal_to(0.0), nullable=False),
    },
    coerce=True,
    strict=False
)

# 4. Master Consolidated Dataset Schema
MasterDatasetSchema = pa.DataFrameSchema(
    columns={
        "customer_id": Column(str, nullable=False),
        "age": Column(int, Check.in_range(18, 120), nullable=False),
        "account_type": Column(str, nullable=False),
        "location_tier": Column(str, nullable=False),
        "account_open_date": Column(pa.DateTime, nullable=False),
        "interaction_date": Column(pa.DateTime, nullable=False),
        "platform_logged": Column(str, nullable=False),
        "service_ticket_raised": Column(str, nullable=False),
        "nps_score": Column(float, Check.in_range(0.0, 10.0), nullable=True),
        "txn_count": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
        "txn_sum": Column(float, Check.greater_than_or_equal_to(0.0), nullable=False),
        "txn_avg": Column(float, Check.greater_than_or_equal_to(0.0), nullable=False),
    },
    coerce=True,
    strict=False
)

# 5. Customer Feature Store Schema
CustomerFeatureSchema = pa.DataFrameSchema(
    columns={
        "customer_id": Column(str, nullable=False, unique=True),
        "age": Column(float, Check.in_range(18, 120), nullable=False),
        "recency_days": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
        "tenure_days": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
        "tenure_years": Column(float, Check.greater_than_or_equal_to(0.0), nullable=False),
        "total_txn_count": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
        "total_txn_sum": Column(float, Check.greater_than_or_equal_to(0.0), nullable=False),
        "is_churned": Column(int, Check.isin([0, 1]), nullable=False),
        "churn_risk": Column(str, Check.isin(["Active", "At Risk", "Churned"]), nullable=False),
        "RFM_Score": Column(str, nullable=False),
        "RFM_Segment": Column(str, nullable=False),
    },
    coerce=True,
    strict=False
)
