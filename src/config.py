from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_DB_DIR = DATA_DIR / "database"

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
HTML_REPORTS_DIR = REPORTS_DIR / "html"

NOTEBOOKS_DIR = BASE_DIR / "notebooks"
MODELS_DIR = BASE_DIR / "models"
TESTS_DIR = BASE_DIR / "tests"
LOGS_DIR = BASE_DIR / "logs"

# Primary & Fallback Dataset Paths
MASTER_DATASET_PATH = DATA_PROCESSED_DIR / "master_dataset.csv"
FALLBACK_MASTER_DATASET_PATH = BASE_DIR / "master_dataset.csv"
DB_PATH = DATA_DB_DIR / "master_dataset.db"

# Serialized Model Artifact Paths
CHURN_RF_MODEL_PATH = MODELS_DIR / "churn_random_forest.joblib"
CHURN_GB_MODEL_PATH = MODELS_DIR / "churn_gradient_boosting.joblib"
CHURN_LR_MODEL_PATH = MODELS_DIR / "churn_logistic_regression.joblib"
SCALER_ML_PATH = MODELS_DIR / "scaler_ml.joblib"
ENCODERS_PATH = MODELS_DIR / "label_encoders.joblib"
KMEANS_MODEL_PATH = MODELS_DIR / "kmeans_cluster.joblib"
PCA_MODEL_PATH = MODELS_DIR / "pca_transformer.joblib"
SHAP_EXPLAINER_PATH = MODELS_DIR / "shap_tree_explainer.joblib"
PIPELINE_METRICS_PATH = MODELS_DIR / "pipeline_metrics.json"

# Business & Pipeline Parameters
# If REFERENCE_DATE is None, the pipeline dynamically derives it as max(interaction_date)
REFERENCE_DATE = None
CHURN_RECENCY_DAYS = 120
AT_RISK_RECENCY_DAYS = 60
RANDOM_STATE = 42

# Clustering & PCA Settings
CLUSTER_FEATURES = ['age', 'total_txn_sum', 'total_txn_count', 'avg_nps', 'tenure_years']
KMEANS_CLUSTERS = 5
KMEANS_RANGE = range(2, 11)

# Churn Modeling Features
MODEL_FEATURE_COLS = [
    'age', 'account_type_enc', 'location_tier_enc', 'primary_platform_enc',
    'ticket_count', 'avg_nps', 'total_txn_count', 'total_txn_sum', 'avg_txn_amount',
    'min_txn', 'max_txn', 'tenure_years', 'txn_frequency', 'interaction_frequency', 'interaction_count'
]

# Raw Data Sources
RAW_CUSTOMER_PROFILES_PATH = DATA_RAW_DIR / "customer_profiles.csv"
RAW_DIGITAL_INTERACTIONS_PATH = DATA_RAW_DIR / "digital_interactions.csv"
RAW_TRANSACTION_LEDGER_PATH = DATA_RAW_DIR / "transaction_ledger.csv"
