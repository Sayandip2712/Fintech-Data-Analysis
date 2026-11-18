"""
Model Training, Cross-Validation, Hyperparameter Tuning & Serialization Pipeline.
Trains K-Means clustering, PCA transformer, and Churn prediction models
(Random Forest with hyperparameter tuning, Gradient Boosting, Logistic Regression),
computes SHAP explainability values, and serializes artifacts into models/.
"""

import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix
)

from src.config import (
    MODELS_DIR,
    FIGURES_DIR,
    CHURN_RF_MODEL_PATH,
    CHURN_GB_MODEL_PATH,
    CHURN_LR_MODEL_PATH,
    SCALER_ML_PATH,
    ENCODERS_PATH,
    KMEANS_MODEL_PATH,
    PCA_MODEL_PATH,
    SHAP_EXPLAINER_PATH,
    PIPELINE_METRICS_PATH,
    RANDOM_STATE,
    CLUSTER_FEATURES,
    KMEANS_CLUSTERS,
    MODEL_FEATURE_COLS
)
from src.logger import setup_logger
from src.pipeline import (
    load_master_dataset,
    aggregate_customer_data,
    engineer_features
)
from src.schemas import CustomerFeatureSchema

logger = setup_logger(__name__)


def train_and_serialize_models(dataset_path=None):
    """Executes full ML training, hyperparameter tuning, SHAP generation, and serialization."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Step 1: Loading data and aggregating to customer level...")
    raw_df = load_master_dataset(dataset_path)
    customer_df = aggregate_customer_data(raw_df)
    customer_df = engineer_features(customer_df)

    # Validate feature store against schema contract
    logger.info("Validating engineered customer features against Pandera contract...")
    customer_df = CustomerFeatureSchema.validate(customer_df)
    logger.info("Feature store contract successfully verified (%s unique accounts).", f"{len(customer_df):,}")

    # 1. Unsupervised Clustering (K-Means & PCA)
    logger.info("Step 2: Fitting K-Means Clustering (k=%d) and PCA (2D)...", KMEANS_CLUSTERS)
    scaler_km = StandardScaler()
    X_km = scaler_km.fit_transform(customer_df[CLUSTER_FEATURES].fillna(0))

    kmeans = KMeans(n_clusters=KMEANS_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    customer_df['Cluster'] = kmeans.fit_predict(X_km)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_km)
    customer_df['PCA1'] = X_pca[:, 0]
    customer_df['PCA2'] = X_pca[:, 1]

    # Save Clustering Artifacts
    joblib.dump(kmeans, KMEANS_MODEL_PATH)
    joblib.dump(pca, PCA_MODEL_PATH)
    joblib.dump(scaler_km, MODELS_DIR / "scaler_kmeans.joblib")
    logger.info("Clustering artifacts persisted to %s", MODELS_DIR)

    # 2. Categorical Encoders for Churn Modeling
    logger.info("Step 3: Encoding categorical features...")
    le_acct = LabelEncoder().fit(customer_df['account_type'].astype(str))
    le_loc = LabelEncoder().fit(customer_df['location_tier'].astype(str))
    le_plat = LabelEncoder().fit(customer_df['primary_platform'].astype(str))

    encoders = {
        'account_type': le_acct,
        'location_tier': le_loc,
        'primary_platform': le_plat
    }
    joblib.dump(encoders, ENCODERS_PATH)

    model_df = customer_df.copy()
    model_df['account_type_enc'] = le_acct.transform(model_df['account_type'].astype(str))
    model_df['location_tier_enc'] = le_loc.transform(model_df['location_tier'].astype(str))
    model_df['primary_platform_enc'] = le_plat.transform(model_df['primary_platform'].astype(str))

    X = model_df[MODEL_FEATURE_COLS].fillna(0)
    y = model_df['is_churned']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Dataset split: Train = %s rows, Test = %s rows (Churn rate = %.2f%%)", 
                f"{len(X_train):,}", f"{len(X_test):,}", (y.mean() * 100))

    # Scale ML Features
    scaler_ml = StandardScaler()
    X_train_scaled = scaler_ml.fit_transform(X_train)
    X_test_scaled = scaler_ml.transform(X_test)
    joblib.dump(scaler_ml, SCALER_ML_PATH)

    # 3. Supervised Model Training & Cross-Validation
    logger.info("Step 4: Executing 5-Fold Stratified Cross-Validation & Hyperparameter Tuning...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Random Forest Hyperparameter Tuning
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [6, 8, 10, 14, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    base_rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=rf_param_grid,
        n_iter=8,
        scoring='average_precision',
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    logger.info("Best Random Forest Hyperparameters: %s (Mean CV PR-AUC: %.4f)", rf_search.best_params_, rf_search.best_score_)

    rf_pred = best_rf.predict(X_test)
    rf_proba = best_rf.predict_proba(X_test)[:, 1]
    joblib.dump(best_rf, CHURN_RF_MODEL_PATH)

    # Gradient Boosting
    logger.info("Training Gradient Boosting Classifier...")
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE)
    gb.fit(X_train, y_train)
    gb_proba = gb.predict_proba(X_test)[:, 1]
    joblib.dump(gb, CHURN_GB_MODEL_PATH)

    # Logistic Regression
    logger.info("Training Balanced Logistic Regression...")
    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced')
    lr.fit(X_train_scaled, y_train)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
    joblib.dump(lr, CHURN_LR_MODEL_PATH)

    # 4. SHAP Explainability Engine
    logger.info("Step 5: Fitting SHAP TreeExplainer and generating global feature attribution...")
    explainer = shap.TreeExplainer(best_rf)
    joblib.dump(explainer, SHAP_EXPLAINER_PATH)

    # Compute SHAP values on test set
    shap_values = explainer(X_test)
    churn_shap_values = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values

    # Generate and save SHAP summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(churn_shap_values, X_test, feature_names=MODEL_FEATURE_COLS, show=False)
    plt.title("SHAP Global Feature Importance (Churn Probability)", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    shap_plot_path = FIGURES_DIR / "shap_summary.png"
    plt.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("SHAP summary plot successfully saved to: %s", shap_plot_path)

    # 5. Comprehensive Performance Metrics
    logger.info("Step 6: Calculating hold-out evaluation metrics...")
    metrics = {
        'Random Forest': {
            'CV_PR_AUC_Mean': float(rf_search.best_score_),
            'Best_Params': rf_search.best_params_,
            'ROC_AUC': float(roc_auc_score(y_test, rf_proba)),
            'PR_AUC': float(average_precision_score(y_test, rf_proba)),
            'F1_Score': float(f1_score(y_test, rf_pred)),
            'Confusion_Matrix': confusion_matrix(y_test, rf_pred).tolist()
        },
        'Gradient Boosting': {
            'ROC_AUC': float(roc_auc_score(y_test, gb_proba)),
            'PR_AUC': float(average_precision_score(y_test, gb_proba)),
            'F1_Score': float(f1_score(y_test, (gb_proba >= 0.5).astype(int))),
            'Confusion_Matrix': confusion_matrix(y_test, (gb_proba >= 0.5).astype(int)).tolist()
        },
        'Logistic Regression': {
            'ROC_AUC': float(roc_auc_score(y_test, lr_proba)),
            'PR_AUC': float(average_precision_score(y_test, lr_proba)),
            'F1_Score': float(f1_score(y_test, (lr_proba >= 0.5).astype(int))),
            'Confusion_Matrix': confusion_matrix(y_test, (lr_proba >= 0.5).astype(int)).tolist()
        }
    }

    with open(PIPELINE_METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    logger.info("=" * 60)
    logger.info("FINAL MODEL BENCHMARK RESULTS")
    logger.info("=" * 60)
    for model_name, score_dict in metrics.items():
        logger.info("[%s]", model_name)
        logger.info("  ROC-AUC  : %.4f", score_dict['ROC_AUC'])
        logger.info("  PR-AUC   : %.4f", score_dict['PR_AUC'])
        logger.info("  F1-Score : %.4f", score_dict['F1_Score'])
    logger.info("=" * 60)
    logger.info("All artifacts successfully saved to: %s", MODELS_DIR)

    return customer_df, metrics


if __name__ == "__main__":
    train_and_serialize_models()
