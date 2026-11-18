import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import shap
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix

from src.config import (
    MASTER_DATASET_PATH,
    FALLBACK_MASTER_DATASET_PATH,
    MODELS_DIR,
    CHURN_RF_MODEL_PATH,
    CHURN_GB_MODEL_PATH,
    CHURN_LR_MODEL_PATH,
    SCALER_ML_PATH,
    ENCODERS_PATH,
    KMEANS_MODEL_PATH,
    PCA_MODEL_PATH,
    SHAP_EXPLAINER_PATH,
    REFERENCE_DATE,
    CHURN_RECENCY_DAYS,
    AT_RISK_RECENCY_DAYS,
    RANDOM_STATE,
    CLUSTER_FEATURES,
    KMEANS_CLUSTERS,
    KMEANS_RANGE,
    MODEL_FEATURE_COLS
)
from src.logger import setup_logger

logger = setup_logger(__name__)


def resolve_master_dataset_path(path=None):
    """Finds the master dataset from explicit path, processed folder, or root fallback."""
    if path and Path(path).exists():
        return Path(path)
    if MASTER_DATASET_PATH.exists():
        return MASTER_DATASET_PATH
    if FALLBACK_MASTER_DATASET_PATH.exists():
        return FALLBACK_MASTER_DATASET_PATH
    raise FileNotFoundError(
        f"Master dataset not found. Checked:\n - {path}\n - {MASTER_DATASET_PATH}\n - {FALLBACK_MASTER_DATASET_PATH}"
    )


def load_master_dataset(path=None):
    """Loads and casts dates for master dataset."""
    resolved_path = resolve_master_dataset_path(path)
    df = pd.read_csv(resolved_path)
    df['account_open_date'] = pd.to_datetime(df['account_open_date'])
    df['interaction_date'] = pd.to_datetime(df['interaction_date'])
    return df


def aggregate_customer_data(df):
    """Aggregates interactions and transactions to customer-level records."""
    customer_df = df.groupby('customer_id').agg({
        'interaction_date': ['max', 'min', 'count'],
        'age': 'first',
        'account_type': 'first',
        'location_tier': 'first',
        'account_open_date': 'first',
        'platform_logged': lambda x: x.mode()[0] if not x.empty else 'Unknown',
        'service_ticket_raised': lambda x: (x == 'Yes').sum(),
        'nps_score': 'mean',
        'txn_count': 'sum',
        'txn_sum': 'sum',
        'txn_avg': 'mean',
        'txn_min': 'min',
        'txn_max': 'max'
    }).reset_index()

    customer_df.columns = [
        'customer_id', 'last_interaction', 'first_interaction', 'interaction_count',
        'age', 'account_type', 'location_tier', 'account_open_date', 'primary_platform',
        'ticket_count', 'avg_nps', 'total_txn_count', 'total_txn_sum', 'avg_txn_amount',
        'min_txn', 'max_txn'
    ]
    return customer_df


def engineer_features(customer_df, reference_date=None):
    """
    Derives recency, tenure, frequencies, churn target, and NPS segments.
    If reference_date is None, dynamically uses the latest interaction date in the dataset.
    """
    customer_df = customer_df.copy()
    if reference_date is None:
        if REFERENCE_DATE is not None:
            current_date = pd.to_datetime(REFERENCE_DATE)
        else:
            current_date = customer_df['last_interaction'].max()
    else:
        current_date = pd.to_datetime(reference_date)

    customer_df['recency_days'] = np.maximum((current_date - customer_df['last_interaction']).dt.days, 0)
    customer_df['tenure_days'] = np.maximum((current_date - customer_df['account_open_date']).dt.days, 0)
    customer_df['tenure_years'] = np.maximum(customer_df['tenure_days'] / 365.25, 0.01)
    customer_df['txn_frequency'] = customer_df['total_txn_count'] / (customer_df['tenure_years'] + 0.01)
    customer_df['avg_txn_size'] = customer_df['total_txn_sum'] / np.maximum(customer_df['total_txn_count'], 1)
    customer_df['ticket_rate'] = customer_df['ticket_count'] / np.maximum(customer_df['interaction_count'], 1)
    customer_df['interaction_frequency'] = customer_df['interaction_count'] / (customer_df['tenure_years'] + 0.01)

    # Churn Target Definition
    def churn_risk(row):
        if row['recency_days'] > CHURN_RECENCY_DAYS:
            return 'Churned'
        elif row['recency_days'] > AT_RISK_RECENCY_DAYS:
            return 'At Risk'
        else:
            return 'Active'

    customer_df['churn_risk'] = customer_df.apply(churn_risk, axis=1)
    customer_df['is_churned'] = (customer_df['churn_risk'] == 'Churned').astype(int)

    # NPS Segments
    customer_df['NPS_Segment'] = pd.cut(
        customer_df['avg_nps'],
        bins=[0, 4, 6, 8, 10],
        labels=['Detractors (1-4)', 'Passives (5-6)', 'Promoters (7-8)', 'Champions (9-10)']
    )

    # RFM Segment Logic
    customer_df['R_Score'] = pd.qcut(customer_df['tenure_years'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    customer_df['F_Score'] = pd.qcut(customer_df['txn_frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    customer_df['M_Score'] = pd.qcut(customer_df['total_txn_sum'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    customer_df['RFM_Score'] = customer_df['R_Score'].astype(str) + customer_df['F_Score'].astype(str) + customer_df['M_Score'].astype(str)

    def rfm_segment(row):
        if row['R_Score'] >= 4 and row['F_Score'] >= 4 and row['M_Score'] >= 4:
            return 'Champions'
        elif row['R_Score'] >= 3 and row['F_Score'] >= 3 and row['M_Score'] >= 3:
            return 'Loyal Customers'
        elif row['R_Score'] >= 4 and row['F_Score'] <= 2:
            return 'New Customers'
        elif row['F_Score'] >= 4 and row['M_Score'] >= 4:
            return 'Big Spenders'
        elif row['R_Score'] <= 2 and row['F_Score'] >= 3:
            return 'At Risk'
        elif row['R_Score'] <= 2 and row['F_Score'] <= 2:
            return 'Lost Customers'
        else:
            return 'Regular Customers'

    customer_df['RFM_Segment'] = customer_df.apply(rfm_segment, axis=1)
    return customer_df


def perform_clustering(customer_df, n_clusters=KMEANS_CLUSTERS):
    """Executes K-Means clustering and PCA 2D dimensionality reduction."""
    scaler_km = StandardScaler()
    X_scaled_km = scaler_km.fit_transform(customer_df[CLUSTER_FEATURES].fillna(0))

    inertias = []
    for k in KMEANS_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_scaled_km)
        inertias.append(km.inertia_)

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    customer_df['Cluster'] = kmeans.fit_predict(X_scaled_km)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled_km)
    customer_df['PCA1'] = X_pca[:, 0]
    customer_df['PCA2'] = X_pca[:, 1]

    clustering_artifacts = {
        'kmeans': kmeans,
        'scaler_km': scaler_km,
        'pca': pca,
        'pca_ratios': pca.explained_variance_ratio_,
        'K_range': KMEANS_RANGE,
        'inertias': inertias
    }
    return customer_df, clustering_artifacts


def train_churn_models(customer_df):
    """Trains Logistic Regression, Gradient Boosting, and Random Forest for churn prediction."""
    model_df = customer_df.copy()
    le_acct, le_loc, le_plat = LabelEncoder(), LabelEncoder(), LabelEncoder()
    model_df['account_type_enc'] = le_acct.fit_transform(model_df['account_type'].astype(str))
    model_df['location_tier_enc'] = le_loc.fit_transform(model_df['location_tier'].astype(str))
    model_df['primary_platform_enc'] = le_plat.fit_transform(model_df['primary_platform'].astype(str))

    encoders = {
        'account_type': le_acct,
        'location_tier': le_loc,
        'primary_platform': le_plat
    }

    feature_cols = MODEL_FEATURE_COLS
    X = model_df[feature_cols].fillna(0)
    y = model_df['is_churned']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    scaler_ml = StandardScaler()
    X_train_scaled = scaler_ml.fit_transform(X_train)
    X_test_scaled = scaler_ml.transform(X_test)

    # 1. Logistic Regression
    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced')
    lr.fit(X_train_scaled, y_train)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]

    # 2. Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE)
    gb.fit(X_train, y_train)
    gb_proba = gb.predict_proba(X_test)[:, 1]

    # 3. Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE, class_weight='balanced')
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    # 4. SHAP TreeExplainer
    explainer = shap.TreeExplainer(rf)

    # Evaluation metrics
    metrics = {
        'rf_auc': roc_auc_score(y_test, rf_proba),
        'rf_prauc': average_precision_score(y_test, rf_proba),
        'rf_f1': f1_score(y_test, rf_pred),
        'gb_auc': roc_auc_score(y_test, gb_proba),
        'gb_prauc': average_precision_score(y_test, gb_proba),
        'lr_auc': roc_auc_score(y_test, lr_proba),
        'lr_prauc': average_precision_score(y_test, lr_proba),
    }

    # Assign predictions back to customer_df
    customer_df['Churn_Probability'] = rf.predict_proba(X)[:, 1]

    def retention_strategy(row):
        if row['Churn_Probability'] >= 0.7:
            return 'Critical Risk - Immediate Action'
        elif row['Churn_Probability'] >= 0.5:
            return 'High Risk - Proactive Outreach'
        elif row['Churn_Probability'] >= 0.3:
            return 'Medium Risk - Monitor Closely'
        else:
            return 'Low Risk - Maintain Engagement'

    customer_df['Retention_Strategy'] = customer_df.apply(retention_strategy, axis=1)
    customer_df['Customer_Value'] = pd.qcut(
        customer_df['total_txn_sum'], 3, labels=['Low Value', 'Medium Value', 'High Value']
    )

    model_artifacts = {
        'rf': rf,
        'gb': gb,
        'lr': lr,
        'shap_explainer': explainer,
        'encoders': encoders,
        'scaler_ml': scaler_ml,
        'feature_cols': feature_cols,
        'y_test': y_test,
        'rf_pred': rf_pred,
        'rf_proba': rf_proba,
        'lr_proba': lr_proba,
        'gb_proba': gb_proba,
        'metrics': metrics
    }
    return customer_df, model_artifacts


def run_full_pipeline(dataset_path=None):
    """
    Executes the end-to-end data pipeline:
    Loads data, aggregates customers, engineers features, clusters, and trains churn models.
    """
    logger.info("Executing run_full_pipeline()...")
    raw_df = load_master_dataset(dataset_path)
    customer_df = aggregate_customer_data(raw_df)
    customer_df = engineer_features(customer_df)
    customer_df, cluster_artifacts = perform_clustering(customer_df)
    customer_df, model_artifacts = train_churn_models(customer_df)

    artifacts = {**cluster_artifacts, **model_artifacts}
    logger.info("Pipeline execution complete.")
    return customer_df, artifacts


def predict_single_customer(customer_input_dict, artifacts):
    """
    Calculates churn risk probability, prescriptive action, and SHAP feature attribution
    for a single customer profile.
    """
    encoders = artifacts['encoders']
    rf_model = artifacts['rf']

    # Robust encoding with safe fallback for unseen/novel categorical values
    def safe_encode(encoder, val):
        val_str = str(val)
        if val_str in encoder.classes_:
            return int(encoder.transform([val_str])[0])
        return 0

    acct_enc = safe_encode(encoders['account_type'], customer_input_dict.get('account_type', 'Savings'))
    loc_enc = safe_encode(encoders['location_tier'], customer_input_dict.get('location_tier', 'Tier 1 (Metro)'))
    plat_enc = safe_encode(encoders['primary_platform'], customer_input_dict.get('primary_platform', 'Mobile App'))

    tenure_years = max(float(customer_input_dict.get('tenure_years', 1.0)), 0.01)
    interaction_count = max(float(customer_input_dict.get('interaction_count', 1.0)), 0.0)
    total_txn_count = max(float(customer_input_dict.get('total_txn_count', 1.0)), 0.0)
    total_txn_sum = max(float(customer_input_dict.get('total_txn_sum', 1000.0)), 0.0)
    avg_txn_amount = float(customer_input_dict.get('avg_txn_amount', total_txn_sum / max(total_txn_count, 1)))

    txn_frequency = total_txn_count / (tenure_years + 0.01)
    interaction_frequency = interaction_count / (tenure_years + 0.01)

    row = {
        'age': float(customer_input_dict.get('age', 35)),
        'account_type_enc': acct_enc,
        'location_tier_enc': loc_enc,
        'primary_platform_enc': plat_enc,
        'ticket_count': float(customer_input_dict.get('ticket_count', 0)),
        'avg_nps': float(customer_input_dict.get('avg_nps', 7.0)),
        'total_txn_count': total_txn_count,
        'total_txn_sum': total_txn_sum,
        'avg_txn_amount': avg_txn_amount,
        'min_txn': float(customer_input_dict.get('min_txn', avg_txn_amount * 0.1)),
        'max_txn': float(customer_input_dict.get('max_txn', avg_txn_amount * 3.0)),
        'tenure_years': tenure_years,
        'txn_frequency': txn_frequency,
        'interaction_frequency': interaction_frequency,
        'interaction_count': interaction_count
    }

    input_df = pd.DataFrame([row])[MODEL_FEATURE_COLS]
    churn_prob = float(rf_model.predict_proba(input_df)[:, 1][0])

    if churn_prob >= 0.70:
        risk_level = "Critical Risk"
        action = "Immediate account manager intervention, fee waiver or exclusive loyalty offer."
        badge_color = "red"
    elif churn_prob >= 0.50:
        risk_level = "High Risk"
        action = "Proactive outreach via preferred channel; resolve pending service tickets."
        badge_color = "orange"
    elif churn_prob >= 0.30:
        risk_level = "Medium Risk"
        action = "Monitor engagement closely; send automated check-in and feature discovery notifications."
        badge_color = "yellow"
    else:
        risk_level = "Low Risk (Healthy)"
        action = "Maintain standard communication; consider upsell/cross-sell products."
        badge_color = "green"

    # Compute SHAP local attribution
    shap_explainer = artifacts.get('shap_explainer')
    shap_contrib = None
    shap_explanation = None

    if shap_explainer is None:
        try:
            shap_explainer = shap.TreeExplainer(rf_model)
            artifacts['shap_explainer'] = shap_explainer
        except Exception as e:
            logger.warning("Could not instantiate SHAP TreeExplainer: %s", e)

    if shap_explainer is not None:
        try:
            exp_res = shap_explainer(input_df)
            shap_explanation = exp_res[0, :, 1] if len(exp_res.shape) == 3 else exp_res[0]
            shap_contrib = pd.DataFrame({
                'Feature': MODEL_FEATURE_COLS,
                'Value': input_df.iloc[0].values,
                'SHAP_Impact': shap_explanation.values
            }).sort_values('SHAP_Impact', key=abs, ascending=False)
        except Exception as e:
            logger.warning("SHAP single customer attribution failed: %s", e)

    return {
        'churn_probability': churn_prob,
        'risk_level': risk_level,
        'recommended_action': action,
        'badge_color': badge_color,
        'shap_contrib': shap_contrib,
        'shap_explanation': shap_explanation,
        'input_df': input_df
    }
