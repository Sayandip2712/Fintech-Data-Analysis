import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import warnings

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run_full_pipeline, predict_single_customer
from src.config import MASTER_DATASET_PATH, FALLBACK_MASTER_DATASET_PATH, FIGURES_DIR

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Fintech Customer Analytics & Churn Engine",
    layout="wide",
    page_icon="💳"
)

# ==============================================================================
# DESIGN SYSTEM (CSS & THEME INJECTION)
# ==============================================================================
def inject_theme():
    """Injects dark-themed, revenue-first design tokens and styles."""
    st.markdown(
        """
        <style>
        /* Import Google Fonts: DM Sans & Fuzzy Bubbles */
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Fuzzy+Bubbles:wght@400;700&display=swap');

        /* Color Tokens:
           canvas:           #1A1A1A
           surface:          #252525
           surface-elevated: #2E2E2E
           action:           #E86A4A
           action-hover:     #D55A3A
           ink:              #F5F5F5
           muted-ink:        #A0A0A0
           border:           #3A3A3A
           chart-line:       #7EC8E3
           chart-bar:        #E86A4A
           success:          #4ADE80
           danger:           #F87171
        */

        html, body {
            font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
            background-color: #1A1A1A !important;
            color: #F5F5F5 !important;
        }

        .stApp {
            background-color: #1A1A1A !important;
        }

        /* Preserve Streamlit Material Icons & Ligatures (prevents 'keyboard_arrow' text from showing) */
        [data-testid="stIconMaterial"],
        [data-testid*="Icon"],
        .material-symbols-rounded,
        .material-symbols-outlined,
        .material-icons,
        button[data-testid="stSidebarCollapseButton"] *,
        button[data-testid="baseButton-headerNoPadding"] *,
        span:has(> svg),
        i.material-symbols-rounded {
            font-family: "Material Symbols Rounded", "Material Icons", sans-serif !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #171717 !important;
            border-right: 1px solid #3A3A3A !important;
        }

        section[data-testid="stSidebar"] .stRadio label {
            color: #A0A0A0 !important;
            font-size: 0.9rem !important;
            font-weight: 500 !important;
        }

        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
            color: #F5F5F5 !important;
        }

        /* Typography & Headers */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'DM Sans', sans-serif !important;
            color: #F5F5F5 !important;
            letter-spacing: -0.02em !important;
            font-weight: 700 !important;
        }

        p, label {
            font-family: 'DM Sans', sans-serif !important;
            color: #A0A0A0;
        }

        /* Handwritten annotation */
        .handwritten-annotation {
            font-family: 'Fuzzy Bubbles', cursive, sans-serif !important;
            color: #E86A4A !important;
            font-size: 0.95rem !important;
            letter-spacing: 0.01em;
            display: inline-block;
        }

        /* Browser Chrome Preview Frame */
        .browser-chrome {
            background-color: #2E2E2E;
            border: 1px solid #3A3A3A;
            border-radius: 0.75rem;
            padding: 0.65rem 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.25rem;
        }

        .browser-dots {
            display: flex;
            gap: 0.4rem;
            align-items: center;
        }

        .browser-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
        }
        .dot-red { background-color: #FF5F56; }
        .dot-yellow { background-color: #FFBD2E; }
        .dot-green { background-color: #27C93F; }

        .browser-url-bar {
            background-color: #252525;
            border: 1px solid #3A3A3A;
            border-radius: 0.4rem;
            padding: 0.25rem 1.25rem;
            color: #A0A0A0;
            font-size: 0.8rem;
            font-family: 'DM Sans', monospace;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Metric Cards */
        [data-testid="stMetric"] {
            background-color: #252525 !important;
            border: 1px solid #3A3A3A !important;
            border-radius: 0.75rem !important;
            padding: 1rem 1.25rem !important;
            box-shadow: none !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            color: #A0A0A0 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.75rem !important;
            font-weight: 700 !important;
            color: #F5F5F5 !important;
            letter-spacing: -0.02em !important;
        }

        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            border-bottom: 1px solid #3A3A3A !important;
            background-color: transparent !important;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: transparent !important;
            border: none !important;
            color: #A0A0A0 !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
            padding: 0.6rem 1rem !important;
            border-radius: 0.5rem 0.5rem 0 0 !important;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: #F5F5F5 !important;
        }

        .stTabs [aria-selected="true"] {
            color: #F5F5F5 !important;
            border-bottom: 2px solid #E86A4A !important;
            font-weight: 700 !important;
        }

        /* Buttons & Actions */
        .stButton > button, .stFormSubmitButton > button {
            background-color: #E86A4A !important;
            color: #F5F5F5 !important;
            border: 1px solid #E86A4A !important;
            border-radius: 0.5rem !important;
            font-weight: 600 !important;
            font-size: 0.875rem !important;
            padding: 0.65rem 1.5rem !important;
            transition: all 0.15s ease-in-out !important;
            letter-spacing: 0.01em !important;
        }

        .stButton > button:hover, .stFormSubmitButton > button:hover {
            background-color: #D55A3A !important;
            border-color: #D55A3A !important;
            color: #FFFFFF !important;
            transform: translateY(-1px);
        }

        /* Form Inputs & Selects */
        input, select, textarea, div[data-baseweb="select"] {
            background-color: #252525 !important;
            border-color: #3A3A3A !important;
            color: #F5F5F5 !important;
            border-radius: 0.5rem !important;
        }

        div[data-baseweb="select"] * {
            background-color: #252525 !important;
            color: #F5F5F5 !important;
        }

        /* Dataframe & Tables */
        [data-testid="stDataFrame"] {
            border: 1px solid #3A3A3A !important;
            border-radius: 0.75rem !important;
            overflow: hidden !important;
        }

        /* Progress Bar */
        .stProgress > div > div > div > div {
            background-color: #E86A4A !important;
        }

        /* Alert Callouts */
        div[data-testid="stAlert"] {
            border-radius: 0.5rem !important;
            border: 1px solid #3A3A3A !important;
            background-color: #252525 !important;
            color: #F5F5F5 !important;
        }

        /* Clean Divider */
        hr {
            border-color: #3A3A3A !important;
            margin: 1.5rem 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

inject_theme()


# ==============================================================================
# CHART STYLING HELPER
# ==============================================================================
def apply_plot_theme():
    """Sets Matplotlib & Seaborn rcParams to adhere to the dark color palette."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#1A1A1A',
        'axes.facecolor': '#252525',
        'axes.edgecolor': '#3A3A3A',
        'axes.linewidth': 1.0,
        'axes.labelcolor': '#A0A0A0',
        'axes.labelsize': 10,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.titlecolor': '#F5F5F5',
        'xtick.color': '#A0A0A0',
        'ytick.color': '#A0A0A0',
        'text.color': '#F5F5F5',
        'grid.color': '#3A3A3A',
        'grid.alpha': 0.35,
        'grid.linestyle': '--',
        'legend.facecolor': '#252525',
        'legend.edgecolor': '#3A3A3A',
        'legend.labelcolor': '#F5F5F5',
        'font.family': 'sans-serif',
    })

apply_plot_theme()

# Palette tokens for plots
COLOR_ACTION = '#E86A4A'    # Coral primary
COLOR_LINE = '#7EC8E3'      # Cool blue trend line
COLOR_SUCCESS = '#4ADE80'   # Green positive
COLOR_DANGER = '#F87171'    # Red churn/danger
COLOR_WARN = '#FBBF24'      # Amber warning
COLOR_MUTED = '#64748B'     # Slate secondary
PALETTE_CATEGORICAL = ['#E86A4A', '#7EC8E3', '#4ADE80', '#FBBF24', '#A78BFA', '#F87171']


def render_browser_chrome(url_slug="analytics/retention-overview"):
    """Renders the browser preview chrome with traffic lights & URL bar."""
    st.markdown(
        f"""
        <div class="browser-chrome">
            <div class="browser-dots">
                <span class="browser-dot dot-red"></span>
                <span class="browser-dot dot-yellow"></span>
                <span class="browser-dot dot-green"></span>
            </div>
            <div class="browser-url-bar">
                <span>🔒</span> https://fintech-data-analysis.streamlit.app
            </div>
            <div class="handwritten-annotation">
                ✦ Live Intelligence Preview
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# --- DATA PIPELINE & MODELING (CACHED) ---
@st.cache_data(show_spinner="Computing feature store and predictive models...")
def get_processed_data_and_artifacts():
    """Executes the full data pipeline, clustering, and ML modeling using the modular src pipeline."""
    return run_full_pipeline()


# --- Application Startup ---
try:
    df, artifacts = get_processed_data_and_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Missing dataset. Please ensure 'master_dataset.csv' exists in either "
        f"'{MASTER_DATASET_PATH}' or '{FALLBACK_MASTER_DATASET_PATH}'."
    )
    st.info(f"Detailed Error: {str(e)}")
    st.stop()
except Exception as e:
    st.error(f"Error loading pipeline: {str(e)}")
    st.stop()


# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown("### 💳 **Fintech Analytics**")
    st.caption("Customer Intelligence & Churn Engine")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        [
            "1. Exploratory Data Analysis",
            "2. RFM Analysis",
            "3. Retention Analysis",
            "4. Live Churn Risk Simulator"
        ]
    )

    st.markdown("---")
    st.markdown("<p style='font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; color: #A0A0A0;'>Live Portfolio Metrics</p>", unsafe_allow_html=True)
    st.markdown(f"**Accounts Analyzed:** `{df['customer_id'].nunique():,}`")
    churn_rate = (df['is_churned'].mean() * 100)
    st.markdown(f"**Portfolio Churn:** <span style='color: {COLOR_DANGER}; font-weight: 600;'>{churn_rate:.1f}%</span>", unsafe_allow_html=True)
    st.markdown(f"**Mean NPS Score:** <span style='color: {COLOR_SUCCESS}; font-weight: 600;'>{df['avg_nps'].mean():.2f}/10</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<div class='handwritten-annotation' style='font-size: 0.85rem;'>✦ Executive Intelligence Suite</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: EXPLORATORY DATA ANALYSIS
# ==============================================================================
if page == "1. Exploratory Data Analysis":
    render_browser_chrome("analytics/exploratory-intelligence")
    
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.title("Revenue-First Customer Analytics")
        st.markdown(
            "<p style='color: #A0A0A0; font-size: 1.1rem; margin-top: -0.5rem;'>"
            "Deep customer segmentation, behavioral distribution patterns, and sentiment attribution."
            "</p>",
            unsafe_allow_html=True
        )
    with col_t2:
        st.markdown("<div style='text-align: right; padding-top: 1rem;'><span class='handwritten-annotation'>✦ Verified data cohort</span></div>", unsafe_allow_html=True)

    # Top Level KPI Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Customers", f"{df['customer_id'].nunique():,}", "Active Cohort")
    kpi2.metric("Mean Age", f"{df['age'].mean():.1f} yrs", "Median 38.0")
    kpi3.metric("Avg NPS Score", f"{df['avg_nps'].mean():.2f}", "Scale 1-10")
    kpi4.metric("Avg Transaction Volume", f"₹{df['total_txn_sum'].mean():,.0f}", "Annualized LTV")
    
    st.markdown("<hr>", unsafe_allow_html=True)

    # Tab-based sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Distribution Analysis", 
        "Bivariate Correlations", 
        "Target Variable (NPS)", 
        "Platform & Channel", 
        "K-Means Clustering (k=5)"
    ])
    
    with tab1:
        st.subheader("Univariate Distribution Analysis")
        fig1, axes1 = plt.subplots(3, 3, figsize=(18, 14))
        num_features = ['age', 'avg_nps', 'total_txn_count', 'total_txn_sum', 'avg_txn_amount', 
                        'tenure_years', 'txn_frequency', 'ticket_count', 'interaction_count']
        
        for idx, col in enumerate(num_features):
            ax = axes1[idx // 3, idx % 3]
            clean_data = df[col].dropna()
            sns.histplot(clean_data, kde=True, ax=ax, color=COLOR_ACTION, alpha=0.6, edgecolor='#1A1A1A')
            ax.set_title(f'{col}', fontsize=11, fontweight='bold', color='#F5F5F5')
            ax.set_xlabel('')
            mean_val, median_val, skew_val = clean_data.mean(), clean_data.median(), clean_data.skew()
            ax.axvline(mean_val, color=COLOR_DANGER, linestyle='--', alpha=0.85, label=f'Mean: {mean_val:.1f}')
            ax.axvline(median_val, color=COLOR_SUCCESS, linestyle='--', alpha=0.85, label=f'Median: {median_val:.1f}')
            ax.legend(fontsize=8, framealpha=0.6)
            ax.text(0.95, 0.95, f'Skew: {skew_val:.2f}', transform=ax.transAxes, fontsize=8, 
                    verticalalignment='top', horizontalalignment='right', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#2E2E2E', edgecolor='#3A3A3A', alpha=0.85))
        plt.tight_layout()
        st.pyplot(fig1)

    with tab2:
        st.subheader("Relationship & Correlation Matrix")
        fig2, axes2 = plt.subplots(2, 2, figsize=(18, 13))
        
        corr_cols = ['age', 'avg_nps', 'total_txn_count', 'total_txn_sum', 'avg_txn_amount', 
                     'tenure_years', 'txn_frequency', 'ticket_count', 'interaction_count']
        corr_matrix = df[corr_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='vlag', center=0, 
                    ax=axes2[0, 0], square=True, linewidths=0.5, linecolor='#1A1A1A',
                    cbar_kws={"shrink": 0.8})
        axes2[0, 0].set_title('Correlation Matrix (Numerical Features)', fontsize=12, fontweight='bold')

        scatter = axes2[0, 1].scatter(df['age'], df['total_txn_sum'], c=df['avg_nps'], cmap='coolwarm', alpha=0.7, s=28, edgecolors='none')
        axes2[0, 1].set_xlabel('Age')
        axes2[0, 1].set_ylabel('Total Transaction Sum (INR)')
        axes2[0, 1].set_title('Age vs Total Transaction Sum (Colored by NPS)', fontsize=12, fontweight='bold')
        cb = plt.colorbar(scatter, ax=axes2[0, 1])
        cb.set_label('Avg NPS', color='#A0A0A0')
        z = np.polyfit(df['age'].fillna(0), df['total_txn_sum'].fillna(0), 1)
        p = np.poly1d(z)
        axes2[0, 1].plot(df['age'].sort_values(), p(df['age'].sort_values()), color=COLOR_ACTION, linestyle="--", alpha=0.9, linewidth=2)

        sns.boxplot(data=df, x='account_type', y='total_txn_sum', ax=axes2[1, 0], palette=['#E86A4A', '#7EC8E3', '#4ADE80', '#FBBF24'])
        axes2[1, 0].set_title('Transaction Volume by Account Type', fontsize=12, fontweight='bold')
        axes2[1, 0].tick_params(axis='x', rotation=15)

        sns.boxplot(data=df, x='primary_platform', y='avg_nps', ax=axes2[1, 1], palette=['#7EC8E3', '#E86A4A', '#4ADE80'])
        axes2[1, 1].set_title('NPS Satisfaction by Primary Platform', fontsize=12, fontweight='bold')
        axes2[1, 1].tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        st.pyplot(fig2)

    with tab3:
        st.subheader("Target Variable Analysis (NPS Score Sentiment)")
        fig3, axes3 = plt.subplots(2, 2, figsize=(16, 11))
        
        nps_counts = df['avg_nps'].value_counts().sort_index()
        axes3[0, 0].bar(nps_counts.index, nps_counts.values, color=COLOR_ACTION, alpha=0.85, edgecolor='#1A1A1A')
        axes3[0, 0].set_title('NPS Score Distribution', fontsize=12, fontweight='bold')
        axes3[0, 0].axvline(df['avg_nps'].mean(), color=COLOR_LINE, linestyle='--', linewidth=2, label=f'Mean: {df["avg_nps"].mean():.2f}')
        axes3[0, 0].legend()

        sns.boxplot(data=df, x='location_tier', y='avg_nps', ax=axes3[0, 1], palette=PALETTE_CATEGORICAL[:4])
        axes3[0, 1].set_title('NPS Distribution by Location Tier', fontsize=12, fontweight='bold')

        hb = axes3[1, 0].hexbin(df['total_txn_sum'], df['avg_nps'], gridsize=28, cmap='inferno', mincnt=1)
        axes3[1, 0].set_title('NPS vs Transaction Volume Density', fontsize=12, fontweight='bold')
        cb2 = plt.colorbar(hb, ax=axes3[1, 0])
        cb2.set_label('Customer Density', color='#A0A0A0')

        nps_seg_counts = df['NPS_Segment'].value_counts()
        nps_seg_counts = nps_seg_counts[nps_seg_counts > 0]
        explode = [0.04] * len(nps_seg_counts)
        pie_colors = [COLOR_DANGER, COLOR_WARN, COLOR_LINE, COLOR_SUCCESS][:len(nps_seg_counts)]
        axes3[1, 1].pie(nps_seg_counts.values, labels=nps_seg_counts.index, autopct='%1.1f%%', 
                        colors=pie_colors, startangle=90, explode=explode,
                        textprops={'color': '#F5F5F5', 'fontsize': 10})
        axes3[1, 1].set_title('NPS Customer Sentiment Breakdown', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig3)

    with tab4:
        st.subheader("Platform & Channel Attribution")
        fig4, axes4 = plt.subplots(2, 2, figsize=(16, 11))
        
        platform_stats = df.groupby('primary_platform').agg({'total_txn_sum': 'mean', 'avg_nps': 'mean'}).round(2)
        x = np.arange(len(platform_stats))
        width = 0.35
        ax1_twin = axes4[0, 0].twinx()
        axes4[0, 0].bar(x - width/2, platform_stats['total_txn_sum']/1000, width, label='Avg Spend (₹K)', color=COLOR_ACTION, alpha=0.85)
        ax1_twin.bar(x + width/2, platform_stats['avg_nps'], width, label='Avg NPS', color=COLOR_LINE, alpha=0.85)
        axes4[0, 0].set_xticks(x)
        axes4[0, 0].set_xticklabels(platform_stats.index, rotation=15)
        axes4[0, 0].set_ylabel('Avg Transaction Sum (₹K)', color='#A0A0A0')
        ax1_twin.set_ylabel('Avg NPS Score', color='#A0A0A0')
        axes4[0, 0].set_title('Platform: Transaction Volume vs Satisfaction', fontsize=12, fontweight='bold')
        axes4[0, 0].legend(loc='upper left')
        ax1_twin.legend(loc='upper right')

        ticket_impact = df.groupby('ticket_count').agg({'avg_nps': 'mean', 'customer_id': 'count'}).reset_index()
        ticket_impact = ticket_impact[ticket_impact['customer_id'] >= 10]
        axes4[0, 1].bar(ticket_impact['ticket_count'], ticket_impact['avg_nps'], color=COLOR_LINE, alpha=0.85)
        axes4[0, 1].set_title('Service Ticket Escalations vs NPS Drop', fontsize=12, fontweight='bold')
        axes4[0, 1].axhline(df['avg_nps'].mean(), color=COLOR_DANGER, linestyle='--', label=f'Benchmark NPS ({df["avg_nps"].mean():.1f})')
        axes4[0, 1].legend()

        loc_stats = df.groupby('location_tier').agg({'total_txn_sum': 'mean', 'customer_id': 'count'}).round(2)
        axes4[1, 0].bar(range(len(loc_stats)), loc_stats['total_txn_sum']/1000, color=PALETTE_CATEGORICAL[:len(loc_stats)])
        axes4[1, 0].set_xticks(range(len(loc_stats)))
        axes4[1, 0].set_xticklabels(loc_stats.index, rotation=15)
        axes4[1, 0].set_title('Average Spend by Geographic Tier (₹K)', fontsize=12, fontweight='bold')

        df['Age_Group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], labels=['18-25', '26-35', '36-45', '46-55', '55+'])
        age_stats = df.groupby('Age_Group', observed=False)['total_txn_sum'].mean().round(2)
        axes4[1, 1].bar(age_stats.index.astype(str), age_stats.values/1000, color=COLOR_ACTION, alpha=0.85, edgecolor='#1A1A1A')
        axes4[1, 1].set_title('Transaction Spend by Demographic Age Tier', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig4)

    with tab5:
        st.subheader("Unsupervised Customer Segmentation (K-Means & PCA)")
        fig5, axes5 = plt.subplots(2, 2, figsize=(18, 11))
        
        axes5[0, 0].plot(artifacts['K_range'], artifacts['inertias'], marker='o', color=COLOR_LINE, linewidth=2, markersize=7)
        axes5[0, 0].set_title('Elbow Criterion for Optimal Clusters', fontsize=12, fontweight='bold')
        axes5[0, 0].axvline(5, color=COLOR_ACTION, linestyle='--', linewidth=2, label='Optimal k=5')
        axes5[0, 0].legend()

        colors = PALETTE_CATEGORICAL[:5]
        for i in range(5):
            cluster_data = df[df['Cluster'] == i]
            axes5[0, 1].scatter(cluster_data['PCA1'], cluster_data['PCA2'], c=colors[i], label=f'Cluster {i}', alpha=0.65, s=22)
        axes5[0, 1].set_xlabel(f"PCA1 ({artifacts['pca_ratios'][0]:.1%} variance)")
        axes5[0, 1].set_ylabel(f"PCA2 ({artifacts['pca_ratios'][1]:.1%} variance)")
        axes5[0, 1].set_title('K-Means Clusters in 2D Latent PCA Space', fontsize=12, fontweight='bold')
        axes5[0, 1].legend()

        for i in range(5):
            cluster_data = df[df['Cluster'] == i]
            axes5[1, 0].scatter(cluster_data['age'], cluster_data['total_txn_sum'], c=colors[i], label=f'Cluster {i}', alpha=0.6, s=20)
        axes5[1, 0].set_title('Cluster Separation: Age vs Spend Volume', fontsize=12, fontweight='bold')
        axes5[1, 0].legend()

        cluster_acct = pd.crosstab(df['Cluster'], df['account_type'], normalize='index') * 100
        cluster_acct.plot(kind='bar', stacked=True, ax=axes5[1, 1], colormap='Spectral')
        axes5[1, 1].set_title('Account Composition across Clusters (%)', fontsize=12, fontweight='bold')
        axes5[1, 1].legend(title='Account Type', bbox_to_anchor=(1.02, 1), loc='upper left')
        axes5[1, 1].tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        st.pyplot(fig5)
        
        st.markdown("**Cluster Characteristic Matrix**")
        cluster_profiles = df.groupby('Cluster').agg({
            'age': 'mean', 'total_txn_sum': 'mean', 'total_txn_count': 'mean',
            'avg_nps': 'mean', 'tenure_years': 'mean', 'customer_id': 'count'
        }).round(2)
        st.dataframe(cluster_profiles, use_container_width=True)


# ==============================================================================
# PAGE 2: RFM ANALYSIS
# ==============================================================================
elif page == "2. RFM Analysis":
    render_browser_chrome("analytics/rfm-segmentation")

    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.title("RFM Customer Segmentation")
        st.markdown(
            "<p style='color: #A0A0A0; font-size: 1.1rem; margin-top: -0.5rem;'>"
            "Behavioral scoring based on Recency (tenure), Frequency (usage rate), and Monetary (spend value)."
            "</p>",
            unsafe_allow_html=True
        )
    with col_t2:
        st.markdown("<div style='text-align: right; padding-top: 1rem;'><span class='handwritten-annotation'>✦ Revenue tiers</span></div>", unsafe_allow_html=True)
    
    # KPIs
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Segments", df['RFM_Segment'].nunique(), "Discrete Personas")
    kpi2.metric("Dominant Cohort", df['RFM_Segment'].mode()[0], "Largest Volume")
    kpi3.metric("At-Risk Accounts", f"{len(df[df['RFM_Segment'] == 'At Risk']):,}", "Urgent Outreach")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("RFM Segment Volume Distribution")
        fig_rfm, ax_rfm = plt.subplots(figsize=(8, 5.5))
        rfm_counts = df['RFM_Segment'].value_counts()
        bars = ax_rfm.barh(range(len(rfm_counts)), rfm_counts.values, color=COLOR_ACTION, alpha=0.85, edgecolor='#1A1A1A')
        ax_rfm.set_yticks(range(len(rfm_counts)))
        ax_rfm.set_yticklabels(rfm_counts.index, color='#F5F5F5', fontweight='500')
        ax_rfm.set_xlabel('Number of Accounts', color='#A0A0A0')
        for i, v in enumerate(rfm_counts.values):
            ax_rfm.text(v + 25, i, f"{v:,}", va='center', color='#F5F5F5', fontsize=9, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_rfm)

    with col2:
        st.subheader("RFM Behavioral Profile Matrix")
        rfm_profiles = df.groupby('RFM_Segment').agg({
            'age': 'mean', 'total_txn_sum': 'mean', 'total_txn_count': 'mean',
            'avg_nps': 'mean', 'tenure_years': 'mean', 'customer_id': 'count'
        }).round(2)
        rfm_profiles.columns = ['Avg Age', 'Avg Spend (₹)', 'Avg Txn Cnt', 'Avg NPS', 'Tenure (Yrs)', 'Account Count']
        st.dataframe(rfm_profiles, use_container_width=True)


# ==============================================================================
# PAGE 3: RETENTION ANALYSIS
# ==============================================================================
elif page == "3. Retention Analysis":
    render_browser_chrome("analytics/churn-retention-matrix")

    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.title("Retention Analysis & Churn Prediction")
        st.markdown(
            "<p style='color: #A0A0A0; font-size: 1.1rem; margin-top: -0.5rem;'>"
            "Machine learning churn scoring, at-risk revenue exposure, and prioritized intervention strategy."
            "</p>",
            unsafe_allow_html=True
        )
    with col_t2:
        st.markdown("<div style='text-align: right; padding-top: 1rem;'><span class='handwritten-annotation'>✦ Revenue protection</span></div>", unsafe_allow_html=True)
    
    # KPIs
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Overall Churn Rate", f"{(df['is_churned'].mean() * 100):.1f}%", "Recency > 120d")
    kpi2.metric("Mean Churn Probability", f"{(df['Churn_Probability'].mean() * 100):.1f}%", "Random Forest Model")
    at_risk_rev = df[df['Retention_Strategy'].isin(['Critical Risk - Immediate Action', 'High Risk - Proactive Outreach'])]['total_txn_sum'].sum()
    kpi3.metric("At-Risk Revenue Exposure", f"₹{at_risk_rev:,.0f}", "High & Critical Accounts")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Tab-based sections
    tab1, tab2, tab3 = st.tabs([
        "Churn Profiling & Drivers", 
        "Machine Learning Performance", 
        "Prescriptive Retention Matrix"
    ])
    
    with tab1:
        st.subheader("Churn Profiling & Risk Dynamics")
        fig6, axes6 = plt.subplots(2, 3, figsize=(20, 11))
        
        for status, color in zip(['Active', 'At Risk', 'Churned'], [COLOR_SUCCESS, COLOR_WARN, COLOR_DANGER]):
            data = df[df['churn_risk'] == status]['recency_days']
            axes6[0, 0].hist(data, bins=30, alpha=0.6, label=status, color=color, edgecolor='#1A1A1A')
        axes6[0, 0].set_title('Recency Distribution by Churn Status', fontsize=12, fontweight='bold')
        axes6[0, 0].axvline(60, color=COLOR_WARN, linestyle='--', label='At-Risk Gate (60d)')
        axes6[0, 0].axvline(120, color=COLOR_DANGER, linestyle='--', label='Churn Gate (120d)')
        axes6[0, 0].legend()
        
        churn_by_acct = df.groupby('account_type')['is_churned'].mean().reset_index()
        axes6[0, 1].bar(churn_by_acct['account_type'], churn_by_acct['is_churned']*100, color=COLOR_ACTION, alpha=0.85, edgecolor='#1A1A1A')
        axes6[0, 1].set_title('Churn Rate by Account Type (%)', fontsize=12, fontweight='bold')

        churn_by_loc = df.groupby('location_tier')['is_churned'].mean().reset_index()
        axes6[0, 2].bar(churn_by_loc['location_tier'], churn_by_loc['is_churned']*100, color=COLOR_LINE, alpha=0.85, edgecolor='#1A1A1A')
        axes6[0, 2].set_title('Churn Rate by Geographic Location Tier (%)', fontsize=12, fontweight='bold')

        for status, color in zip(['Active', 'At Risk', 'Churned'], [COLOR_SUCCESS, COLOR_WARN, COLOR_DANGER]):
            data = df[df['churn_risk'] == status]['avg_nps']
            axes6[1, 0].hist(data, bins=15, alpha=0.6, label=status, color=color, edgecolor='#1A1A1A')
        axes6[1, 0].set_title('NPS Sentiment by Churn Status', fontsize=12, fontweight='bold')
        axes6[1, 0].legend()

        df['txn_quartile'] = pd.qcut(df['total_txn_sum'], 4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
        txn_churn = df.groupby('txn_quartile', observed=False)['is_churned'].mean() * 100
        axes6[1, 1].bar(txn_churn.index.astype(str), txn_churn.values, color=[COLOR_DANGER, COLOR_WARN, COLOR_LINE, COLOR_SUCCESS], edgecolor='#1A1A1A')
        axes6[1, 1].set_title('Churn Rate by Spend Quartile (%)', fontsize=12, fontweight='bold')

        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 100], labels=['18-30', '31-40', '41-50', '51-60', '60+'])
        age_churn = df.groupby('age_group', observed=False)['is_churned'].mean() * 100
        axes6[1, 2].bar(age_churn.index.astype(str), age_churn.values, color=COLOR_ACTION, alpha=0.85, edgecolor='#1A1A1A')
        axes6[1, 2].set_title('Churn Rate across Age Cohorts (%)', fontsize=12, fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig6)

    with tab2:
        st.subheader("Machine Learning Performance & Metrics")
        
        # Executive Metrics Ribbon
        if 'metrics' in artifacts:
            m = artifacts['metrics']
            c1, c2, c3 = st.columns(3)
            c1.metric("Random Forest ROC-AUC", f"{m['rf_auc']:.4f}", "Discrimination Ability")
            c2.metric("Precision-Recall AUC (PR-AUC)", f"{m['rf_prauc']:.4f}", "Imbalanced Churn Metric")
            c3.metric("Random Forest F1-Score", f"{m['rf_f1']:.4f}", "Balanced Precision/Recall")

        fig7, axes7 = plt.subplots(2, 2, figsize=(18, 12))

        importance_df = pd.DataFrame({
            'Feature': artifacts['feature_cols'], 'Importance': artifacts['rf'].feature_importances_
        }).sort_values('Importance', ascending=True)
        axes7[0, 0].barh(importance_df['Feature'], importance_df['Importance'], color=COLOR_ACTION, alpha=0.9)
        axes7[0, 0].set_title('Random Forest: Feature Importance', fontsize=12, fontweight='bold')

        # ROC Curves
        models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting']
        probas = [artifacts['lr_proba'], artifacts['rf_proba'], artifacts['gb_proba']]
        colors = [COLOR_LINE, COLOR_ACTION, COLOR_SUCCESS]
        for m_name, proba, color in zip(models, probas, colors):
            fpr, tpr, _ = roc_curve(artifacts['y_test'], proba)
            auc = roc_auc_score(artifacts['y_test'], proba)
            axes7[0, 1].plot(fpr, tpr, color=color, linewidth=2.2, label=f'{m_name} (AUC = {auc:.4f})')
        axes7[0, 1].plot([0, 1], [0, 1], color='#64748B', linestyle='--', alpha=0.5)
        axes7[0, 1].set_title('ROC Curves Comparison', fontsize=12, fontweight='bold')
        axes7[0, 1].legend(loc='lower right')

        # Confusion Matrix
        cm = confusion_matrix(artifacts['y_test'], artifacts['rf_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes7[1, 0], 
                    xticklabels=['Retained', 'Churned'], yticklabels=['Retained', 'Churned'],
                    cbar=False)
        axes7[1, 0].set_title('Confusion Matrix (Random Forest Test Split)', fontsize=12, fontweight='bold')

        # Probability Distribution
        y_test_array = artifacts['y_test'].values 
        retained_proba = artifacts['rf_proba'][y_test_array == 0]
        churned_proba = artifacts['rf_proba'][y_test_array == 1]
        
        axes7[1, 1].hist(retained_proba, bins=30, alpha=0.6, label='Retained', color=COLOR_SUCCESS, edgecolor='#1A1A1A')
        axes7[1, 1].hist(churned_proba, bins=30, alpha=0.6, label='Churned', color=COLOR_DANGER, edgecolor='#1A1A1A')
        axes7[1, 1].axvline(0.5, color='#F5F5F5', linestyle='--', linewidth=1.5, label='Decision Threshold (0.5)')
        axes7[1, 1].set_title('Predicted Probability Density Separation', fontsize=12, fontweight='bold')
        axes7[1, 1].legend()

        plt.tight_layout()
        st.pyplot(fig7)

        # SHAP Global Explainability
        shap_plot_path = FIGURES_DIR / "shap_summary.png"
        if shap_plot_path.exists():
            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("🧠 SHAP (SHapley Additive exPlanations) Global Interpretability")
            st.markdown(
                "<p style='color: #A0A0A0; font-size: 0.95rem;'>"
                "SHAP values measure the exact directional contribution of each feature towards or against churn risk, "
                "surpassing traditional tree-based impurity importance by maintaining local accuracy and consistency."
                "</p>",
                unsafe_allow_html=True
            )
            st.image(str(shap_plot_path), caption="Global SHAP Summary Beeswarm Plot (Minority Churn Class)", use_container_width=True)

    with tab3:
        st.subheader("Prescriptive Retention Strategy Matrix")
        fig8, axes8 = plt.subplots(2, 2, figsize=(18, 12))

        strategy_counts = df['Retention_Strategy'].value_counts()
        strategy_counts = strategy_counts[strategy_counts > 0]
        explode = [0.04] * len(strategy_counts)
        strat_colors = [COLOR_SUCCESS, COLOR_WARN, COLOR_ACTION, COLOR_DANGER][:len(strategy_counts)]
        
        axes8[0, 0].pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.1f%%', 
                        colors=strat_colors, startangle=90, explode=explode,
                        textprops={'color': '#F5F5F5', 'fontsize': 10})
        axes8[0, 0].set_title('Portfolio Distribution by Urgency Tier', fontsize=12, fontweight='bold')

        value_risk_pct = pd.crosstab(df['Customer_Value'], df['Retention_Strategy'], normalize='index') * 100
        sns.heatmap(value_risk_pct, annot=True, fmt='.1f', cmap='vlag_r', ax=axes8[0, 1], cbar_kws={'label': 'Percentage'})
        axes8[0, 1].set_title('Risk Concentration by Customer LTV Tier (%)', fontsize=12, fontweight='bold')

        bins = pd.qcut(df['interaction_frequency'], 5, duplicates='drop')
        interaction_churn = df.groupby(bins, observed=True)['Churn_Probability'].mean()
        x_labels = [f'{interval.left:.1f}-{interval.right:.1f}' for interval in interaction_churn.index]
        axes8[1, 0].bar(range(len(interaction_churn)), interaction_churn.values, color=PALETTE_CATEGORICAL[:len(interaction_churn)], alpha=0.85, edgecolor='#1A1A1A')
        axes8[1, 0].set_xticks(range(len(interaction_churn)))
        axes8[1, 0].set_xticklabels(x_labels, rotation=15)
        axes8[1, 0].set_ylabel('Mean Churn Probability')
        axes8[1, 0].set_title('Churn Risk by Digital Interaction Velocity', fontsize=12, fontweight='bold')
        axes8[1, 0].axhline(0.5, color=COLOR_DANGER, linestyle='--', alpha=0.8, label='High Risk Threshold (0.50)')
        axes8[1, 0].legend()

        value_risk_summary = df.groupby(['Customer_Value', 'Retention_Strategy'], observed=True).agg(
            Count=('customer_id', 'count'), Avg_Churn_Prob=('Churn_Probability', 'mean')
        ).reset_index()
        value_map = {'Low Value': 1, 'Medium Value': 2, 'High Value': 3}
        strategy_map = {'Low Risk - Maintain Engagement': 1, 'Medium Risk - Monitor Closely': 2, 'High Risk - Proactive Outreach': 3, 'Critical Risk - Immediate Action': 4}
        value_risk_summary['Value_Num'] = value_risk_summary['Customer_Value'].map(value_map)
        value_risk_summary['Strategy_Num'] = value_risk_summary['Retention_Strategy'].map(strategy_map)
        
        scatter = axes8[1, 1].scatter(value_risk_summary['Value_Num'], value_risk_summary['Strategy_Num'], s=value_risk_summary['Count']*3, c=value_risk_summary['Avg_Churn_Prob'], cmap='coolwarm', alpha=0.8, edgecolors='#3A3A3A')
        axes8[1, 1].set_xticks([1, 2, 3])
        axes8[1, 1].set_xticklabels(['Low LTV', 'Medium LTV', 'High LTV'])
        axes8[1, 1].set_yticks([1, 2, 3, 4])
        axes8[1, 1].set_yticklabels(['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk'])
        axes8[1, 1].set_title('Prioritized Action Matrix (Bubble Size = Account Count)', fontsize=12, fontweight='bold')
        cb3 = plt.colorbar(scatter, ax=axes8[1, 1])
        cb3.set_label('Avg Churn Probability', color='#A0A0A0')
        axes8[1, 1].axhline(2.5, color='#3A3A3A', linestyle='--', alpha=0.6)
        axes8[1, 1].axvline(2.5, color='#3A3A3A', linestyle='--', alpha=0.6)
        axes8[1, 1].text(2.7, 3.8, 'CRITICAL FOCUS\n(High LTV + High Risk)', fontsize=9, fontweight='bold', color=COLOR_ACTION, ha='center')

        plt.tight_layout()
        st.pyplot(fig8)


# ==============================================================================
# PAGE 4: LIVE CHURN RISK SIMULATOR
# ==============================================================================
elif page == "4. Live Churn Risk Simulator":
    render_browser_chrome("simulate/instant-churn-evaluator")

    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.title("Interactive Churn Risk Simulator")
        st.markdown(
            "<p style='color: #A0A0A0; font-size: 1.1rem; margin-top: -0.5rem;'>"
            "Test hypothetical customer profiles against the trained model for instant churn risk scoring."
            "</p>",
            unsafe_allow_html=True
        )
    with col_t2:
        st.markdown("<div style='text-align: right; padding-top: 1rem;'><span class='handwritten-annotation'>✦ Real-time scoring</span></div>", unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)

    # Preset Persona Quick-Select
    st.markdown("##### 1. Select a Sample Persona or Customize Below")
    preset = st.selectbox(
        "Choose an archetype profile to prepopulate parameters:",
        [
            "Custom Profile",
            "High-Value VIP at Risk (Detractor, Unresolved Tickets)",
            "Active Digital Native (Frequent UPI User, High NPS)",
            "Dormant Rural Account (Infrequent, Low Balance)",
            "New Salary Customer (Recent Account, Medium Usage)"
        ]
    )

    # Defaults based on preset
    if preset == "High-Value VIP at Risk (Detractor, Unresolved Tickets)":
        d_age, d_acct, d_loc, d_plat = 48, "Current", "Tier 1 (Metro)", "NetBanking Portal"
        d_nps, d_tickets, d_tenure = 2.0, 5, 3.5
        d_txn_cnt, d_txn_sum, d_inter = 85, 250000.0, 12
    elif preset == "Active Digital Native (Frequent UPI User, High NPS)":
        d_age, d_acct, d_loc, d_plat = 26, "Savings", "Tier 1 (Metro)", "Mobile App"
        d_nps, d_tickets, d_tenure = 9.0, 0, 1.8
        d_txn_cnt, d_txn_sum, d_inter = 120, 85000.0, 45
    elif preset == "Dormant Rural Account (Infrequent, Low Balance)":
        d_age, d_acct, d_loc, d_plat = 62, "Jan Dhan", "Tier 4 (Rural)", "WhatsApp Banking"
        d_nps, d_tickets, d_tenure = 4.0, 2, 4.2
        d_txn_cnt, d_txn_sum, d_inter = 6, 8500.0, 3
    elif preset == "New Salary Customer (Recent Account, Medium Usage)":
        d_age, d_acct, d_loc, d_plat = 31, "Salary", "Tier 2 (Urban)", "Mobile App"
        d_nps, d_tickets, d_tenure = 7.0, 1, 0.6
        d_txn_cnt, d_txn_sum, d_inter = 28, 95000.0, 15
    else:
        d_age, d_acct, d_loc, d_plat = 35, "Savings", "Tier 2 (Urban)", "Mobile App"
        d_nps, d_tickets, d_tenure = 6.0, 1, 2.0
        d_txn_cnt, d_txn_sum, d_inter = 40, 50000.0, 18

    with st.form("churn_simulation_form"):
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown("<p style='font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; color: #E86A4A;'>Demographics & Profile</p>", unsafe_allow_html=True)
            age = st.slider("Customer Age", 18, 85, d_age)
            account_types = sorted(list(df['account_type'].dropna().unique()))
            account_type = st.selectbox(
                "Account Type", account_types, 
                index=account_types.index(d_acct) if d_acct in account_types else 0
            )
            location_tiers = sorted(list(df['location_tier'].dropna().unique()))
            location_tier = st.selectbox(
                "Location Tier", location_tiers,
                index=location_tiers.index(d_loc) if d_loc in location_tiers else 0
            )
            tenure_years = st.slider("Tenure with Institution (Years)", 0.1, 10.0, float(d_tenure), 0.1)

        with col_b:
            st.markdown("<p style='font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; color: #E86A4A;'>Channel & Sentiment</p>", unsafe_allow_html=True)
            platforms = sorted(list(df['primary_platform'].dropna().unique()))
            primary_platform = st.selectbox(
                "Primary Access Channel", platforms,
                index=platforms.index(d_plat) if d_plat in platforms else 0
            )
            avg_nps = st.slider("Customer NPS Rating (1 Detractor to 10 Promoter)", 1.0, 10.0, float(d_nps), 0.5)
            ticket_count = st.number_input("Support Tickets Raised (Annual)", 0, 20, int(d_tickets))
            interaction_count = st.number_input("Annual Digital Logins", 1, 250, int(d_inter))

        with col_c:
            st.markdown("<p style='font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; color: #E86A4A;'>Transaction Ledger</p>", unsafe_allow_html=True)
            total_txn_count = st.number_input("Total Transaction Count", 1, 500, int(d_txn_cnt))
            total_txn_sum = st.number_input("Total Spend Volume (INR)", 100.0, 1000000.0, float(d_txn_sum), 5000.0)
            avg_txn_amount = total_txn_sum / max(total_txn_count, 1)
            st.markdown(f"<div style='background: #252525; border: 1px solid #3A3A3A; border-radius: 0.5rem; padding: 0.6rem; color: #A0A0A0; font-size: 0.85rem;'>Calculated Avg Ticket: <span style='color: #F5F5F5; font-weight: 600;'>₹{avg_txn_amount:,.2f}</span></div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        submit = st.form_submit_button("⚡ Predict Churn Probability & Strategy", use_container_width=True)

    if submit:
        input_data = {
            'age': age,
            'account_type': account_type,
            'location_tier': location_tier,
            'primary_platform': primary_platform,
            'tenure_years': tenure_years,
            'avg_nps': avg_nps,
            'ticket_count': ticket_count,
            'interaction_count': interaction_count,
            'total_txn_count': total_txn_count,
            'total_txn_sum': total_txn_sum,
            'avg_txn_amount': avg_txn_amount,
            'min_txn': avg_txn_amount * 0.1,
            'max_txn': avg_txn_amount * 3.0
        }

        result = predict_single_customer(input_data, artifacts)
        churn_prob = result['churn_probability']
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 🔍 Model Inference Output")
        
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            # Custom styled output card
            status_color = COLOR_DANGER if result['risk_level'] in ["Critical Risk", "High Risk"] else (COLOR_WARN if result['risk_level'] == "Medium Risk" else COLOR_SUCCESS)
            st.markdown(
                f"""
                <div style="background-color: #252525; border: 1px solid #3A3A3A; border-radius: 0.75rem; padding: 1.5rem; text-align: center;">
                    <div style="font-size: 0.75rem; text-transform: uppercase; color: #A0A0A0; letter-spacing: 0.05em; font-weight: 600;">Predicted Churn Probability</div>
                    <div style="font-size: 2.75rem; font-weight: 700; color: {status_color}; margin: 0.25rem 0;">{(churn_prob * 100):.1f}%</div>
                    <div style="display: inline-block; background-color: #2E2E2E; border: 1px solid {status_color}; color: {status_color}; padding: 0.25rem 0.85rem; border-radius: 1rem; font-size: 0.85rem; font-weight: 600;">
                        {result['risk_level']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with res_col2:
            st.markdown(
                f"""
                <div style="background-color: #252525; border: 1px solid #3A3A3A; border-radius: 0.75rem; padding: 1.25rem 1.5rem;">
                    <div style="font-size: 0.75rem; text-transform: uppercase; color: #E86A4A; letter-spacing: 0.05em; font-weight: 700;">Prescriptive Action Recommendation</div>
                    <p style="color: #F5F5F5; font-size: 1.05rem; font-weight: 500; margin: 0.5rem 0 1rem 0;">{result['recommended_action']}</p>
                    <div style="font-size: 0.75rem; text-transform: uppercase; color: #A0A0A0; letter-spacing: 0.05em; font-weight: 600; margin-bottom: 0.4rem;">Attribution Factors</div>
                """,
                unsafe_allow_html=True
            )
            
            drivers = []
            if avg_nps <= 4:
                drivers.append("⚠️ **NPS Detractor (≤ 4)**: Poor sentiment strongly correlates with exit intent.")
            if ticket_count >= 3:
                drivers.append("⚠️ **Operational Friction (≥ 3 Tickets)**: Repeat service escalations.")
            if (total_txn_count / (tenure_years + 0.01)) < 15:
                drivers.append("⚠️ **Transaction Velocity Drop**: Low frequency relative to account tenure.")
            if interaction_count < 10:
                drivers.append("⚠️ **Digital Dormancy**: Infrequent mobile/web logins indicate waning engagement.")
            
            if not drivers:
                st.markdown("<p style='color: #4ADE80; font-size: 0.9rem;'>✅ No critical negative risk anomalies detected. Customer profile exhibits stable behavior.</p>", unsafe_allow_html=True)
            else:
                for d in drivers:
                    st.markdown(f"<p style='color: #F87171; font-size: 0.9rem; margin-bottom: 0.25rem;'>{d}</p>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

        # SHAP Local Explainability
        if result.get('shap_contrib') is not None and not result['shap_contrib'].empty:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("💡 Explainable AI: SHAP Feature Attribution for this Profile")
            st.markdown(
                f"<p style='color: #A0A0A0; font-size: 0.95rem;'>"
                f"Red bars indicate features that <span style='color: {COLOR_DANGER}; font-weight: 600;'>increase</span> churn risk; "
                f"green bars indicate features that <span style='color: {COLOR_SUCCESS}; font-weight: 600;'>protect</span> and retain the customer."
                f"</p>",
                unsafe_allow_html=True
            )

            shap_df = result['shap_contrib'].head(8).sort_values('SHAP_Impact', ascending=True)
            fig_shap, ax_shap = plt.subplots(figsize=(9, 4.5))
            bar_colors = [COLOR_DANGER if val > 0 else COLOR_SUCCESS for val in shap_df['SHAP_Impact']]
            ax_shap.barh(shap_df['Feature'], shap_df['SHAP_Impact'], color=bar_colors, alpha=0.85, edgecolor='#1A1A1A')
            ax_shap.axvline(0, color='#64748B', linestyle='--', linewidth=1)
            ax_shap.set_xlabel('SHAP Impact on Churn Probability', color='#A0A0A0')
            ax_shap.set_title('Top 8 SHAP Drivers for Evaluated Profile', fontsize=11, fontweight='bold')
            plt.tight_layout()

            shap_col1, shap_col2 = st.columns([3, 2])
            with shap_col1:
                st.pyplot(fig_shap)
            with shap_col2:
                st.markdown("<p style='font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; color: #A0A0A0;'>Attribution Scores</p>", unsafe_allow_html=True)
                st.dataframe(result['shap_contrib'].round(4), use_container_width=True)
