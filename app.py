"""
Smart Library Management System (SLMS)
Statistical Machine Learning and Data Science Practical Syllabus — Unified Project.
Main Streamlit Application Entrypoint.
"""

import streamlit as st
import pandas as pd
import os
import sys

# Ensure local imports work reliably
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.db import init_db, get_dashboard_kpis, get_all_books, get_all_students, get_all_transactions, get_master_dataset, reset_database
from data.generate_data import initialize_data_if_needed, generate_synthetic_data, save_and_seed_database
from utils.helpers import inject_custom_css, render_kpi_card, download_csv_button
from utils.visualizations import plot_time_series, plot_pie, plot_bar

# Import laboratory and management modules
from modules.book_management import render_book_management
from modules.student_management import render_student_management
from modules.transactions import render_transactions
from modules.eda import render_eda
from modules.preprocessing import render_preprocessing
from modules.statistics_analysis import render_statistics_analysis
from modules.hypothesis_testing import render_hypothesis_testing
from modules.regression_models import render_regression_models
from modules.classification_models import render_classification_models
from modules.tree_models import render_tree_models
from modules.clustering import render_clustering
from modules.predictive_analytics import render_predictive_analytics
from modules.project_info import render_project_info

# Configure page layout and aesthetics
st.set_page_config(
    page_title="Smart Library Management System | SML & DS Lab",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database and populate synthetic dataset if empty
init_db()
initialize_data_if_needed()

# Inject modern styling
inject_custom_css()

def render_dashboard():
    """Renders the executive dashboard with real-time KPI metrics and high-level charts."""
    st.title("📚 Smart Library Management System")
    st.caption("AI-Powered Institutional Circulation, Statistical Analytics, and Decision Support Platform")

    kpis = get_dashboard_kpis()
    overdue_rate = (kpis["historical_overdue_txns"] / kpis["total_transactions"] * 100) if kpis["total_transactions"] > 0 else 0.0

    # Top KPI Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(render_kpi_card("Total Book Titles", kpis["total_titles"], f"{kpis['total_copies']} total copies in stock", "#6366f1"), unsafe_allow_html=True)
    with col2:
        st.markdown(render_kpi_card("Available Copies", kpis["available_copies"], f"{kpis['total_copies'] - kpis['available_copies']} copies currently on loan", "#10b981"), unsafe_allow_html=True)
    with col3:
        st.markdown(render_kpi_card("Active Issued Loans", kpis["active_issued"], f"{kpis['active_overdue']} currently overdue", "#f59e0b"), unsafe_allow_html=True)
    with col4:
        st.markdown(render_kpi_card("Registered Students", kpis["total_students"], f"{kpis['total_transactions']} total circulations", "#ec4899"), unsafe_allow_html=True)

    st.write("")

    # Secondary KPI Row
    c_sub1, c_sub2, c_sub3, c_sub4 = st.columns(4)
    with c_sub1:
        st.metric("Total Circulation Records", kpis["total_transactions"])
    with c_sub2:
        st.metric("Historical Overdue Ratio", f"{overdue_rate:.1f}%")
    with c_sub3:
        st.metric("Currently Overdue Active Loans", kpis["active_overdue"], delta_color="inverse")
    with c_sub4:
        st.metric("Database Status", "Active (SQLite)", "Thread-Safe")

    st.markdown("---")

    # Visual Analytics Overview
    col_chart1, col_chart2 = st.columns([3, 2])
    
    master_df = get_master_dataset()
    books_df = get_all_books()

    with col_chart1:
        if not master_df.empty:
            master_df["issue_month_yr"] = master_df["issue_date_dt"].dt.to_period("M").astype(str)
            trend_df = master_df.groupby("issue_month_yr")["transaction_id"].count().reset_index()
            trend_df.columns = ["Month", "Borrow Count"]
            fig_trend = plot_time_series(trend_df, x="Month", y="Borrow Count", title="Monthly Borrowing Circulation Velocity")
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No transaction records available to plot trends.")

    with col_chart2:
        if not books_df.empty:
            cat_df = books_df.groupby("category")["total_copies"].sum().reset_index()
            fig_pie = plot_pie(cat_df, names="category", values="total_copies", title="Catalog Domain Composition")
            st.plotly_chart(fig_pie, use_container_width=True)

    # Quick Access Shortcuts & Recent Activity
    st.markdown("### ⚡ Quick Navigation & Circulation Operations")
    col_act1, col_act2, col_act3, col_act4 = st.columns(4)
    with col_act1:
        if st.button("📤 Issue a Book", use_container_width=True):
            st.session_state["nav_selection"] = "🔄 Issue & Return"
            st.rerun()
    with col_act2:
        if st.button("📥 Return a Book", use_container_width=True):
            st.session_state["nav_selection"] = "🔄 Issue & Return"
            st.rerun()
    with col_act3:
        if st.button("🔮 Live Risk Prediction", use_container_width=True):
            st.session_state["nav_selection"] = "🛡️ Predictive Decision Support"
            st.rerun()
    with col_act4:
        if st.button("📊 Exploratory Data Analysis", use_container_width=True):
            st.session_state["nav_selection"] = "📊 Exploratory Data Analysis (Exp 1)"
            st.rerun()

    st.markdown("### 🕒 Recent Circulation Transactions")
    txns_df = get_all_transactions()
    if not txns_df.empty:
        st.dataframe(
            txns_df[[
                "transaction_id", "book_title", "student_name", "student_department", 
                "issue_date", "due_date", "return_date", "overdue_status"
            ]].head(10),
            use_container_width=True,
            hide_index=True
        )

# ==========================================
# SIDEBAR NAVIGATION & DATA MANAGEMENT
# ==========================================
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 10px 0 15px 0;">
        <h2 style="margin: 0; color: #f8fafc; font-size: 20px; font-weight: 800;">📚 SLMS Portal</h2>
        <span style="font-size: 11px; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
            Statistical ML & Data Science
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

# Navigation Options
NAVIGATION_CATEGORIES = {
    "Core Library Management": [
        "🏠 Dashboard",
        "📚 Book Management",
        "👨‍🎓 Student Management",
        "🔄 Issue & Return"
    ],
    "Data Science Lab (Exps 1-4)": [
        "📊 Exploratory Data Analysis (Exp 1)",
        "🧹 Data Preprocessing & Features (Exp 2)",
        "📐 Statistical Techniques (Exp 3)",
        "🔬 Hypothesis Testing (Exp 4)"
    ],
    "Machine Learning Suite (Exps 5-9)": [
        "📈 Regression & Regularization (Exp 5 & 6)",
        "🎯 Classification Models (Exp 7)",
        "🌳 Decision Tree & Random Forest (Exp 8)",
        "🔍 PCA & Cluster Analysis (Exp 9)"
    ],
    "Application & Evaluation (Exp 10)": [
        "🛡️ Predictive Decision Support (Exp 10)",
        "📖 Project Info & Viva Guide"
    ]
}

# Flatten list for selection state
all_nav_items = []
for items in NAVIGATION_CATEGORIES.values():
    all_nav_items.extend(items)

if "nav_selection" not in st.session_state:
    st.session_state["nav_selection"] = "🏠 Dashboard"

selected_nav = st.sidebar.radio(
    "Navigation Menu",
    all_nav_items,
    index=all_nav_items.index(st.session_state["nav_selection"]) if st.session_state["nav_selection"] in all_nav_items else 0
)
st.session_state["nav_selection"] = selected_nav

st.sidebar.markdown("---")

# Data Management Section in Sidebar
with st.sidebar.expander("⚙️ Database & Synthetic Data Tools"):
    st.caption("Manage local SQLite database records and synthetic test sets.")
    
    if st.button("🔄 Regenerate Synthetic Data", use_container_width=True):
        with st.spinner("Generating fresh 120 books, 65 students, 700 transactions..."):
            b, s, t = generate_synthetic_data()
            save_and_seed_database(b, s, t)
            st.success("Database regenerated and re-seeded successfully!")
            st.rerun()

    st.markdown("##### 📥 Upload Custom Real CSV")
    uploaded_file = st.file_uploader("Upload CSV (Books/Students/Transactions)", type=["csv"])
    if uploaded_file is not None:
        target_table = st.selectbox("Select Target Table", ["books", "students", "transactions"])
        if st.button("Upload & Append Data", use_container_width=True):
            try:
                up_df = pd.read_csv(uploaded_file)
                from database.db import get_connection
                with get_connection() as conn:
                    up_df.to_sql(target_table, conn, if_exists="append", index=False)
                    conn.commit()
                st.success(f"Appended {len(up_df)} records to '{target_table}'!")
                st.rerun()
            except Exception as e:
                st.error(f"Upload failed: {e}")

st.sidebar.markdown(
    """
    <div style="font-size: 11px; color: #64748b; padding-top: 20px;">
        🎓 <b>Third-Year Engineering Practical</b><br>
        Department of Computer Science & Engineering<br>
        Statistical Machine Learning & Data Science Lab
    </div>
    """,
    unsafe_allow_html=True
)

# Route to selected page
if selected_nav == "🏠 Dashboard":
    render_dashboard()
elif selected_nav == "📚 Book Management":
    render_book_management()
elif selected_nav == "👨‍🎓 Student Management":
    render_student_management()
elif selected_nav == "🔄 Issue & Return":
    render_transactions()
elif selected_nav == "📊 Exploratory Data Analysis (Exp 1)":
    render_eda()
elif selected_nav == "🧹 Data Preprocessing & Features (Exp 2)":
    render_preprocessing()
elif selected_nav == "📐 Statistical Techniques (Exp 3)":
    render_statistics_analysis()
elif selected_nav == "🔬 Hypothesis Testing (Exp 4)":
    render_hypothesis_testing()
elif selected_nav == "📈 Regression & Regularization (Exp 5 & 6)":
    render_regression_models()
elif selected_nav == "🎯 Classification Models (Exp 7)":
    render_classification_models()
elif selected_nav == "🌳 Decision Tree & Random Forest (Exp 8)":
    render_tree_models()
elif selected_nav == "🔍 PCA & Cluster Analysis (Exp 9)":
    render_clustering()
elif selected_nav == "🛡️ Predictive Decision Support (Exp 10)":
    render_predictive_analytics()
elif selected_nav == "📖 Project Info & Viva Guide":
    render_project_info()
