"""
Experiment 10: Real-World Application — Predictive Modelling & Decision Support System.
Integrates trained regression and classification models into an operational decision-support tool
for college librarians with live inference, advisory recommendations, and execution time benchmarks.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import joblib
from datetime import date, timedelta
from database.db import get_all_books, get_all_students, get_master_dataset
from utils.helpers import render_lab_banner, download_csv_button
from utils.visualizations import plot_bar, DARK_LAYOUT
import plotly.graph_objects as go

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

def render_predictive_analytics():
    render_lab_banner(
        experiment_num=10,
        title="Real-World Application: Predictive Modelling & Decision Support",
        description="Operational decision-support console for librarians featuring real-time overdue risk scoring, semester book-demand forecasting, model execution profiling, and ethical AI safeguards.",
        tags=["Decision Support System", "Real-Time Inference", "Demand Forecasting", "Latency Profiling", "Ethical AI"]
    )

    books_df = get_all_books()
    students_df = get_all_students()
    master_df = get_master_dataset()

    if books_df.empty or students_df.empty:
        st.warning("Ensure books and students are available in the database.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "🛡️ Librarian Live Risk Console",
        "📈 Book Demand Forecaster",
        "⚡ Execution Time Benchmarking",
        "⚖️ Ethical AI Governance & Policy"
    ])

    # Load models if available, or train on the fly
    rf_model_path = os.path.join(MODELS_DIR, "random_forest_risk.joblib")
    log_model_path = os.path.join(MODELS_DIR, "logistic_regression_risk.joblib")
    reg_model_path = os.path.join(MODELS_DIR, "linear_regression_demand.joblib")
    ridge_model_path = os.path.join(MODELS_DIR, "ridge_demand.joblib")

    # TAB 1: Live Risk Console
    with tab1:
        st.subheader("Interactive Overdue Risk Assessment Console")
        st.markdown("Evaluate risk for a student checking out a library book to determine optimal reminder workflows.")

        col_s, col_b = st.columns(2)
        with col_s:
            student_dict = {f"{r['student_id']} - {r['name']} ({r['department']}, Yr {r['year_of_study']})": r['student_id'] for _, r in students_df.iterrows()}
            selected_stu_str = st.selectbox("Select Registered Student", list(student_dict.keys()), key="pred_stu_sel")
            stu_id = student_dict[selected_stu_str]
            stu_info = students_df[students_df["student_id"] == stu_id].iloc[0]

        with col_b:
            book_dict = {f"{r['book_id']} - {r['title']} ({r['category']})": r['book_id'] for _, r in books_df.iterrows()}
            selected_bk_str = st.selectbox("Select Target Book", list(book_dict.keys()), key="pred_bk_sel")
            bk_id = book_dict[selected_bk_str]
            bk_info = books_df[books_df["book_id"] == bk_id].iloc[0]

        # Calculate student historical borrow frequency
        stu_txns = master_df[master_df["student_id"] == stu_id] if not master_df.empty else pd.DataFrame()
        stu_freq = len(stu_txns) + 1

        # Prediction input dataframe
        today_obj = date.today()
        input_data = pd.DataFrame([{
            "borrowing_frequency": stu_freq,
            "student_year": int(stu_info["year_of_study"]),
            "book_pub_year": int(bk_info["publication_year"]),
            "book_total_copies": int(bk_info["total_copies"]),
            "book_age": today_obj.year - int(bk_info["publication_year"]),
            "issue_month": today_obj.month,
            "issue_day_num": today_obj.weekday(),
            "book_category": bk_info["category"],
            "student_department": stu_info["department"]
        }])

        model_choice = st.radio("Predictive Engine", ["Random Forest Ensemble (Balanced)", "Logistic Regression (Log-Odds)"], horizontal=True)

        if st.button("🔮 Evaluate Overdue Risk Probability", type="primary", use_container_width=True):
            model_to_use = None
            if "Random Forest" in model_choice and os.path.exists(rf_model_path):
                model_to_use = joblib.load(rf_model_path)
            elif os.path.exists(log_model_path):
                model_to_use = joblib.load(log_model_path)

            if model_to_use is None:
                # Train fallback model quickly
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import StandardScaler, OneHotEncoder
                from sklearn.pipeline import Pipeline
                
                prep = ColumnTransformer([
                    ("num", StandardScaler(), ["borrowing_frequency", "student_year", "book_pub_year", "book_total_copies", "book_age", "issue_month", "issue_day_num"]),
                    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), ["book_category", "student_department"])
                ])
                model_to_use = Pipeline([("prep", prep), ("rf", RandomForestClassifier(n_estimators=50, random_state=42))])
                model_to_use.fit(
                    master_df[["borrowing_frequency", "student_year", "book_pub_year", "book_total_copies", "book_age", "issue_month", "issue_day_num", "book_category", "student_department"]],
                    master_df["overdue_status"]
                )

            start_t = time.perf_counter()
            prob = model_to_use.predict_proba(input_data)[0][1]
            latency_ms = (time.perf_counter() - start_t) * 1000

            prob_pct = prob * 100

            col_g, col_rec = st.columns([1, 2])
            with col_g:
                # Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_pct,
                    title={'text': "Predicted Overdue Probability (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#ef4444" if prob_pct > 50 else ("#f59e0b" if prob_pct > 25 else "#10b981")},
                        'steps': [
                            {'range': [0, 25], 'color': "rgba(16, 185, 129, 0.2)"},
                            {'range': [25, 50], 'color': "rgba(245, 158, 11, 0.2)"},
                            {'range': [50, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                        ],
                        'threshold': {'line': {'color': "white", 'width': 3}, 'thickness': 0.75, 'value': 50}
                    }
                ))
                fig_gauge.update_layout(**DARK_LAYOUT, height=260)
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_rec:
                st.markdown("#### Decision Support Advisory")
                if prob_pct <= 25:
                    st.success(f"🟢 **LOW RISK TIER ({prob_pct:.1f}%)**")
                    st.write("• **Recommended Action:** Standard 14-day checkout with default receipt notification.")
                    st.write("• **Intervention Level:** None required.")
                elif prob_pct <= 50:
                    st.warning(f"🟡 **MODERATE RISK TIER ({prob_pct:.1f}%)**")
                    st.write("• **Recommended Action:** Schedule automated courtesy reminder email 3 days prior to due date.")
                    st.write("• **Intervention Level:** Standard automated notifications.")
                else:
                    st.error(f"🔴 **ELEVATED RISK TIER ({prob_pct:.1f}%)**")
                    st.write("• **Recommended Action:** Friendly librarian verbal check-in: 'Please ensure return or renewal before due date.'")
                    st.write("• **Intervention Level:** Enhanced multi-channel notifications (SMS + Email).")

                st.caption(f"⚡ Inference Latency: **{latency_ms:.2f} ms**")

    # TAB 2: Book Demand Forecaster
    with tab2:
        st.subheader("Semester Circulation Demand Forecaster")
        st.markdown("Forecast expected borrowing demand for book acquisitions and shelf-stocking planning.")

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            cat_fore = st.selectbox("Book Category", books_df["category"].unique().tolist())
            copies_fore = st.slider("Number of Purchased Physical Copies", min_value=1, max_value=15, value=5)
        with col_f2:
            pub_year_fore = st.number_input("Publication Year", min_value=1980, max_value=2025, value=2022)
            age_fore = date.today().year - pub_year_fore

        reg_input = pd.DataFrame([{
            "book_pub_year": pub_year_fore,
            "book_total_copies": copies_fore,
            "book_age": age_fore,
            "book_category": cat_fore
        }])

        reg_model = None
        if os.path.exists(ridge_model_path):
            reg_model = joblib.load(ridge_model_path)
        elif os.path.exists(reg_model_path):
            reg_model = joblib.load(reg_model_path)

        if reg_model is not None:
            pred_demand = max(1.0, reg_model.predict(reg_input)[0])
            st.metric("Predicted Semester Borrow Count", f"{pred_demand:.1f} loans", f"~{pred_demand/copies_fore:.1f} circulations per copy")
            
            if pred_demand / copies_fore > 3.0:
                st.warning("⚠️ **High Utilization Alert:** Predicted circulation exceeds 3x per copy. Consider procuring 2 additional copies to prevent student queues.")
            else:
                st.success("✅ **Sufficient Inventory:** Planned copy count comfortably meets expected semester demand.")

    # TAB 3: Execution Time Benchmarking
    with tab3:
        st.subheader("Model Execution Profiling & Latency Benchmarks")
        st.write("Quantitative runtime profiling comparing training duration (ms) and inference latency (μs) across statistical algorithms.")

        # Synthetic profiling on standardized master slice
        sample_X = master_df[["borrowing_frequency", "student_year", "book_pub_year", "book_total_copies", "book_age", "issue_month", "issue_day_num", "book_category", "student_department"]].copy()
        sample_y = master_df["overdue_status"].copy()

        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.naive_bayes import GaussianNB
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.pipeline import Pipeline

        prep_bm = ColumnTransformer([
            ("num", StandardScaler(), ["borrowing_frequency", "student_year", "book_pub_year", "book_total_copies", "book_age", "issue_month", "issue_day_num"]),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), ["book_category", "student_department"])
        ])

        benchmark_clfs = {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Decision Tree": DecisionTreeClassifier(max_depth=5),
            "Gaussian Naive Bayes": GaussianNB(),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Support Vector Machine": SVC(kernel="rbf", probability=True),
            "Random Forest (100 Trees)": RandomForestClassifier(n_estimators=100)
        }

        bench_results = []
        for name, clf in benchmark_clfs.items():
            pipe = Pipeline([("prep", prep_bm), ("clf", clf)])
            
            # Training time
            t0 = time.perf_counter()
            pipe.fit(sample_X, sample_y)
            train_time_ms = (time.perf_counter() - t0) * 1000

            # Inference latency (100 runs single record)
            single_rec = sample_X.iloc[[0]]
            t_inf_0 = time.perf_counter()
            for _ in range(50):
                pipe.predict(single_rec)
            inf_time_us = ((time.perf_counter() - t_inf_0) / 50) * 1_000_000

            bench_results.append({
                "Algorithm": name,
                "Training Time (ms)": round(train_time_ms, 2),
                "Inference Latency (μs)": round(inf_time_us, 2),
                "Real-Time Suitability": "Ultra-Fast (<1ms)" if inf_time_us < 1000 else "Standard"
            })

        df_bench = pd.DataFrame(bench_results)
        st.dataframe(df_bench, use_container_width=True)

        fig_bench = plot_bar(df_bench, x="Algorithm", y="Training Time (ms)", title="Model Training Time (ms) Comparison", color="Algorithm")
        st.plotly_chart(fig_bench, use_container_width=True)

    # TAB 4: Ethical AI Governance
    with tab4:
        st.subheader("Ethical AI Safeguards in Academic Decision Systems")
        st.markdown("""
        > [!CAUTION]
        > ### Cardinal Rule of Decision-Support Systems:
        > **A statistical risk probability is NOT evidence of a rule violation.**
        
        The Smart Library Management System is architected under strict Responsible AI principles:
        
        1. **Presumption of On-Time Compliance:**
           No student may have borrowing privileges curtailed, fines levied, or loan periods shortened solely on the basis of an algorithmic prediction.
        2. **Advisory Assistance Only:**
           Predictions serve exclusively to trigger **helpful courtesy reminders** (e.g. email prompts before due dates) to assist students with schedule management.
        3. **Transparent Auditing:**
           Students have the right to know how loan policies work. Demographic attributes (e.g. gender, religion, race) are excluded entirely from model inputs.
        4. **Continuous Feedback:**
           Models must be retrained each semester to avoid concept drift and unfair historical biases.
        """)
