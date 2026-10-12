"""
Experiment 2: Data Preprocessing and Feature Selection.
Demonstrates missing value imputation, duplicate removal, categorical encoding,
feature scaling, and statistical feature selection using SelectKBest and correlation analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from database.db import get_master_dataset
from utils.helpers import render_lab_banner, download_csv_button
from utils.visualizations import plot_histogram, plot_bar

def render_preprocessing():
    render_lab_banner(
        experiment_num=2,
        title="Data Preprocessing and Feature Selection",
        description="Transform raw heterogeneous library records into standardized, leak-free feature vectors for regression and classification models.",
        tags=["Data Cleaning", "Categorical Encoding", "Feature Scaling", "SelectKBest", "Leakage Prevention"]
    )

    df_raw = get_master_dataset()
    if df_raw.empty:
        st.warning("Master dataset not available. Please initialize database.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "🧹 Data Cleaning & Audit",
        "🏷️ Categorical Encoding",
        "⚖️ Feature Scaling Comparison",
        "🎯 Feature Selection (SelectKBest)"
    ])

    # TAB 1: Data Cleaning & Audit
    with tab1:
        st.subheader("Data Quality Inspection & Cleansing")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records Analyzed", len(df_raw))
        with col2:
            missing_count = df_raw.isnull().sum().sum()
            st.metric("Missing Value Cells", missing_count)
        with col3:
            dup_count = df_raw.duplicated(subset=["transaction_id"]).sum()
            st.metric("Duplicate Transaction IDs", dup_count)

        st.markdown("#### Column-wise Null and DataType Audit")
        audit_df = pd.DataFrame({
            "Data Type": df_raw.dtypes.astype(str),
            "Null Count": df_raw.isnull().sum(),
            "Null Percentage (%)": (df_raw.isnull().sum() / len(df_raw) * 100).round(2),
            "Unique Cardinality": df_raw.nunique()
        })
        st.dataframe(audit_df, use_container_width=True)

        st.markdown("""
        > [!NOTE]
        > In the transaction schema, `return_date` has null values **only** for actively issued books (which have not yet been returned to the library). When training models to predict overdue risk prior to checkout, we exclude post-return timestamps to prevent **target leakage**.
        """)

    # TAB 2: Categorical Encoding
    with tab2:
        st.subheader("Categorical Feature Encoding")
        st.write("Machine learning models require numerical tensors. Categorical attributes such as `book_category` and `student_department` must be transformed.")

        enc_method = st.radio("Encoding Strategy", ["One-Hot Encoding (OHE)", "Label / Ordinal Encoding"], horizontal=True)

        cat_cols = ["book_category", "student_department", "issue_day_of_week"]
        sample_cat = df_raw[cat_cols].copy()

        if enc_method == "One-Hot Encoding (OHE)":
            st.markdown("##### One-Hot Encoded Representation (Dummy Variables)")
            ohe = OneHotEncoder(sparse_output=False, drop="first")
            ohe_encoded = ohe.fit_transform(sample_cat)
            ohe_cols = ohe.get_feature_names_out(cat_cols)
            df_ohe = pd.DataFrame(ohe_encoded, columns=ohe_cols)
            st.dataframe(pd.concat([sample_cat.head(10), df_ohe.head(10)], axis=1), use_container_width=True)
            st.caption(f"Expanded {len(cat_cols)} categorical columns into {df_ohe.shape[1]} binary dummy indicators (drop='first' avoids dummy variable trap / multicollinearity).")
        else:
            st.markdown("##### Label Encoded Representation")
            df_le = sample_cat.copy()
            mappings = {}
            for col in cat_cols:
                le = LabelEncoder()
                df_le[col + "_encoded"] = le.fit_transform(df_le[col])
                mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
            
            st.dataframe(df_le.head(15), use_container_width=True)
            st.json(mappings)

    # TAB 3: Feature Scaling Comparison
    with tab3:
        st.subheader("Feature Scaling: Standardization vs. Normalization")
        st.write("Algorithms sensitive to feature magnitudes (e.g. Ridge/Lasso, KNN, SVM, PCA) require features to be scaled uniformly.")

        scale_feature = st.selectbox("Select Numerical Feature to Inspect Scaling", ["borrowing_frequency", "book_pub_year", "book_total_copies", "book_age"])

        original_vals = df_raw[[scale_feature]].values
        
        # Apply standard, minmax, robust
        scaler_std = StandardScaler()
        scaler_minmax = MinMaxScaler()
        scaler_robust = RobustScaler()

        scaled_std = scaler_std.fit_transform(original_vals).flatten()
        scaled_minmax = scaler_minmax.fit_transform(original_vals).flatten()
        scaled_robust = scaler_robust.fit_transform(original_vals).flatten()

        comparison_df = pd.DataFrame({
            "Original": original_vals.flatten(),
            "StandardScaler (Z-Score)": scaled_std,
            "MinMaxScaler [0, 1]": scaled_minmax,
            "RobustScaler (IQR)": scaled_robust
        })

        st.dataframe(comparison_df.describe().T, use_container_width=True)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            fig_before = plot_histogram(comparison_df, x="Original", title=f"Before Scaling: {scale_feature}")
            st.plotly_chart(fig_before, use_container_width=True)
        with col_p2:
            fig_after = plot_histogram(comparison_df, x="StandardScaler (Z-Score)", title=f"After StandardScaler: {scale_feature} (μ=0, σ=1)")
            st.plotly_chart(fig_after, use_container_width=True)

    # TAB 4: Feature Selection (SelectKBest)
    with tab4:
        st.subheader("Statistical Feature Selection (SelectKBest)")
        st.write("Identify which candidate features contain the strongest statistical relationship with target outcomes.")

        task_choice = st.radio("Select Prediction Task for Feature Evaluation", [
            "Task A: Overdue Risk Classification (Target: overdue_status)",
            "Task B: Book Borrow Demand Regression (Target: book_borrow_count)"
        ])

        if "Task A" in task_choice:
            # Overdue risk features
            # Available at checkout: borrowing_frequency, student_year, book_pub_year, book_total_copies, book_age, issue_month, issue_day_num
            # Categorical: book_category, student_department
            feature_candidates = [
                "borrowing_frequency", "student_year", "book_pub_year", 
                "book_total_copies", "book_age", "issue_month", "issue_day_num"
            ]
            
            # Prepare clean dataset
            clean_sub = df_raw.dropna(subset=feature_candidates + ["overdue_status"])
            X = clean_sub[feature_candidates]
            y = clean_sub["overdue_status"]

            k_val = st.slider("Select Top K Features", min_value=2, max_value=len(feature_candidates), value=5)
            selector = SelectKBest(score_func=f_classif, k=k_val)
            selector.fit(X, y)

            scores_df = pd.DataFrame({
                "Feature": feature_candidates,
                "ANOVA F-Score": selector.scores_.round(3),
                "p-value": selector.pvalues_.round(5),
                "Selected": [col in X.columns[selector.get_support()] for col in feature_candidates]
            }).sort_values(by="ANOVA F-Score", ascending=False)

            st.dataframe(scores_df, use_container_width=True)
            fig_scores = plot_bar(scores_df, x="ANOVA F-Score", y="Feature", orientation="h", title="ANOVA F-Score Ranking for Overdue Risk", color="Selected")
            st.plotly_chart(fig_scores, use_container_width=True)

            st.success(f"**Selected Top {k_val} Features for Classification:** {', '.join(scores_df[scores_df['Selected']]['Feature'].tolist())}")

        else:
            # Book Demand Regression
            # Aggregate per book
            book_agg = df_raw.groupby("book_id").agg({
                "book_pub_year": "first",
                "book_total_copies": "first",
                "book_age": "first",
                "transaction_id": "count"
            }).reset_index().rename(columns={"transaction_id": "demand_count"})

            features_reg = ["book_pub_year", "book_total_copies", "book_age"]
            X_reg = book_agg[features_reg]
            y_reg = book_agg["demand_count"]

            k_reg = st.slider("Select Top K Features", min_value=1, max_value=len(features_reg), value=2)
            selector_reg = SelectKBest(score_func=f_regression, k=k_reg)
            selector_reg.fit(X_reg, y_reg)

            scores_reg_df = pd.DataFrame({
                "Feature": features_reg,
                "F-Regression Score": selector_reg.scores_.round(3),
                "p-value": selector_reg.pvalues_.round(5),
                "Selected": [col in X_reg.columns[selector_reg.get_support()] for col in features_reg]
            }).sort_values(by="F-Regression Score", ascending=False)

            st.dataframe(scores_reg_df, use_container_width=True)
            fig_reg_scores = plot_bar(scores_reg_df, x="F-Regression Score", y="Feature", orientation="h", title="F-Regression Ranking for Book Demand", color="Selected")
            st.plotly_chart(fig_reg_scores, use_container_width=True)
