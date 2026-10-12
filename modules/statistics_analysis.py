"""
Experiment 3: Statistical Techniques for Data Exploration.
Computes measures of central tendency, dispersion, shape (skewness/kurtosis),
outlier detection (IQR & Z-score), and bivariate relationship summaries.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from database.db import get_master_dataset
from utils.helpers import render_lab_banner, download_csv_button
from utils.visualizations import plot_histogram, plot_box

def render_statistics_analysis():
    render_lab_banner(
        experiment_num=3,
        title="Statistical Techniques for Data Exploration",
        description="Comprehensive descriptive statistics, probability distribution shapes, interquartile range (IQR) outlier detection, and category-behaviour contingency analysis.",
        tags=["Central Tendency", "Dispersion", "IQR Outlier Detection", "Skewness & Kurtosis", "Bivariate Analysis"]
    )

    df = get_master_dataset()
    if df.empty:
        st.warning("Master dataset not available.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "📐 Central Tendency & Dispersion",
        "📊 Distribution Shape (Skew/Kurtosis)",
        "🚨 Outlier Detection (IQR & Z-Score)",
        "🔄 Bivariate & Contingency Analysis"
    ])

    numeric_vars = ["overdue_days", "borrowing_frequency", "loan_duration_days", "book_pub_year", "book_total_copies", "book_age"]

    # TAB 1: Central Tendency & Dispersion
    with tab1:
        st.subheader("Measures of Central Tendency and Dispersion")
        selected_var = st.selectbox("Select Numerical Variable to Analyze", numeric_vars, index=0)

        data = df[selected_var].dropna()
        
        # Calculations
        val_mean = data.mean()
        val_median = data.median()
        val_mode_res = stats.mode(data, keepdims=True)
        val_mode = val_mode_res.mode[0] if len(val_mode_res.mode) > 0 else np.nan
        val_variance = data.var()
        val_std = data.std()
        val_min = data.min()
        val_max = data.max()
        val_range = val_max - val_min
        q1 = data.quantile(0.25)
        q2 = data.quantile(0.50)
        q3 = data.quantile(0.75)
        iqr = q3 - q1

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean (μ)", f"{val_mean:.3f}")
            st.metric("Variance (σ²)", f"{val_variance:.3f}")
        with col2:
            st.metric("Median (Q2)", f"{val_median:.3f}")
            st.metric("Standard Deviation (σ)", f"{val_std:.3f}")
        with col3:
            st.metric("Mode", f"{val_mode}")
            st.metric("Range (Max - Min)", f"{val_range:.3f}")
        with col4:
            st.metric("IQR (Q3 - Q1)", f"{iqr:.3f}")
            st.metric("Quartiles [Q1, Q3]", f"[{q1:.1f}, {q3:.1f}]")

        st.markdown("#### Comprehensive Statistical Summary Table")
        summary_rows = []
        for var in numeric_vars:
            d = df[var].dropna()
            m = d.mean()
            med = d.median()
            sd = d.std()
            vr = d.var()
            q_25 = d.quantile(0.25)
            q_75 = d.quantile(0.75)
            summary_rows.append({
                "Metric Variable": var,
                "Count": len(d),
                "Mean": round(m, 2),
                "Median": round(med, 2),
                "Std Dev": round(sd, 2),
                "Variance": round(vr, 2),
                "Min": round(d.min(), 2),
                "25% (Q1)": round(q_25, 2),
                "75% (Q3)": round(q_75, 2),
                "IQR": round(q_75 - q_25, 2),
                "Max": round(d.max(), 2)
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    # TAB 2: Distribution Shape
    with tab2:
        st.subheader("Distribution Shape: Skewness and Kurtosis")
        var_shape = st.selectbox("Select Variable for Shape Diagnostics", numeric_vars, index=0, key="shape_var")
        d_shape = df[var_shape].dropna()

        skew = stats.skew(d_shape)
        kurt = stats.kurtosis(d_shape) # Fisher kurtosis (normal = 0.0)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Skewness", f"{skew:.4f}")
            if skew > 0.5:
                st.info("Right-skewed (Positive skew): Tail extends towards higher values (e.g. rare high overdue days).")
            elif skew < -0.5:
                st.info("Left-skewed (Negative skew): Tail extends towards lower values.")
            else:
                st.info("Approximately symmetric distribution.")

        with col_s2:
            st.metric("Excess Kurtosis", f"{kurt:.4f}")
            if kurt > 1.0:
                st.info("Leptokurtic (Heavy-tailed): Fat tails with sharp peak; higher propensity for extreme outliers.")
            elif kurt < -1.0:
                st.info("Platykurtic (Light-tailed): Flat peak with thinner tails.")
            else:
                st.info("Mesokurtic: Similar peakedness to standard normal distribution.")

        fig_shape = plot_histogram(df, x=var_shape, title=f"Empirical Probability Distribution: {var_shape}", nbins=30)
        st.plotly_chart(fig_shape, use_container_width=True)

    # TAB 3: Outlier Detection
    with tab3:
        st.subheader("Outlier Detection: IQR Rule vs. Z-Score Method")
        outlier_var = st.selectbox("Select Variable for Outlier Inspection", ["overdue_days", "loan_duration_days", "borrowing_frequency"], key="out_var")
        d_out = df[outlier_var].dropna()

        # 1. IQR Method
        q1_out = d_out.quantile(0.25)
        q3_out = d_out.quantile(0.75)
        iqr_val = q3_out - q1_out
        lower_bound = q1_out - 1.5 * iqr_val
        upper_bound = q3_out + 1.5 * iqr_val
        
        iqr_outliers = df[(df[outlier_var] < lower_bound) | (df[outlier_var] > upper_bound)]

        # 2. Z-Score Method (|Z| > 3)
        z_scores = np.abs(stats.zscore(d_out))
        z_outliers = df[z_scores > 3.0]

        col_o1, col_o2 = st.columns(2)
        with col_o1:
            st.markdown("##### 📦 1.5 × IQR Rule")
            st.write(f"Lower Bound: **{lower_bound:.2f}** | Upper Bound: **{upper_bound:.2f}**")
            st.metric("IQR Outlier Count", len(iqr_outliers), delta=f"{len(iqr_outliers)/len(df)*100:.1f}% of data", delta_color="inverse")
        
        with col_o2:
            st.markdown("##### 📏 Z-Score Threshold (|Z| > 3)")
            st.write("Identifies values further than 3 standard deviations from mean.")
            st.metric("Z-Score Outlier Count", len(z_outliers), delta=f"{len(z_outliers)/len(df)*100:.1f}% of data", delta_color="inverse")

        fig_box_out = plot_box(df, x="student_department", y=outlier_var, title=f"Box Plot with Outlier Points: {outlier_var}")
        st.plotly_chart(fig_box_out, use_container_width=True)

        if not iqr_outliers.empty:
            st.markdown("##### Detected Outlier Transactions (IQR Method)")
            st.dataframe(
                iqr_outliers[["transaction_id", "student_name", "book_title", outlier_var, "overdue_status", "student_department"]].head(25),
                use_container_width=True
            )

    # TAB 4: Bivariate & Contingency Analysis
    with tab4:
        st.subheader("Bivariate Relationships & Cross-Tabulation")
        
        st.markdown("##### Cross-Tabulation: Book Category vs. Overdue Status")
        ctab = pd.crosstab(df["book_category"], df["overdue_status"], margins=True, margins_name="Total")
        ctab.columns = ["Returned On-Time", "Returned Overdue", "Total"]
        st.dataframe(ctab, use_container_width=True)

        # Department vs Overdue rate
        st.markdown("##### Overdue Proportion by Department")
        dept_grp = df.groupby("student_department").agg(
            Total_Transactions=("transaction_id", "count"),
            Overdue_Transactions=("overdue_status", "sum"),
            Avg_Overdue_Days=("overdue_days", "mean")
        ).reset_index()
        dept_grp["Overdue_Rate (%)"] = (dept_grp["Overdue_Transactions"] / dept_grp["Total_Transactions"] * 100).round(2)
        dept_grp["Avg_Overdue_Days"] = dept_grp["Avg_Overdue_Days"].round(2)
        st.dataframe(dept_grp, use_container_width=True)
