"""
Experiment 4: Hypothesis Testing with Real-World Data.
Implements Two-Sample Independent t-test, Chi-square Test of Independence,
and One-Way ANOVA with complete hypothesis formulations, p-value decisions, and conclusions.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from database.db import get_master_dataset
from utils.helpers import render_lab_banner
from utils.visualizations import plot_box, plot_bar

def render_hypothesis_testing():
    render_lab_banner(
        experiment_num=4,
        title="Hypothesis Testing with Real-World Data",
        description="Formulate and evaluate statistical hypotheses on student borrowing behaviour using Student's t-test, Chi-Square independence test, and One-Way ANOVA.",
        tags=["Inferential Statistics", "Independent t-Test", "Chi-Square Test", "ANOVA", "p-Value Interpretation"]
    )

    df = get_master_dataset()
    if df.empty:
        st.warning("Master dataset not available.")
        return

    # Global significance level selector
    col_alpha, _ = st.columns([1, 3])
    with col_alpha:
        alpha = st.select_slider("Select Significance Level (α)", options=[0.01, 0.05, 0.10], value=0.05)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Test 1: Two-Sample t-Test",
        "📊 Test 2: Chi-Square Test",
        "📈 Test 3: One-Way ANOVA",
        "🧪 Test 4: Normality Diagnostic"
    ])

    # TAB 1: Two-Sample t-Test
    with tab1:
        st.subheader("Independent Two-Sample Student's t-Test")
        st.markdown("""
        **Objective:** Determine whether there is a statistically significant difference between the average overdue days of two distinct student cohorts.
        """)

        test_group_type = st.radio("Select Comparison Basis", ["By Department", "By Academic Year"], horizontal=True)

        if test_group_type == "By Department":
            depts = df["student_department"].unique().tolist()
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                grp1_name = st.selectbox("Group 1 (Department)", depts, index=0)
            with col_g2:
                grp2_name = st.selectbox("Group 2 (Department)", depts, index=min(1, len(depts)-1))

            s1 = df[df["student_department"] == grp1_name]["overdue_days"].dropna()
            s2 = df[df["student_department"] == grp2_name]["overdue_days"].dropna()
        else:
            years = sorted(df["student_year"].unique().tolist())
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                y1 = st.selectbox("Group 1 (Year)", years, index=0)
                grp1_name = f"Year {y1}"
            with col_g2:
                y2 = st.selectbox("Group 2 (Year)", years, index=min(len(years)-1, 3))
                grp2_name = f"Year {y2}"

            s1 = df[df["student_year"] == y1]["overdue_days"].dropna()
            s2 = df[df["student_year"] == y2]["overdue_days"].dropna()

        # Display Hypotheses
        st.markdown(f"""
        <div class="formula-card">
            <b>Null Hypothesis (H₀):</b> μ₁ = μ₂ (There is no significant difference in mean overdue days between {grp1_name} and {grp2_name})<br>
            <b>Alternative Hypothesis (H₁):</b> μ₁ ≠ μ₂ (There is a significant difference in mean overdue days between {grp1_name} and {grp2_name})
        </div>
        """, unsafe_allow_html=True)

        if len(s1) < 2 or len(s2) < 2:
            st.error("Insufficient sample size to perform t-test.")
        else:
            # Levene's test for equality of variance
            lev_stat, lev_p = stats.levene(s1, s2)
            equal_var = lev_p > 0.05

            t_stat, p_val = stats.ttest_ind(s1, s2, equal_var=equal_var)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric(f"Mean ({grp1_name})", f"{s1.mean():.2f}", f"n = {len(s1)}")
            with c2:
                st.metric(f"Mean ({grp2_name})", f"{s2.mean():.2f}", f"n = {len(s2)}")
            with c3:
                st.metric("t-Statistic", f"{t_stat:.4f}")
            with c4:
                st.metric("p-Value", f"{p_val:.4e}")

            # Conclusion
            if p_val <= alpha:
                st.error(f"🚨 **Decision: Reject H₀ at α = {alpha}** (p-value {p_val:.4f} ≤ {alpha}). There IS a statistically significant difference in mean overdue days between {grp1_name} and {grp2_name}.")
            else:
                st.success(f"✅ **Decision: Fail to Reject H₀ at α = {alpha}** (p-value {p_val:.4f} > {alpha}). There is NO statistically significant difference in mean overdue days between {grp1_name} and {grp2_name}.")

            # Visualization
            plot_df = df[df["student_department"].isin([grp1_name, grp2_name])] if test_group_type == "By Department" else df[df["student_year"].isin([y1, y2])]
            x_col = "student_department" if test_group_type == "By Department" else "student_year"
            fig_t = plot_box(plot_df, x=x_col, y="overdue_days", title=f"Overdue Days Comparison: {grp1_name} vs. {grp2_name}")
            st.plotly_chart(fig_t, use_container_width=True)

    # TAB 2: Chi-Square Test
    with tab2:
        st.subheader("Chi-Square Test of Independence (χ²)")
        st.markdown("""
        **Objective:** Test whether book category is statistically independent of overdue status (On-Time vs. Overdue).
        """)

        st.markdown(f"""
        <div class="formula-card">
            <b>Null Hypothesis (H₀):</b> Book category and overdue status are independent (No association).<br>
            <b>Alternative Hypothesis (H₁):</b> Book category and overdue status are dependent (Categorical association exists).
        </div>
        """, unsafe_allow_html=True)

        contingency = pd.crosstab(df["book_category"], df["overdue_status"])
        contingency.columns = ["On-Time", "Overdue"]

        chi2_stat, p_chi, dof, expected = stats.chi2_contingency(contingency)

        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            st.metric("Chi-Square (χ²)", f"{chi2_stat:.4f}")
        with col_c2:
            st.metric("Degrees of Freedom (df)", dof)
        with col_c3:
            st.metric("p-Value", f"{p_chi:.4e}")
        with col_c4:
            # Cramer's V effect size
            n_tot = contingency.sum().sum()
            min_dim = min(contingency.shape) - 1
            cramer_v = np.sqrt(chi2_stat / (n_tot * min_dim)) if min_dim > 0 else 0
            st.metric("Cramér's V (Effect Size)", f"{cramer_v:.4f}")

        if p_chi <= alpha:
            st.error(f"🚨 **Decision: Reject H₀ at α = {alpha}** (p-value {p_chi:.4f} ≤ {alpha}). A statistically significant association exists between book category and overdue likelihood.")
        else:
            st.success(f"✅ **Decision: Fail to Reject H₀ at α = {alpha}** (p-value {p_chi:.4f} > {alpha}). Book category and overdue status appear statistically independent.")

        st.markdown("#### Contingency Table: Observed vs. Expected Frequencies")
        exp_df = pd.DataFrame(expected.round(1), index=contingency.index, columns=["Expected On-Time", "Expected Overdue"])
        merged_table = pd.concat([contingency, exp_df], axis=1)
        st.dataframe(merged_table, use_container_width=True)

    # TAB 3: One-Way ANOVA
    with tab3:
        st.subheader("One-Way Analysis of Variance (ANOVA)")
        st.markdown("""
        **Objective:** Compare the variance in borrowing frequency or loan duration across 3 or more book categories to test if at least one category mean differs.
        """)

        anova_target = st.selectbox("Select Target Variable for ANOVA", ["borrowing_frequency", "loan_duration_days", "overdue_days"])

        st.markdown(f"""
        <div class="formula-card">
            <b>Null Hypothesis (H₀):</b> μ₁ = μ₂ = ... = μₖ (All category population means are equal for {anova_target})<br>
            <b>Alternative Hypothesis (H₁):</b> At least one category population mean differs.
        </div>
        """, unsafe_allow_html=True)

        cats = df["book_category"].unique()
        groups = [df[df["book_category"] == c][anova_target].dropna() for c in cats]

        f_stat, p_anova = stats.f_oneway(*groups)

        c_a1, c_a2, c_a3 = st.columns(3)
        with c_a1:
            st.metric("F-Statistic", f"{f_stat:.4f}")
        with c_a2:
            st.metric("p-Value", f"{p_anova:.4e}")
        with c_a3:
            st.metric("Number of Groups (k)", len(cats))

        if p_anova <= alpha:
            st.error(f"🚨 **Decision: Reject H₀ at α = {alpha}** (p-value {p_anova:.4f} ≤ {alpha}). There is a statistically significant difference in {anova_target} across different book categories.")
        else:
            st.success(f"✅ **Decision: Fail to Reject H₀ at α = {alpha}** (p-value {p_anova:.4f} > {alpha}). There is no statistically significant difference in {anova_target} among book categories.")

        fig_anova = plot_box(df, x="book_category", y=anova_target, title=f"Distribution of {anova_target} by Book Category")
        st.plotly_chart(fig_anova, use_container_width=True)

    # TAB 4: Normality Diagnostic
    with tab4:
        st.subheader("Parametric Assumptions: Normality Testing")
        st.markdown("""
        Standard parametric tests (like Student's t-test and ANOVA) assume data is normally distributed.
        We perform the **Shapiro-Wilk test** and **D'Agostino-Pearson test** to verify normality.
        """)

        norm_var = st.selectbox("Select Variable for Normality Evaluation", ["overdue_days", "loan_duration_days", "borrowing_frequency"], key="norm_eval_var")
        data_norm = df[norm_var].dropna()

        # Shapiro-Wilk (take sample if n > 5000)
        sample_for_shapiro = data_norm.sample(min(len(data_norm), 500), random_state=42)
        shapiro_stat, shapiro_p = stats.shapiro(sample_for_shapiro)
        dag_stat, dag_p = stats.normaltest(data_norm)

        cn1, cn2 = st.columns(2)
        with cn1:
            st.markdown("##### Shapiro-Wilk Test")
            st.metric("Test Statistic (W)", f"{shapiro_stat:.4f}")
            st.metric("p-value", f"{shapiro_p:.4e}")
            if shapiro_p < 0.05:
                st.warning("⚠️ Rejects normality (p < 0.05). Distribution deviates from standard bell curve.")
            else:
                st.success("✅ Consistent with a normal distribution (p ≥ 0.05).")

        with cn2:
            st.markdown("##### D'Agostino & Pearson Omninormality Test")
            st.metric("Omnibus Statistic (s² + k²)", f"{dag_stat:.4f}")
            st.metric("p-value", f"{dag_p:.4e}")
            if dag_p < 0.05:
                st.warning("⚠️ Significant skewness/kurtosis detected.")
            else:
                st.success("✅ Normal distribution assumption holds.")
