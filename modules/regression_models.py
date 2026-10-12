"""
Experiments 5 & 6: Regression Models.
Experiment 5: Linear Regression (Book Demand) & Logistic Regression (Overdue Risk probability).
Experiment 6: Ridge, Lasso, and Elastic Net Regularization comparison with Cross-Validation and Coefficient Shrinkage.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from database.db import get_master_dataset, get_all_books
from utils.helpers import render_lab_banner, download_csv_button
from utils.metrics import compute_regression_metrics, compute_classification_metrics, get_confusion_matrix_df
from utils.visualizations import (
    plot_regression_actual_vs_pred,
    plot_regression_residuals,
    plot_confusion_matrix,
    plot_bar
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def render_regression_models():
    st.title("📈 Regression Models & Regularization")
    st.caption("Experiments 5 & 6: Ordinary Least Squares (OLS), Logistic Classification, and Regularized Regression (Ridge, Lasso, Elastic Net).")

    tab1, tab2, tab3 = st.tabs([
        "🔬 Exp 5A: Linear Regression (Demand)",
        "🎯 Exp 5B: Logistic Regression (Risk)",
        "⚖️ Exp 6: Ridge, Lasso & Elastic Net"
    ])

    master_df = get_master_dataset()
    books_df = get_all_books()

    if master_df.empty or books_df.empty:
        st.warning("Data not available. Please seed database.")
        return

    # Prepare Book-Level Aggregation for Demand Forecasting
    book_demand_df = master_df.groupby("book_id").agg({
        "book_title": "first",
        "book_category": "first",
        "book_pub_year": "first",
        "book_total_copies": "first",
        "book_age": "first",
        "transaction_id": "count"
    }).reset_index().rename(columns={"transaction_id": "total_demand"})

    # TAB 1: Linear Regression (Demand Forecasting)
    with tab1:
        render_lab_banner(
            experiment_num=5,
            title="Experiment 5A: Linear Regression for Book Demand Forecasting",
            description="Predict future library borrowing volume per title using book publication age, inventory copies, and domain category.",
            tags=["Linear Regression", "OLS", "R² Score", "Residual Diagnostics", "Continuous Prediction"]
        )

        st.markdown("#### Model Specification & Features")
        st.write("Target ($y$): `total_demand` (Continuous count of historical borrows).")
        st.write("Predictors ($X$): `book_pub_year`, `book_total_copies`, `book_age`, `book_category`.")

        test_size_reg = st.slider("Test Set Split Ratio", min_value=0.15, max_value=0.35, value=0.20, step=0.05, key="test_reg_split")
        
        feature_cols_num = ["book_pub_year", "book_total_copies", "book_age"]
        feature_cols_cat = ["book_category"]

        X_book = book_demand_df[feature_cols_num + feature_cols_cat]
        y_book = book_demand_df["total_demand"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), feature_cols_num),
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), feature_cols_cat)
            ]
        )

        lin_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ])

        X_train, X_test, y_train, y_test = train_test_split(X_book, y_book, test_size=test_size_reg, random_state=42)
        lin_pipeline.fit(X_train, y_train)

        # Predictions
        y_pred_train = lin_pipeline.predict(X_train)
        y_pred_test = lin_pipeline.predict(X_test)

        metrics_test = compute_regression_metrics(y_test, y_pred_test, p_features=X_book.shape[1])

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("R² Score (Test)", f"{metrics_test['R2']:.4f}")
        with col_m2:
            st.metric("Adjusted R²", f"{metrics_test['Adjusted_R2']:.4f}")
        with col_m3:
            st.metric("MAE (Books)", f"{metrics_test['MAE']:.3f}")
        with col_m4:
            st.metric("RMSE", f"{metrics_test['RMSE']:.3f}")

        # Visualizations
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            fig_act_pred = plot_regression_actual_vs_pred(y_test, y_pred_test, "Linear Regression: Actual vs. Predicted Demand")
            st.plotly_chart(fig_act_pred, use_container_width=True)
        with col_v2:
            fig_res = plot_regression_residuals(y_test, y_pred_test, "Residuals Distribution (Actual - Predicted)")
            st.plotly_chart(fig_res, use_container_width=True)

        # Save model
        joblib.dump(lin_pipeline, os.path.join(MODELS_DIR, "linear_regression_demand.joblib"))

    # TAB 2: Logistic Regression (Overdue Risk)
    with tab2:
        render_lab_banner(
            experiment_num=5,
            title="Experiment 5B: Logistic Regression for Overdue Risk Probability",
            description="Predict the discrete probability of a transaction becoming overdue at point-of-issue, strictly preventing post-issue target leakage.",
            tags=["Binary Classification", "Logistic Sigmoid", "Log-Odds", "Odds Ratios", "Risk Scoring"]
        )

        st.markdown(r"""
        <div class="formula-card">
            <b>Logistic Model Formulation:</b><br>
            $$P(Y=1 \mid X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \dots + \beta_p X_p)}}$$
            Log-Odds / Logit: $$\ln\left(\frac{p}{1-p}\right) = X\beta$$
        </div>
        """, unsafe_allow_html=True)

        # Classification Dataset Preparation (Prevent Target Leakage)
        feature_num_cls = ["borrowing_frequency", "student_year", "book_pub_year", "book_total_copies", "book_age", "issue_month", "issue_day_num"]
        feature_cat_cls = ["book_category", "student_department"]

        cls_df = master_df.dropna(subset=feature_num_cls + feature_cat_cls + ["overdue_status"])
        X_cls = cls_df[feature_num_cls + feature_cat_cls]
        y_cls = cls_df["overdue_status"]

        X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_cls, y_cls, test_size=0.25, random_state=42, stratify=y_cls)

        preprocessor_cls = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), feature_num_cls),
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), feature_cat_cls)
            ]
        )

        log_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor_cls),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
        ])

        log_pipeline.fit(X_tr_c, y_tr_c)
        y_pred_cls = log_pipeline.predict(X_te_c)
        y_prob_cls = log_pipeline.predict_proba(X_te_c)

        cls_metrics = compute_classification_metrics(y_te_c, y_pred_cls, y_prob_cls)

        c_l1, c_l2, c_l3, c_l4 = st.columns(4)
        with c_l1:
            st.metric("Accuracy", f"{cls_metrics['Accuracy']:.3f}")
        with c_l2:
            st.metric("Precision", f"{cls_metrics['Precision']:.3f}")
        with c_l3:
            st.metric("Recall (Sensitivity)", f"{cls_metrics['Recall']:.3f}")
        with c_l4:
            st.metric("ROC-AUC", f"{cls_metrics['ROC_AUC']:.3f}")

        col_c_v1, col_c_v2 = st.columns(2)
        with col_c_v1:
            cm_df = get_confusion_matrix_df(y_te_c, y_pred_cls)
            fig_cm = plot_confusion_matrix(cm_df, "Logistic Regression Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)
        with col_c_v2:
            # Feature Log-Odds Coefficients
            regressor = log_pipeline.named_steps["classifier"]
            prep = log_pipeline.named_steps["preprocessor"]
            feature_names = prep.get_feature_names_out()
            coefs = regressor.coef_[0]
            coef_df = pd.DataFrame({"Feature": [f.split("__")[-1] for f in feature_names], "Coefficient (Log-Odds)": coefs.round(3)})
            coef_df["Odds Ratio (e^β)"] = np.exp(coef_df["Coefficient (Log-Odds)"]).round(3)
            coef_df = coef_df.sort_values(by="Coefficient (Log-Odds)", key=abs, ascending=False)
            st.markdown("##### Top Predictive Feature Weights (Odds Ratios)")
            st.dataframe(coef_df.head(8), use_container_width=True)

        joblib.dump(log_pipeline, os.path.join(MODELS_DIR, "logistic_regression_risk.joblib"))

    # TAB 3: Ridge, Lasso, and Elastic Net (Experiment 6)
    with tab3:
        render_lab_banner(
            experiment_num=6,
            title="Experiment 6: Ridge, Lasso and Elastic Net Regularization",
            description="Regularization techniques penalize complex models to prevent overfitting and multicollinearity through L1 and L2 penalty norms.",
            tags=["L2 Ridge", "L1 Lasso", "Elastic Net", "Cross-Validation", "Coefficient Shrinkage", "Feature Sparsity"]
        )

        st.markdown(r"""
        <div class="formula-card">
            <b>Regularization Loss Formulations:</b><br>
            • <b>Ridge (L2 penalty):</b> $$\min_\beta \left\{ \sum (y_i - \hat{y}_i)^2 + \alpha \sum \beta_j^2 \right\}$$ (Shrinks coefficients asymptotically toward zero)<br>
            • <b>Lasso (L1 penalty):</b> $$\min_\beta \left\{ \sum (y_i - \hat{y}_i)^2 + \alpha \sum |\beta_j| \right\}$$ (Forces coefficients strictly to 0; automatic feature selection)<br>
            • <b>Elastic Net (L1 + L2):</b> $$\min_\beta \left\{ \sum (y_i - \hat{y}_i)^2 + \alpha \left[ \rho \sum |\beta_j| + \frac{1-\rho}{2} \sum \beta_j^2 \right] \right\}$$
        </div>
        """, unsafe_allow_html=True)

        cv_folds = st.slider("Cross-Validation Folds (k-fold CV)", min_value=3, max_value=10, value=5)

        # Build regularized pipelines
        alphas = np.logspace(-3, 3, 50)
        ridge_model = RidgeCV(alphas=alphas, cv=cv_folds)
        lasso_model = LassoCV(alphas=alphas, cv=cv_folds, max_iter=5000, random_state=42)
        elastic_model = ElasticNetCV(alphas=alphas, l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95], cv=cv_folds, max_iter=5000, random_state=42)

        pipe_ridge = Pipeline([("prep", preprocessor), ("reg", ridge_model)])
        pipe_lasso = Pipeline([("prep", preprocessor), ("reg", lasso_model)])
        pipe_elastic = Pipeline([("prep", preprocessor), ("reg", elastic_model)])

        pipe_ridge.fit(X_train, y_train)
        pipe_lasso.fit(X_train, y_train)
        pipe_elastic.fit(X_train, y_train)

        # Predictions
        pred_ols = lin_pipeline.predict(X_test)
        pred_ridge = pipe_ridge.predict(X_test)
        pred_lasso = pipe_lasso.predict(X_test)
        pred_elastic = pipe_elastic.predict(X_test)

        m_ols = compute_regression_metrics(y_test, pred_ols)
        m_ridge = compute_regression_metrics(y_test, pred_ridge)
        m_lasso = compute_regression_metrics(y_test, pred_lasso)
        m_elastic = compute_regression_metrics(y_test, pred_elastic)

        st.markdown("#### Side-by-Side Model Comparison Matrix")
        comparison_table = pd.DataFrame([
            {"Model": "Ordinary Least Squares (OLS)", "Best Alpha (α)": "N/A", **m_ols},
            {"Model": "Ridge Regression (L2)", "Best Alpha (α)": round(pipe_ridge.named_steps["reg"].alpha_, 4), **m_ridge},
            {"Model": "Lasso Regression (L1)", "Best Alpha (α)": round(pipe_lasso.named_steps["reg"].alpha_, 4), **m_lasso},
            {"Model": "Elastic Net (L1 + L2)", "Best Alpha (α)": round(pipe_elastic.named_steps["reg"].alpha_, 4), **m_elastic}
        ])
        st.dataframe(comparison_table, use_container_width=True)

        # Coefficient Shrinkage Visualization
        raw_feat_names = [f.split("__")[-1] for f in pipe_ridge.named_steps["prep"].get_feature_names_out()]
        coef_comparison = pd.DataFrame({
            "Feature": raw_feat_names,
            "OLS (No Penalty)": lin_pipeline.named_steps["regressor"].coef_,
            "Ridge (L2 Shrinkage)": pipe_ridge.named_steps["reg"].coef_,
            "Lasso (L1 Sparsity)": pipe_lasso.named_steps["reg"].coef_,
            "Elastic Net": pipe_elastic.named_steps["reg"].coef_
        })

        st.markdown("#### Regularization Coefficient Shrinkage Comparison")
        st.dataframe(coef_comparison.round(4), use_container_width=True)

        # Plot bar comparison
        coef_melted = coef_comparison.melt(id_vars=["Feature"], var_name="Algorithm", value_name="Weight Coefficient")
        fig_coef = plot_bar(coef_melted, x="Feature", y="Weight Coefficient", title="Coefficient Shrinkage across Regularization Techniques", color="Algorithm")
        st.plotly_chart(fig_coef, use_container_width=True)

        st.info("💡 **Academic Note:** Notice how Lasso sets several less-informative coefficients to exactly **0.000**, effectively acting as embedded feature selection, whereas Ridge shrinks coefficients smoothly toward zero without eliminating them.")

        joblib.dump(pipe_ridge, os.path.join(MODELS_DIR, "ridge_demand.joblib"))
        joblib.dump(pipe_lasso, os.path.join(MODELS_DIR, "lasso_demand.joblib"))
        joblib.dump(pipe_elastic, os.path.join(MODELS_DIR, "elastic_net_demand.joblib"))
