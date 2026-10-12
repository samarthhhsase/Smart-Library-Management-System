"""
Experiment 8: Decision Trees and Random Forest.
Implements Decision Tree and Random Forest classifiers, visualizes tree logic and feature importance,
and analyzes tree depth, bias-variance tradeoff, and ensemble generalization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from database.db import get_master_dataset
from utils.helpers import render_lab_banner
from utils.metrics import compute_classification_metrics, get_confusion_matrix_df
from utils.visualizations import plot_confusion_matrix, plot_feature_importance

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def render_tree_models():
    render_lab_banner(
        experiment_num=8,
        title="Decision Trees and Random Forest Classifiers",
        description="Interpret recursive decision boundaries, evaluate Gini impurity vs Entropy splits, compare single-tree overfitting with Random Forest bagging ensembles.",
        tags=["Tree-Based Learning", "Gini Impurity", "Entropy", "Ensemble Bagging", "Feature Importance", "Overfitting Analysis"]
    )

    master_df = get_master_dataset()
    if master_df.empty:
        st.warning("Master dataset not available.")
        return

    # Prepare features
    feature_num = ["borrowing_frequency", "student_year", "book_pub_year", "book_total_copies", "book_age", "issue_month", "issue_day_num"]
    feature_cat = ["book_category", "student_department"]

    clean_df = master_df.dropna(subset=feature_num + feature_cat + ["overdue_status"])
    X = clean_df[feature_num + feature_cat]
    y = clean_df["overdue_status"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feature_num),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), feature_cat)
        ]
    )

    tab1, tab2, tab3 = st.tabs([
        "🌳 Decision Tree Classifier",
        "🌲 Random Forest Ensemble",
        "⚖️ Tree vs. Forest Comparison"
    ])

    # TAB 1: Decision Tree
    with tab1:
        st.subheader("Decision Tree Hyperparameters & Structure")
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            tree_criterion = st.selectbox("Split Criterion", ["gini", "entropy"])
        with col_t2:
            max_depth_dt = st.slider("Max Tree Depth (None for unpruned)", min_value=2, max_value=12, value=4)
        with col_t3:
            min_samples_split_dt = st.slider("Min Samples Split", min_value=2, max_value=20, value=5)

        dt_clf = DecisionTreeClassifier(
            criterion=tree_criterion,
            max_depth=max_depth_dt,
            min_samples_split=min_samples_split_dt,
            class_weight="balanced",
            random_state=42
        )

        dt_pipe = Pipeline([("prep", preprocessor), ("dt", dt_clf)])
        dt_pipe.fit(X_train, y_train)

        # Performance
        train_pred_dt = dt_pipe.predict(X_train)
        test_pred_dt = dt_pipe.predict(X_test)
        test_prob_dt = dt_pipe.predict_proba(X_test)

        train_metrics_dt = compute_classification_metrics(y_train, train_pred_dt)
        test_metrics_dt = compute_classification_metrics(y_test, test_pred_dt, test_prob_dt)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Train Accuracy", f"{train_metrics_dt['Accuracy']:.3f}")
        with c2:
            st.metric("Test Accuracy", f"{test_metrics_dt['Accuracy']:.3f}", delta=f"{test_metrics_dt['Accuracy'] - train_metrics_dt['Accuracy']:.3f}")
        with c3:
            st.metric("Test Recall", f"{test_metrics_dt['Recall']:.3f}")
        with c4:
            st.metric("Test F1-Score", f"{test_metrics_dt['F1_Score']:.3f}")

        # Visualizing Tree Diagram
        st.markdown("#### Decision Tree Visual Graph (Top Nodes)")
        raw_feat_names = [f.split("__")[-1] for f in dt_pipe.named_steps["prep"].get_feature_names_out()]
        
        fig_tree, ax_tree = plt.subplots(figsize=(16, 7), dpi=100)
        plot_tree(
            dt_clf,
            feature_names=raw_feat_names,
            class_names=["On-Time", "Overdue"],
            filled=True,
            rounded=True,
            max_depth=3, # keep readable
            fontsize=8,
            ax=ax_tree
        )
        st.pyplot(fig_tree)
        plt.close(fig_tree)

        # Text Decision Rules Expander
        with st.expander("📜 View Text-Based Decision Rules"):
            tree_rules = export_text(dt_clf, feature_names=raw_feat_names, max_depth=4)
            st.code(tree_rules, language="text")

        joblib.dump(dt_pipe, os.path.join(MODELS_DIR, "decision_tree_risk.joblib"))

    # TAB 2: Random Forest
    with tab2:
        st.subheader("Random Forest Bagging Ensemble")
        col_rf1, col_rf2, col_rf3 = st.columns(3)
        with col_rf1:
            n_estimators_rf = st.slider("Number of Trees (n_estimators)", min_value=20, max_value=250, value=100, step=10)
        with col_rf2:
            max_depth_rf = st.slider("Max Depth per Tree", min_value=2, max_value=15, value=6)
        with col_rf3:
            rf_criterion = st.selectbox("Impurity Metric", ["gini", "entropy"], key="rf_crit")

        rf_clf = RandomForestClassifier(
            n_estimators=n_estimators_rf,
            max_depth=max_depth_rf,
            criterion=rf_criterion,
            class_weight="balanced",
            random_state=42
        )

        rf_pipe = Pipeline([("prep", preprocessor), ("rf", rf_clf)])
        rf_pipe.fit(X_train, y_train)

        train_pred_rf = rf_pipe.predict(X_train)
        test_pred_rf = rf_pipe.predict(X_test)
        test_prob_rf = rf_pipe.predict_proba(X_test)

        train_metrics_rf = compute_classification_metrics(y_train, train_pred_rf)
        test_metrics_rf = compute_classification_metrics(y_test, test_pred_rf, test_prob_rf)

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.metric("Train Accuracy", f"{train_metrics_rf['Accuracy']:.3f}")
        with col_r2:
            st.metric("Test Accuracy", f"{test_metrics_rf['Accuracy']:.3f}", delta=f"{test_metrics_rf['Accuracy'] - train_metrics_rf['Accuracy']:.3f}")
        with col_r3:
            st.metric("Test Recall", f"{test_metrics_rf['Recall']:.3f}")
        with col_r4:
            st.metric("Test ROC-AUC", f"{test_metrics_rf['ROC_AUC']:.3f}")

        # Feature Importance
        st.markdown("#### Random Forest Gini Feature Importance Ranking")
        feat_importances = rf_clf.feature_importances_
        fig_fi = plot_feature_importance(raw_feat_names, feat_importances, "Relative Feature Importance in Overdue Risk Detection")
        st.plotly_chart(fig_fi, use_container_width=True)

        joblib.dump(rf_pipe, os.path.join(MODELS_DIR, "random_forest_risk.joblib"))

    # TAB 3: Tree vs. Forest Comparison
    with tab3:
        st.subheader("Model Generalization: Decision Tree vs. Random Forest")
        
        comp_df = pd.DataFrame([
            {"Model": "Decision Tree (Single Estimator)", "Train Accuracy": train_metrics_dt["Accuracy"], "Test Accuracy": test_metrics_dt["Accuracy"], "Test Recall": test_metrics_dt["Recall"], "Test F1": test_metrics_dt["F1_Score"]},
            {"Model": "Random Forest (Bootstrap Aggregated Ensemble)", "Train Accuracy": train_metrics_rf["Accuracy"], "Test Accuracy": test_metrics_rf["Accuracy"], "Test Recall": test_metrics_rf["Recall"], "Test F1": test_metrics_rf["F1_Score"]}
        ])
        st.dataframe(comp_df, use_container_width=True)

        st.markdown(r"""
        #### Key Theoretical Insights for College Viva:
        1. **Why Single Trees Overfit:** A single decision tree with high depth creates rectangular partition cells that tightly enclose specific noise data points, causing high variance ($\text{Train Acc} \gg \text{Test Acc}$).
        2. **How Bagging (Bootstrap Aggregation) Resolves Overfitting:** Random Forest generates hundreds of distinct subsets through bootstrap sampling with replacement and selects a random subset of features at each split ($m = \sqrt{p}$).
        3. **Variance Reduction Formula:**
           $$\text{Var}(\bar{X}) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$$
           By decorrelating individual trees (lowering $\rho$), averaging predictions dramatically reduces variance without increasing bias.
        """)
