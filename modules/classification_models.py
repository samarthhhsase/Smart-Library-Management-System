"""
Experiment 7: Classification Models Comparison.
Compares Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machine (SVM),
Naive Bayes, and Random Forest for overdue-risk classification.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc

from database.db import get_master_dataset
from utils.helpers import render_lab_banner, download_csv_button
from utils.metrics import compute_classification_metrics, get_confusion_matrix_df
from utils.visualizations import plot_confusion_matrix, plot_bar, DARK_LAYOUT
import plotly.graph_objects as go

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def render_classification_models():
    render_lab_banner(
        experiment_num=7,
        title="Classification Models Comparison for Overdue Risk",
        description="Comprehensive benchmark of five foundational classification algorithms for detecting overdue transaction risk with class-imbalance mitigation and ROC-AUC profiling.",
        tags=["Supervised Classification", "Logistic Regression", "KNN", "SVM", "Naive Bayes", "Random Forest", "Recall Optimization"]
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

    tab1, tab2, tab3 = st.tabs([
        "🏆 Model Benchmark & Comparison",
        "🎛️ Hyperparameter Playground",
        "💡 Clinical Importance of Recall"
    ])

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feature_num),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), feature_cat)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # TAB 1: Benchmark & Comparison
    with tab1:
        st.subheader("Multi-Algorithm Classification Benchmark")

        # Train standard instances
        classifiers = {
            "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5, weights="distance"),
            "Support Vector Machine (SVM)": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
            "Gaussian Naive Bayes": GaussianNB(),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        }

        results = []
        roc_data = {}

        for name, clf in classifiers.items():
            pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
            pipe.fit(X_train, y_train)
            
            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)

            metrics = compute_classification_metrics(y_test, y_pred, y_prob)
            results.append({"Classifier": name, **metrics})

            # Save ROC Curve coords
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            roc_data[name] = (fpr, tpr, metrics["ROC_AUC"])

            # Save model
            safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            joblib.dump(pipe, os.path.join(MODELS_DIR, f"{safe_name}.joblib"))

        res_df = pd.DataFrame(results).sort_values(by="F1_Score", ascending=False)
        st.dataframe(res_df, use_container_width=True)

        # ROC Curve Plot
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Guess (AUC = 0.50)", line=dict(dash="dash", color="#64748b")))
        for name, (fpr, tpr, auc_val) in roc_data.items():
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC = {auc_val:.3f})"))
        fig_roc.update_layout(
            **DARK_LAYOUT,
            title="Receiver Operating Characteristic (ROC) Comparison",
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Recall / Sensitivity)"
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        # Bar comparison of F1 and Recall
        fig_bar = plot_bar(res_df, x="Classifier", y="Recall", title="Recall Comparison Across Classifiers", color="Classifier")
        st.plotly_chart(fig_bar, use_container_width=True)

    # TAB 2: Hyperparameter Playground
    with tab2:
        st.subheader("Tune Algorithm Parameters & Inspect Diagnostics")
        selected_clf_name = st.selectbox("Select Model to Tune", list(classifiers.keys()))

        if selected_clf_name == "K-Nearest Neighbors (KNN)":
            k_val = st.slider("Number of Neighbors (k)", min_value=1, max_value=25, value=5)
            weights_val = st.selectbox("Weighting Function", ["uniform", "distance"])
            chosen_clf = KNeighborsClassifier(n_neighbors=k_val, weights=weights_val)
        elif selected_clf_name == "Support Vector Machine (SVM)":
            c_val = st.slider("Regularization Parameter (C)", min_value=0.1, max_value=10.0, value=1.0)
            kernel_val = st.selectbox("Kernel Type", ["rbf", "linear", "poly"])
            chosen_clf = SVC(C=c_val, kernel=kernel_val, probability=True, class_weight="balanced", random_state=42)
        elif selected_clf_name == "Logistic Regression":
            c_log = st.slider("Inverse Regularization Strength (C)", min_value=0.01, max_value=10.0, value=1.0)
            chosen_clf = LogisticRegression(C=c_log, class_weight="balanced", max_iter=1000, random_state=42)
        elif selected_clf_name == "Random Forest Classifier":
            n_est = st.slider("Number of Estimators", min_value=10, max_value=200, value=100, step=10)
            chosen_clf = RandomForestClassifier(n_estimators=n_est, class_weight="balanced", random_state=42)
        else:
            chosen_clf = GaussianNB()

        interactive_pipe = Pipeline([("prep", preprocessor), ("clf", chosen_clf)])
        interactive_pipe.fit(X_train, y_train)
        i_pred = interactive_pipe.predict(X_test)
        i_prob = interactive_pipe.predict_proba(X_test)

        i_metrics = compute_classification_metrics(y_test, i_pred, i_prob)

        col_i1, col_i2, col_i3, col_i4 = st.columns(4)
        with col_i1:
            st.metric("Tuned Accuracy", f"{i_metrics['Accuracy']:.3f}")
        with col_i2:
            st.metric("Tuned Precision", f"{i_metrics['Precision']:.3f}")
        with col_i3:
            st.metric("Tuned Recall", f"{i_metrics['Recall']:.3f}")
        with col_i4:
            st.metric("Tuned F1-Score", f"{i_metrics['F1_Score']:.3f}")

        cm_interactive = get_confusion_matrix_df(y_test, i_pred)
        fig_cm_i = plot_confusion_matrix(cm_interactive, f"Confusion Matrix: {selected_clf_name}")
        st.plotly_chart(fig_cm_i, use_container_width=True)

    # TAB 3: Academic Discussion on Recall
    with tab3:
        st.subheader("Why Recall is Paramount in Library Overdue Prediction")
        st.markdown(r"""
        In standard machine learning applications, overall **Accuracy** can be misleading when classes are imbalanced (e.g. ~70% on-time vs ~30% overdue transactions). A trivial model predicting "always on-time" achieves 70% accuracy while failing 100% of the actual problem!

        #### Understanding the Cost Matrix:
        | Scenario | Model Prediction | Actual Reality | Real-World Consequence |
        |---|---|---|---|
        | **False Positive (Type I Error)** | Overdue Risk | Returns On-Time | Student receives a polite, friendly courtesy reminder email 2 days before the due date. Minimal harm done. |
        | **False Negative (Type II Error)** | On-Time | Becomes Overdue | Library system takes zero preventive action. Book becomes lost, students in waiting queue are deprived, and library assets sit dormant. |

        Therefore, our objective function must prioritize **Recall (Sensitivity)**:
        $$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$

        By employing `class_weight='balanced'` and setting decision thresholds strategically, we penalize False Negatives, ensuring that high-risk loans are caught early.
        """)
