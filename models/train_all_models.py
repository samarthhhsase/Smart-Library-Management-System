"""
Pre-training and Model Persistence Script for SLMS.
Trains and persists all baseline ML models into models/ directory.
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from database.db import get_master_dataset

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_persist_all():
    df = get_master_dataset()
    if df.empty:
        print("Master dataset is empty. Cannot train models.")
        return

    print("Training Regression Models (Book Demand)...")
    book_demand_df = df.groupby("book_id").agg({
        "book_category": "first",
        "book_pub_year": "first",
        "book_total_copies": "first",
        "book_age": "first",
        "transaction_id": "count"
    }).reset_index().rename(columns={"transaction_id": "total_demand"})

    feat_num_reg = ["book_pub_year", "book_total_copies", "book_age"]
    feat_cat_reg = ["book_category"]

    prep_reg = ColumnTransformer([
        ("num", StandardScaler(), feat_num_reg),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), feat_cat_reg)
    ])

    X_reg = book_demand_df[feat_num_reg + feat_cat_reg]
    y_reg = book_demand_df["total_demand"]

    pipe_lin = Pipeline([("prep", prep_reg), ("reg", LinearRegression())])
    pipe_ridge = Pipeline([("prep", prep_reg), ("reg", RidgeCV(cv=5))])
    pipe_lasso = Pipeline([("prep", prep_reg), ("reg", LassoCV(cv=5, max_iter=5000, random_state=42))])
    pipe_elastic = Pipeline([("prep", prep_reg), ("reg", ElasticNetCV(cv=5, max_iter=5000, random_state=42))])

    pipe_lin.fit(X_reg, y_reg)
    pipe_ridge.fit(X_reg, y_reg)
    pipe_lasso.fit(X_reg, y_reg)
    pipe_elastic.fit(X_reg, y_reg)

    joblib.dump(pipe_lin, os.path.join(MODELS_DIR, "linear_regression_demand.joblib"))
    joblib.dump(pipe_ridge, os.path.join(MODELS_DIR, "ridge_demand.joblib"))
    joblib.dump(pipe_lasso, os.path.join(MODELS_DIR, "lasso_demand.joblib"))
    joblib.dump(pipe_elastic, os.path.join(MODELS_DIR, "elastic_net_demand.joblib"))

    print("Training Classification Models (Overdue Risk)...")
    feat_num_cls = ["borrowing_frequency", "student_year", "book_pub_year", "book_total_copies", "book_age", "issue_month", "issue_day_num"]
    feat_cat_cls = ["book_category", "student_department"]

    clean_cls = df.dropna(subset=feat_num_cls + feat_cat_cls + ["overdue_status"])
    X_cls = clean_cls[feat_num_cls + feat_cat_cls]
    y_cls = clean_cls["overdue_status"]

    prep_cls = ColumnTransformer([
        ("num", StandardScaler(), feat_num_cls),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), feat_cat_cls)
    ])

    clfs = {
        "logistic_regression_risk": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "random_forest_risk": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
        "decision_tree_risk": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
        "k-nearest_neighbors_knn": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "support_vector_machine_svm": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
        "gaussian_naive_bayes": GaussianNB()
    }

    for fname, clf in clfs.items():
        pipe = Pipeline([("prep", prep_cls), ("clf", clf)])
        pipe.fit(X_cls, y_cls)
        joblib.dump(pipe, os.path.join(MODELS_DIR, f"{fname}.joblib"))

    print("All models successfully trained and persisted with Joblib!")

if __name__ == "__main__":
    train_and_persist_all()
