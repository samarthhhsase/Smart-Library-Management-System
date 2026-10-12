"""
Statistical and Machine Learning Evaluation Metrics Suite.
Provides rigorous calculations for Regression, Classification, Clustering, and Hypothesis Tests.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

def compute_regression_metrics(y_true, y_pred, p_features: int = 0) -> dict:
    """
    Computes standard regression evaluation metrics.
    Args:
        y_true: Ground truth target values
        y_pred: Predicted values
        p_features: Number of predictors (used for Adjusted R2)
    Returns:
        dict with MAE, MSE, RMSE, R2, Adjusted R2
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    n = len(y_true)
    if p_features > 0 and n > p_features + 1:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p_features - 1)
    else:
        adj_r2 = r2

    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "Adjusted_R2": round(adj_r2, 4)
    }

def compute_classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    Computes classification evaluation metrics including class-sensitive metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0

    roc_auc = None
    if y_prob is not None:
        try:
            # Check if y_prob has 2 classes or 1D array
            if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
                roc_auc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                roc_auc = roc_auc_score(y_true, y_prob)
            roc_auc = round(roc_auc, 4)
        except Exception:
            roc_auc = None

    return {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1_Score": round(f1, 4),
        "Specificity": round(specificity, 4),
        "ROC_AUC": roc_auc
    }

def get_confusion_matrix_df(y_true, y_pred, labels: list = ["On-Time", "Overdue"]) -> pd.DataFrame:
    """Returns a labeled confusion matrix DataFrame."""
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(
        cm,
        index=[f"Actual {l}" for l in labels],
        columns=[f"Pred {l}" for l in labels]
    )
    return df_cm

def compute_clustering_metrics(X, labels) -> dict:
    """
    Evaluates unsupervised cluster quality.
    """
    unique_labels = set(labels)
    # Check if we have between 2 and n-1 clusters
    if len(unique_labels) < 2 or len(unique_labels) >= len(X):
        return {
            "Silhouette_Score": None,
            "Davies_Bouldin_Index": None,
            "Calinski_Harabasz": None,
            "Num_Clusters": len(unique_labels)
        }
    
    # Exclude noise points (-1 in DBSCAN) if calculating silhouette
    valid_mask = labels != -1 if -1 in labels else np.ones(len(labels), dtype=bool)
    if np.sum(valid_mask) < 2 or len(set(labels[valid_mask])) < 2:
        return {
            "Silhouette_Score": None,
            "Davies_Bouldin_Index": None,
            "Calinski_Harabasz": None,
            "Num_Clusters": len(unique_labels)
        }

    try:
        sil = silhouette_score(X[valid_mask], labels[valid_mask])
        db = davies_bouldin_score(X[valid_mask], labels[valid_mask])
        ch = calinski_harabasz_score(X[valid_mask], labels[valid_mask])
        return {
            "Silhouette_Score": round(sil, 4),
            "Davies_Bouldin_Index": round(db, 4),
            "Calinski_Harabasz": round(ch, 2),
            "Num_Clusters": len(set(labels[valid_mask]))
        }
    except Exception:
        return {
            "Silhouette_Score": None,
            "Davies_Bouldin_Index": None,
            "Calinski_Harabasz": None,
            "Num_Clusters": len(unique_labels)
        }
