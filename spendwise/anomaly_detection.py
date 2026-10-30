"""
Unsupervised Anomaly Detection module for SpendWise using scikit-learn's Isolation Forest.

How Isolation Forest Works:
---------------------------
Isolation Forest is a tree-based, unsupervised machine learning algorithm designed
to isolate anomalies rather than profile normal data points.

1. Intuition:
   Anomalous transactions (e.g., a $145,000 laptop purchase or an accidental duplicate charge)
   are few in number and possess extreme feature values. Therefore, in random recursive binary
   partitions, they become isolated much closer to the root of a tree (shorter average path length)
   than normal, routine transactions which require many splits to separate.

2. Feature Engineering:
   We construct a transparent, 3-feature representation for each expense:
   - 'amount': The raw numerical transaction amount.
   - 'log_amount': np.log1p(amount), which compresses the long right-skewed tail.
   - 'category_amount_ratio': The transaction amount divided by its category median.
     This captures context: a $2,000 expense is normal for Housing (rent), but highly
     abnormal for Food & Dining (coffee/lunch).

3. scikit-learn IsolationForest API Details:
   - `model.predict(X)` returns:
       +1 for inliers (normal transactions)
       -1 for outliers (anomalous transactions)
   - `model.decision_function(X)` returns the anomaly score:
       * Positive scores represent normal inliers (deep in the trees).
       * Negative scores represent flagged anomalies (isolated near the tree root).
       * More negative = more abnormal.
"""

from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

RANDOM_STATE = 42
DEFAULT_CONTAMINATION = 0.05  # 5% expected anomaly rate


def build_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Construct a clean, normalized numerical feature matrix for Isolation Forest.

    Features:
    1. 'amount': Raw expense value.
    2. 'log_amount': Log-transformed amount via log(1 + x) for skew resilience.
    3. 'category_amount_ratio': Ratio of expense to the median of its specific category.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned expense DataFrame.

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        - Numerical feature matrix of shape (n_samples, n_features).
        - List of feature names.
    """
    feature_names = ["amount", "log_amount", "category_amount_ratio"]

    # 1. Raw amount
    amount_col = df["amount"].astype(float).to_numpy()

    # 2. Log-transformed amount
    log_amount_col = np.log1p(amount_col)

    # 3. Category-relative ratio
    # Calculate category medians
    category_medians = df.groupby("category")["amount"].transform("median").to_numpy(dtype=float)
    # Add small epsilon 1e-4 to prevent division by zero
    ratio_col = amount_col / (category_medians + 1e-4)

    # Stack into 2D array
    X = np.column_stack([amount_col, log_amount_col, ratio_col])
    return X, feature_names


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = DEFAULT_CONTAMINATION,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fit an Isolation Forest model and detect anomalous transactions.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned expense DataFrame.
    contamination : float
        Proportion of expected anomalies in the dataset (range: 0.01 to 0.15).
        Higher sensitivity flags more transactions.
    random_state : int
        Seed for pseudo-random number generator to guarantee reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        - Enriched DataFrame with columns:
            * 'is_anomaly': bool (True if flagged)
            * 'anomaly_label': 'Anomaly' vs 'Normal'
            * 'raw_score': scikit-learn decision_function output (negative = outlier)
            * 'severity_score': normalized 0-100 severity metric (higher = more extreme)
            * 'category_ratio': ratio of amount to category median
        - Summary metadata dictionary containing model diagnostics.
    """
    if df.empty:
        empty_df = df.copy()
        empty_df["is_anomaly"] = False
        empty_df["anomaly_label"] = "Normal"
        empty_df["raw_score"] = 0.0
        empty_df["severity_score"] = 0.0
        return empty_df, {
            "total_analyzed": 0,
            "anomaly_count": 0,
            "anomaly_rate_pct": 0.0,
            "contamination_used": contamination,
            "status": "empty_dataset",
        }

    # Ensure contamination is safely bounded between 0.01 and 0.20
    safe_contamination = float(np.clip(contamination, 0.01, 0.20))
    n_samples = len(df)

    # Edge case 1: Very small dataset (< 5 rows)
    if n_samples < 5:
        result_df = df.copy()
        result_df["is_anomaly"] = False
        result_df["anomaly_label"] = "Normal"
        result_df["raw_score"] = 0.0
        result_df["severity_score"] = 0.0
        result_df["category_ratio"] = 1.0
        return result_df, {
            "total_analyzed": n_samples,
            "anomaly_count": 0,
            "anomaly_rate_pct": 0.0,
            "contamination_used": safe_contamination,
            "status": "insufficient_data",
            "message": "At least 5 transactions are recommended for Isolation Forest modeling.",
        }

    # Edge case 2: Zero variance (all amounts identical)
    if df["amount"].nunique() == 1:
        result_df = df.copy()
        result_df["is_anomaly"] = False
        result_df["anomaly_label"] = "Normal"
        result_df["raw_score"] = 0.0
        result_df["severity_score"] = 0.0
        result_df["category_ratio"] = 1.0
        return result_df, {
            "total_analyzed": n_samples,
            "anomaly_count": 0,
            "anomaly_rate_pct": 0.0,
            "contamination_used": safe_contamination,
            "status": "constant_values",
            "message": "All expense amounts are identical; no anomalies exist.",
        }

    X, _ = build_features(df)

    # Initialize Isolation Forest with fixed seed and sensible hyperparameters
    # n_estimators=100 provides a robust ensemble average without sluggish latency
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=safe_contamination,
        random_state=random_state,
        n_jobs=-1,
    )

    # Fit model and predict
    # predict returns: 1 for inlier, -1 for outlier
    predictions = iso_forest.fit_predict(X)
    is_anomaly_mask = predictions == -1

    # decision_function returns anomaly scores (negative = outlier, positive = inlier)
    decision_scores = iso_forest.decision_function(X)

    # Convert raw decision scores into a human-friendly 0-100 severity scale
    # Invert scores so that more anomalous transactions have higher severity values
    inverted_scores = -decision_scores
    min_inv = inverted_scores.min()
    max_inv = inverted_scores.max()

    if max_inv > min_inv:
        normalized_severity = ((inverted_scores - min_inv) / (max_inv - min_inv)) * 100.0
    else:
        normalized_severity = np.zeros_like(inverted_scores)

    # Build enriched results DataFrame
    result_df = df.copy()
    result_df["is_anomaly"] = is_anomaly_mask
    result_df["anomaly_label"] = np.where(is_anomaly_mask, "Flagged Anomaly", "Normal Expense")
    result_df["raw_score"] = np.round(decision_scores, 4)
    result_df["severity_score"] = np.round(normalized_severity, 1)
    result_df["category_ratio"] = np.round(X[:, 2], 2)

    anomaly_count = int(is_anomaly_mask.sum())
    anomaly_rate_pct = round((anomaly_count / n_samples) * 100.0, 1)

    metadata: Dict[str, Any] = {
        "total_analyzed": n_samples,
        "anomaly_count": anomaly_count,
        "anomaly_rate_pct": anomaly_rate_pct,
        "contamination_used": safe_contamination,
        "random_state": random_state,
        "status": "success",
    }

    return result_df, metadata
