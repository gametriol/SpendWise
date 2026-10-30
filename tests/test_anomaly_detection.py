"""
Unit tests for spendwise.anomaly_detection module.
"""

import numpy as np
import pandas as pd
import pytest
from spendwise.anomaly_detection import detect_anomalies, build_features, RANDOM_STATE


@pytest.fixture
def synthetic_expense_df():
    # 50 normal transactions between 50 and 500, plus 2 extreme outliers (15000, 25000)
    np.random.seed(42)
    normal_amounts = np.random.uniform(50, 500, size=50).tolist()
    outlier_amounts = [15000.0, 25000.0]
    all_amounts = normal_amounts + outlier_amounts

    categories = (["Food"] * 25) + (["Transport"] * 25) + (["Shopping"] * 2)

    df = pd.DataFrame({
        "amount": all_amounts,
        "category": categories,
        "description": [f"Item {i}" for i in range(len(all_amounts))],
        "date": pd.date_range("2026-01-01", periods=len(all_amounts), freq="D"),
    })
    return df


def test_build_features(synthetic_expense_df):
    X, feature_names = build_features(synthetic_expense_df)

    assert X.shape == (len(synthetic_expense_df), 3)
    assert feature_names == ["amount", "log_amount", "category_amount_ratio"]
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()


def test_detect_anomalies_flags_outliers(synthetic_expense_df):
    result_df, metadata = detect_anomalies(synthetic_expense_df, contamination=0.05)

    assert metadata["status"] == "success"
    assert metadata["total_analyzed"] == len(synthetic_expense_df)
    assert metadata["anomaly_count"] > 0

    assert "is_anomaly" in result_df.columns
    assert "severity_score" in result_df.columns
    assert "raw_score" in result_df.columns

    # The 2 massive outliers (15000 and 25000) should be flagged with high severity
    top_outlier = result_df[result_df["amount"] == 25000.0].iloc[0]
    assert top_outlier["is_anomaly"] == True
    assert top_outlier["severity_score"] > 80.0


def test_detect_anomalies_reproducibility(synthetic_expense_df):
    # Running twice with the same fixed seed must yield identical results
    res1, _ = detect_anomalies(synthetic_expense_df, contamination=0.05, random_state=RANDOM_STATE)
    res2, _ = detect_anomalies(synthetic_expense_df, contamination=0.05, random_state=RANDOM_STATE)

    pd.testing.assert_series_equal(res1["is_anomaly"], res2["is_anomaly"])
    pd.testing.assert_series_equal(res1["severity_score"], res2["severity_score"])


def test_detect_anomalies_sensitivity_effect(synthetic_expense_df):
    # Higher contamination (sensitivity) should flag at least as many or more items
    _, meta_low = detect_anomalies(synthetic_expense_df, contamination=0.02)
    _, meta_high = detect_anomalies(synthetic_expense_df, contamination=0.10)

    assert meta_high["anomaly_count"] >= meta_low["anomaly_count"]


def test_small_dataset_edge_case():
    small_df = pd.DataFrame({
        "amount": [10.0, 20.0, 30.0],
        "category": ["Food", "Food", "Food"],
        "description": ["A", "B", "C"],
    })
    result_df, metadata = detect_anomalies(small_df)

    assert metadata["status"] == "insufficient_data"
    assert metadata["anomaly_count"] == 0
    assert not result_df["is_anomaly"].any()


def test_constant_values_edge_case():
    constant_df = pd.DataFrame({
        "amount": [100.0] * 20,
        "category": ["Food"] * 20,
        "description": ["Item"] * 20,
    })
    result_df, metadata = detect_anomalies(constant_df)

    assert metadata["status"] == "constant_values"
    assert metadata["anomaly_count"] == 0
    assert not result_df["is_anomaly"].any()
