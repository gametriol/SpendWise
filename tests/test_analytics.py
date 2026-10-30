"""
Unit tests for spendwise.analytics module.
"""

import pandas as pd
import numpy as np
import pytest
from spendwise.analytics import (
    compute_summary_statistics,
    compute_category_breakdown,
    compute_monthly_trends,
    compute_mad_threshold,
    generate_spending_insights,
)


@pytest.fixture
def sample_cleaned_df():
    data = {
        "date": pd.to_datetime(["2026-01-05", "2026-01-15", "2026-02-10", "2026-02-20"]),
        "month": ["2026-01", "2026-01", "2026-02", "2026-02"],
        "month_name": ["Jan 2026", "Jan 2026", "Feb 2026", "Feb 2026"],
        "description": ["Groceries", "Dinner", "Bus Pass", "Rent"],
        "category": ["Food", "Food", "Transport", "Housing"],
        "amount": [100.0, 200.0, 50.0, 1000.0],
    }
    return pd.DataFrame(data)


def test_compute_summary_statistics(sample_cleaned_df):
    stats = compute_summary_statistics(sample_cleaned_df)

    assert stats["transaction_count"] == 4
    assert stats["total_spend"] == 1350.0
    assert stats["mean_amount"] == 337.5
    assert stats["median_amount"] == 150.0
    assert stats["min_amount"] == 50.0
    assert stats["max_amount"] == 1000.0
    assert stats["is_right_skewed"] is True


def test_single_transaction_edge_case():
    single_df = pd.DataFrame({
        "amount": [500.0],
        "category": ["Food"],
    })
    stats = compute_summary_statistics(single_df)

    assert stats["transaction_count"] == 1
    assert stats["total_spend"] == 500.0
    assert stats["mean_amount"] == 500.0
    assert stats["median_amount"] == 500.0
    assert stats["std_amount"] == 0.0  # Safe handling of N=1


def test_category_breakdown_percentages_sum_to_100(sample_cleaned_df):
    cat_df = compute_category_breakdown(sample_cleaned_df)

    assert len(cat_df) == 3
    # Top category should be Housing ($1000)
    assert cat_df.iloc[0]["category"] == "Housing"
    assert cat_df.iloc[0]["total"] == 1000.0

    # Percentages should sum to 100% within reasonable rounding tolerance
    assert np.isclose(cat_df["percentage"].sum(), 100.0, atol=0.5)


def test_monthly_trends(sample_cleaned_df):
    month_df = compute_monthly_trends(sample_cleaned_df)

    assert len(month_df) == 2
    jan_row = month_df[month_df["month"] == "2026-01"].iloc[0]
    feb_row = month_df[month_df["month"] == "2026-02"].iloc[0]

    assert jan_row["total"] == 300.0
    assert feb_row["total"] == 1050.0
    # Month over month growth: (1050 - 300) / 300 = +250%
    assert feb_row["mom_change_pct"] == 250.0


def test_mad_threshold_and_constant_values():
    # Regular dataset
    df = pd.DataFrame({"amount": [10.0, 12.0, 11.0, 13.0, 10.0, 500.0]})
    mad_res = compute_mad_threshold(df)
    assert mad_res["threshold"] > 0
    assert mad_res["flagged_count"] == 1  # 500 is flagged

    # Constant values dataset (MAD = 0 edge case)
    constant_df = pd.DataFrame({"amount": [50.0, 50.0, 50.0, 50.0, 50.0]})
    const_res = compute_mad_threshold(constant_df)
    assert const_res["is_fallback"] is True
    assert const_res["mad"] == 0.0


def test_generate_spending_insights(sample_cleaned_df):
    insights = generate_spending_insights(sample_cleaned_df)

    assert len(insights) >= 3
    titles = [i["title"] for i in insights]
    assert "Largest Spending Category" in titles
    assert "Highest Frequency Category" in titles
