"""
Statistical analysis and spending insights module for SpendWise.

This module computes:
1. Core statistical metrics (mean, median, standard deviation, min, max, total).
2. Robust outlier thresholds using Median Absolute Deviation (MAD).
3. Category-level aggregations and spending percentages.
4. Month-level spending trends and period comparisons.
5. Plain-English analytical insights derived strictly from the active dataset.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def compute_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive summary statistics for the expense dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned expense DataFrame containing at least an 'amount' column.

    Returns
    -------
    Dict[str, Any]
        Dictionary of statistical figures and mathematical interpretations.
    """
    if df.empty:
        return {
            "total_spend": 0.0,
            "transaction_count": 0,
            "mean_amount": 0.0,
            "median_amount": 0.0,
            "std_amount": 0.0,
            "min_amount": 0.0,
            "max_amount": 0.0,
            "mean_median_ratio": 1.0,
            "is_right_skewed": False,
        }

    amounts = df["amount"].astype(float)
    count = len(amounts)
    total_spend = float(amounts.sum())
    mean_val = float(amounts.mean())
    median_val = float(amounts.median())

    # Standard deviation with ddof=1 (sample standard deviation)
    # If N <= 1, sample standard deviation is mathematically undefined (0.0 for safety)
    std_val = float(amounts.std(ddof=1)) if count > 1 else 0.0
    if np.isnan(std_val):
        std_val = 0.0

    min_val = float(amounts.min())
    max_val = float(amounts.max())

    # Skewness intuition: If Mean >> Median, a few large purchases heavily inflate the average
    ratio = (mean_val / median_val) if median_val > 0 else 1.0
    is_right_skewed = ratio > 1.25

    return {
        "total_spend": round(total_spend, 2),
        "transaction_count": count,
        "mean_amount": round(mean_val, 2),
        "median_amount": round(median_val, 2),
        "std_amount": round(std_val, 2),
        "min_amount": round(min_val, 2),
        "max_amount": round(max_val, 2),
        "mean_median_ratio": round(ratio, 2),
        "is_right_skewed": is_right_skewed,
    }


def compute_mad_threshold(df: pd.DataFrame, multiplier: float = 3.0) -> Dict[str, Any]:
    """
    Compute a robust statistical upper baseline using Median Absolute Deviation (MAD).

    Mathematical Foundation:
    ------------------------
    MAD = median(|x_i - median(X)|)
    For a normal distribution, standard deviation can be estimated consistently as:
        sigma_hat = 1.4826 * MAD
    The upper cutoff threshold is defined as:
        Threshold = Median + (multiplier * 1.4826 * MAD)

    Why MAD instead of Mean + 3*Std?
    - The standard deviation is squared deviations from the mean; one huge transaction
      (e.g., $145,000) inflates BOTH the mean and standard deviation, causing other
      unusual expenses to be masked (the masking effect).
    - The median and MAD have a 50% breakdown point, meaning up to 50% of the data can be
      outliers without corrupting the threshold.

    Edge Case Handling:
    - If MAD == 0 (more than half the transactions have identical amounts), we fall back
      to the 95th percentile or mean + 2*std to prevent threshold collapse.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned expense DataFrame.
    multiplier : float
        Tolerance multiplier (default is 3.0, representing ~3 sigma equivalent).

    Returns
    -------
    Dict[str, Any]
        Threshold value, MAD, fallback status, and flagged transaction count.
    """
    if df.empty:
        return {"threshold": 0.0, "mad": 0.0, "is_fallback": False, "flagged_count": 0}

    amounts = df["amount"].to_numpy(dtype=float)
    median_val = np.median(amounts)
    abs_deviations = np.abs(amounts - median_val)
    mad_val = float(np.median(abs_deviations))

    is_fallback = False
    if mad_val > 0:
        # 1.4826 is the normal distribution consistency scale factor
        normal_scale_mad = 1.4826 * mad_val
        threshold = median_val + (multiplier * normal_scale_mad)
    else:
        # Fallback when >50% of values are identical
        is_fallback = True
        std_val = float(np.std(amounts, ddof=1)) if len(amounts) > 1 else 0.0
        if std_val > 0:
            threshold = float(np.mean(amounts) + (multiplier * std_val))
        else:
            threshold = float(median_val * 1.5)

    flagged_mask = amounts > threshold
    flagged_count = int(np.sum(flagged_mask))

    return {
        "threshold": round(float(threshold), 2),
        "mad": round(float(mad_val), 2),
        "is_fallback": is_fallback,
        "flagged_count": flagged_count,
        "median": round(float(median_val), 2),
    }


def compute_category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group transactions by category and calculate total spend, percentage,
    transaction frequency, and average expense.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned expense DataFrame.

    Returns
    -------
    pd.DataFrame
        Aggregated category breakdown sorted descending by total spend.
    """
    if df.empty:
        return pd.DataFrame(columns=["category", "total", "percentage", "count", "average", "median"])

    total_spend = df["amount"].sum()

    grouped = (
        df.groupby("category")
        .agg(
            total=("amount", "sum"),
            count=("amount", "count"),
            average=("amount", "mean"),
            median=("amount", "median"),
        )
        .reset_index()
    )

    if total_spend > 0:
        grouped["percentage"] = (grouped["total"] / total_spend) * 100.0
    else:
        grouped["percentage"] = 0.0

    grouped["total"] = grouped["total"].round(2)
    grouped["percentage"] = grouped["percentage"].round(1)
    grouped["average"] = grouped["average"].round(2)
    grouped["median"] = grouped["median"].round(2)

    return grouped.sort_values(by="total", ascending=False).reset_index(drop=True)


def compute_monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate spending chronologically by month.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned expense DataFrame with 'month', 'month_name', and 'amount' columns.

    Returns
    -------
    pd.DataFrame
        Monthly summary with spend totals, transaction counts, and month-over-month % change.
    """
    if df.empty or "month" not in df.columns:
        return pd.DataFrame(columns=["month", "month_name", "total", "count", "average", "mom_change_pct"])

    monthly = (
        df.groupby(["month", "month_name"])
        .agg(
            total=("amount", "sum"),
            count=("amount", "count"),
            average=("amount", "mean"),
        )
        .reset_index()
    )

    # Sort strictly by chronological period string 'YYYY-MM'
    monthly = monthly.sort_values(by="month", ascending=True).reset_index(drop=True)

    monthly["total"] = monthly["total"].round(2)
    monthly["average"] = monthly["average"].round(2)

    # Month-over-Month percentage change
    monthly["mom_change_pct"] = monthly["total"].pct_change() * 100.0
    monthly["mom_change_pct"] = monthly["mom_change_pct"].round(1)

    return monthly


def generate_spending_insights(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Derive transparent, human-readable insights directly from the current dataset numbers.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned expense DataFrame.

    Returns
    -------
    List[Dict[str, str]]
        List of structured insight cards with titles, descriptions, and metric tags.
    """
    if df.empty:
        return []

    insights: List[Dict[str, str]] = []
    stats = compute_summary_statistics(df)
    cat_df = compute_category_breakdown(df)
    month_df = compute_monthly_trends(df)
    mad_info = compute_mad_threshold(df)

    # 1. Top Spending Category
    if not cat_df.empty:
        top_cat = cat_df.iloc[0]
        insights.append({
            "title": "Largest Spending Category",
            "badge": f"{top_cat['percentage']}% of Wallet",
            "description": (
                f"**{top_cat['category']}** is your highest expense area, accounting for "
                f"**${top_cat['total']:,.2f}** ({top_cat['percentage']}% of your total ${stats['total_spend']:,.2f} spend) "
                f"across **{int(top_cat['count'])}** transactions."
            ),
        })

    # 2. Most Frequent Spending Category
    if not cat_df.empty:
        most_frequent_cat = cat_df.sort_values(by="count", ascending=False).iloc[0]
        insights.append({
            "title": "Highest Frequency Category",
            "badge": f"{int(most_frequent_cat['count'])} transactions",
            "description": (
                f"You transact most frequently in **{most_frequent_cat['category']}** with "
                f"**{int(most_frequent_cat['count'])}** recorded expenses, averaging "
                f"**${most_frequent_cat['average']:,.2f}** per transaction."
            ),
        })

    # 3. Mean vs Median Robustness Observation
    if stats["is_right_skewed"]:
        insights.append({
            "title": "Skewness & Distribution Gap",
            "badge": f"Mean is {stats['mean_median_ratio']}x Median",
            "description": (
                f"Your average expense (**${stats['mean_amount']:,.2f}**) is significantly higher than your median "
                f"expense (**${stats['median_amount']:,.2f}**). This gap reveals a right-skewed distribution caused "
                f"by a few large outlier purchases (such as your maximum expense of **${stats['max_amount']:,.2f}**)."
            ),
        })
    else:
        insights.append({
            "title": "Balanced Spending Distribution",
            "badge": "Evenly Distributed",
            "description": (
                f"Your mean transaction (**${stats['mean_amount']:,.2f}**) and median transaction "
                f"(**${stats['median_amount']:,.2f}**) are closely aligned, showing consistent day-to-day spending."
            ),
        })

    # 4. Month-over-Month Trend Insight
    if len(month_df) >= 2:
        last_month = month_df.iloc[-1]
        prev_month = month_df.iloc[-2]
        change_pct = last_month["mom_change_pct"]

        direction = "increased" if change_pct > 0 else "decreased"
        insights.append({
            "title": "Recent Monthly Trajectory",
            "badge": f"{abs(change_pct):.1f}% {direction}",
            "description": (
                f"In **{last_month['month_name']}**, total spending was **${last_month['total']:,.2f}**, which "
                f"{direction} by **{abs(change_pct):.1f}%** compared to **{prev_month['month_name']}** "
                f"(${prev_month['total']:,.2f})."
            ),
        })

    # 5. Statistical MAD Baseline Flag
    insights.append({
        "title": "Statistical MAD Baseline",
        "badge": f"{mad_info['flagged_count']} High Expenses",
        "description": (
            f"Using Median Absolute Deviation, expenses above **${mad_info['threshold']:,.2f}** are flagged as "
            f"statistically elevated (Baseline: Median ${mad_info['median']:,.2f} + 3 × 1.4826 × MAD ${mad_info['mad']:,.2f}). "
            f"Currently, **{mad_info['flagged_count']}** transaction(s) exceed this statistical baseline."
        ),
    })

    return insights
