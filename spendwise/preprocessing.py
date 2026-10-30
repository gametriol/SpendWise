"""
Data validation and preprocessing module for SpendWise.

This module is responsible for:
1. Loading expense CSV files safely (from a file path, buffer, or DataFrame).
2. Verifying required column schema: date, description, category, amount.
3. Cleaning and normalizing string values.
4. Safely parsing dates into datetime objects.
5. Converting amounts into numeric values and enforcing the positive expense policy.
6. Generating a transparent validation report so users know exactly why any row was rejected.
"""

from typing import Union, Dict, Any, Tuple
import io
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = ["date", "description", "category", "amount"]
DEFAULT_DESCRIPTION = "Uncategorized Expense"
DEFAULT_CATEGORY = "Other"


class ValidationError(Exception):
    """Custom exception raised when data validation fails irreparably."""
    pass


def load_raw_data(source: Union[str, io.BytesIO, io.StringIO, pd.DataFrame]) -> pd.DataFrame:
    """
    Safely load raw expense data from a file path, file buffer, or existing DataFrame.

    Parameters
    ----------
    source : str, BytesIO, StringIO, or pd.DataFrame
        The data source to read.

    Returns
    -------
    pd.DataFrame
        The raw DataFrame loaded from the source.

    Raises
    ------
    ValidationError
        If the file is empty, cannot be parsed as CSV, or contains no rows.
    """
    if isinstance(source, pd.DataFrame):
        return source.copy()

    try:
        df = pd.read_csv(source)
    except pd.errors.EmptyDataError:
        raise ValidationError("The provided CSV file is completely empty. Please upload a file with data.")
    except Exception as e:
        raise ValidationError(f"Could not read CSV file: {str(e)}")

    if df.empty:
        raise ValidationError("The provided CSV file contains column headers but zero data rows.")

    return df


def validate_schema(df: pd.DataFrame) -> None:
    """
    Ensure all required columns are present in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The raw DataFrame to check.

    Raises
    ------
    ValidationError
        If one or more required columns ('date', 'description', 'category', 'amount') are missing.
    """
    # Standardize column names to lower case and stripped whitespace for resilient matching
    clean_columns = [str(c).strip().lower() for c in df.columns]
    df.columns = clean_columns

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in clean_columns]
    if missing_cols:
        raise ValidationError(
            f"Missing required column(s): {', '.join(missing_cols)}. "
            f"Your CSV must have these columns: {', '.join(REQUIRED_COLUMNS)}."
        )


def clean_expenses_data(source: Union[str, io.BytesIO, io.StringIO, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute the end-to-end data validation and cleaning pipeline.

    Cleaning Rules & Policies:
    --------------------------
    1. Schema: Requires ['date', 'description', 'category', 'amount'].
    2. Date parsing: Parsed using pandas to_datetime. Rows with unparseable or missing
       dates are rejected.
    3. Amounts: Converted to numeric. Expenses must be strictly positive (> 0) and finite.
       Zero, negative, NaN, or infinite amounts are rejected.
    4. Missing Descriptions: Imputed with 'Uncategorized Expense'.
    5. Missing Categories: Imputed with 'Other'.
    6. Sorting: Clean data is sorted chronologically by date.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        - Cleaned and enriched DataFrame.
        - Detailed validation report dictionary.
    """
    raw_df = load_raw_data(source)
    validate_schema(raw_df)

    total_rows = len(raw_df)
    df = raw_df.copy()

    # Normalize string columns: fill NaN first, then strip whitespace
    imputed_descriptions = int(df["description"].isna().sum())
    df["description"] = df["description"].fillna(DEFAULT_DESCRIPTION).astype(str).str.strip()
    df.loc[df["description"] == "", "description"] = DEFAULT_DESCRIPTION

    imputed_categories = int(df["category"].isna().sum())
    df["category"] = df["category"].fillna(DEFAULT_CATEGORY).astype(str).str.strip()
    df.loc[df["category"] == "", "category"] = DEFAULT_CATEGORY

    # Parse Dates safely (errors='coerce' turns invalid dates into NaT)
    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    invalid_date_mask = parsed_dates.isna()
    rejected_invalid_dates = int(invalid_date_mask.sum())

    # Convert amounts to numeric (errors='coerce' turns non-numeric values into NaN)
    numeric_amounts = pd.to_numeric(df["amount"], errors="coerce")
    invalid_amount_type_mask = numeric_amounts.isna() | np.isinf(numeric_amounts)
    rejected_invalid_amounts = int(invalid_amount_type_mask.sum())

    # Check for non-positive amounts (amount <= 0)
    non_positive_mask = (~invalid_amount_type_mask) & (numeric_amounts <= 0)
    rejected_non_positive = int(non_positive_mask.sum())

    # Overall valid mask: date must be valid, amount must be numeric, finite, and strictly > 0
    valid_mask = (~invalid_date_mask) & (~invalid_amount_type_mask) & (numeric_amounts > 0)

    # Filter to valid rows
    clean_df = df[valid_mask].copy()
    clean_df["date"] = parsed_dates[valid_mask]
    clean_df["amount"] = numeric_amounts[valid_mask].round(2)

    # Derived time features for convenient grouping and filtering
    clean_df["year"] = clean_df["date"].dt.year
    clean_df["month"] = clean_df["date"].dt.to_period("M").astype(str)  # e.g., '2026-01'
    clean_df["month_name"] = clean_df["date"].dt.strftime("%b %Y")       # e.g., 'Jan 2026'
    clean_df["day_name"] = clean_df["date"].dt.strftime("%A")            # e.g., 'Monday'
    clean_df["formatted_date"] = clean_df["date"].dt.strftime("%Y-%m-%d")

    # Sort chronologically
    clean_df = clean_df.sort_values(by=["date", "amount"], ascending=[True, False]).reset_index(drop=True)

    valid_rows_count = len(clean_df)
    rejected_rows_count = total_rows - valid_rows_count

    report: Dict[str, Any] = {
        "total_rows_read": total_rows,
        "valid_rows": valid_rows_count,
        "rejected_rows": rejected_rows_count,
        "rejected_invalid_dates": rejected_invalid_dates,
        "rejected_invalid_amounts": rejected_invalid_amounts,
        "rejected_non_positive_amounts": rejected_non_positive,
        "imputed_descriptions": imputed_descriptions,
        "imputed_categories": imputed_categories,
        "is_valid": valid_rows_count > 0,
    }

    if valid_rows_count == 0:
        raise ValidationError(
            f"No valid expense records could be extracted from the file ({total_rows} row(s) read). "
            f"Rejections: {rejected_invalid_dates} invalid dates, "
            f"{rejected_invalid_amounts} non-numeric amounts, "
            f"{rejected_non_positive} zero or negative amounts."
        )

    return clean_df, report
