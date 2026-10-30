"""
Unit tests for spendwise.preprocessing module.
"""

import io
import pytest
import pandas as pd
from spendwise.preprocessing import (
    clean_expenses_data,
    validate_schema,
    load_raw_data,
    ValidationError,
    DEFAULT_DESCRIPTION,
    DEFAULT_CATEGORY,
)


def test_clean_expenses_data_valid():
    csv_data = """date,description,category,amount
2026-01-01,Groceries,Food,150.00
2026-01-02,Bus Pass,Transport,45.00
2026-01-03,Movie,Entertainment,20.00
"""
    clean_df, report = clean_expenses_data(io.StringIO(csv_data))

    assert len(clean_df) == 3
    assert report["valid_rows"] == 3
    assert report["rejected_rows"] == 0
    assert report["total_rows_read"] == 3
    assert list(clean_df["amount"]) == [150.00, 45.00, 20.00]
    assert "formatted_date" in clean_df.columns
    assert "month" in clean_df.columns


def test_missing_required_columns_raises_error():
    # Missing 'amount' column
    csv_data = """date,description,category
2026-01-01,Groceries,Food
"""
    with pytest.raises(ValidationError, match="Missing required column"):
        clean_expenses_data(io.StringIO(csv_data))


def test_invalid_dates_and_amounts_rejected():
    csv_data = """date,description,category,amount
2026-01-01,Valid Food,Food,120.00
invalid-date,Invalid Date Item,Food,50.00
2026-01-03,Not A Number Amount,Food,FREE
2026-01-04,Negative Expense,Food,-50.00
2026-01-05,Zero Expense,Food,0.00
"""
    clean_df, report = clean_expenses_data(io.StringIO(csv_data))

    # Only the first row is valid
    assert len(clean_df) == 1
    assert report["valid_rows"] == 1
    assert report["rejected_rows"] == 4
    assert report["rejected_invalid_dates"] == 1
    assert report["rejected_invalid_amounts"] == 1
    assert report["rejected_non_positive_amounts"] == 2
    assert clean_df.iloc[0]["description"] == "Valid Food"


def test_imputation_of_missing_description_and_category():
    csv_data = """date,description,category,amount
2026-01-01,,Food,100.00
2026-01-02,Lunch,,50.00
2026-01-03,,,30.00
"""
    clean_df, report = clean_expenses_data(io.StringIO(csv_data))

    assert len(clean_df) == 3
    assert clean_df.iloc[0]["description"] == DEFAULT_DESCRIPTION
    assert clean_df.iloc[1]["category"] == DEFAULT_CATEGORY
    assert clean_df.iloc[2]["description"] == DEFAULT_DESCRIPTION
    assert clean_df.iloc[2]["category"] == DEFAULT_CATEGORY


def test_empty_csv_raises_validation_error():
    with pytest.raises(ValidationError):
        clean_expenses_data(io.StringIO(""))


def test_csv_with_only_headers_raises_error():
    csv_data = "date,description,category,amount\n"
    with pytest.raises(ValidationError, match="zero data rows"):
        clean_expenses_data(io.StringIO(csv_data))


def test_no_valid_rows_raises_validation_error():
    csv_data = """date,description,category,amount
invalid,item,cat,-100
bad-date,item2,cat,0
"""
    with pytest.raises(ValidationError, match="No valid expense records"):
        clean_expenses_data(io.StringIO(csv_data))
