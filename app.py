"""
SpendWise - Smart Expense Analyzer & ML Anomaly Detection Dashboard.

A beginner-friendly personal finance analytics platform built with Streamlit,
Pandas, NumPy, and scikit-learn Isolation Forest.
"""

from pathlib import Path
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from spendwise.preprocessing import clean_expenses_data, ValidationError
from spendwise.analytics import (
    compute_summary_statistics,
    compute_category_breakdown,
    compute_monthly_trends,
    generate_spending_insights,
    compute_mad_threshold,
)
from spendwise.anomaly_detection import detect_anomalies, DEFAULT_CONTAMINATION

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SpendWise | Smart Expense Analyzer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Subtle, polished CSS styling for clean card aesthetic
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 1rem 1.25rem;
        border-radius: 0.75rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    div[data-testid="stMetric"]:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    /* Section Card Containers */
    .spendwise-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 0.75rem;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.04);
    }
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
    }
    .badge-sample {
        background-color: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
    }
    .badge-custom {
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
    }
    .badge-alert {
        background-color: #FEF2F2;
        color: #B91C1C;
        border: 1px solid #FECACA;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SAMPLE_DATA_PATH = Path(__file__).parent / "data" / "sample_expenses.csv"


# --- CACHED DATA PIPELINE ---
@st.cache_data(show_spinner=False)
def load_and_preprocess_file(file_content: bytes, is_sample: bool = False):
    """
    Load raw CSV content and execute the cleaning and validation pipeline.
    Results are cached to avoid redundant processing on re-renders.
    """
    buffer = io.BytesIO(file_content)
    clean_df, report = clean_expenses_data(buffer)
    return clean_df, report


@st.cache_data(show_spinner=False)
def run_cached_anomaly_detection(df_records: tuple, contamination: float):
    """
    Execute Isolation Forest anomaly detection with caching based on
    data hash and contamination parameter.
    """
    # Reconstruct DataFrame from records tuple
    df = pd.DataFrame(df_records)
    result_df, metadata = detect_anomalies(df, contamination=contamination)
    return result_df, metadata


def get_sample_csv_bytes() -> bytes:
    """Read the built-in sample CSV file as raw bytes."""
    with open(SAMPLE_DATA_PATH, "rb") as f:
        return f.read()


# --- SESSION STATE INITIALIZATION ---
if "data_bytes" not in st.session_state:
    st.session_state["data_bytes"] = get_sample_csv_bytes()
    st.session_state["is_sample"] = True
    st.session_state["filename"] = "sample_expenses.csv"
    st.session_state["uploader_key"] = 0


def reset_to_sample_data():
    """Reset the application back to the default bundled dataset."""
    st.session_state["data_bytes"] = get_sample_csv_bytes()
    st.session_state["is_sample"] = True
    st.session_state["filename"] = "sample_expenses.csv"
    st.session_state["uploader_key"] += 1
    st.rerun()


# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("## 💳 **SpendWise**")
    st.caption("Smart Personal Expense Analyzer & ML Anomaly Detector")
    st.divider()

    # Active Dataset Status Badge
    if st.session_state["is_sample"]:
        st.markdown(
            '<span class="badge-pill badge-sample">🟢 Built-in Sample Dataset Active</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span class="badge-pill badge-custom">🔵 Custom CSV: {st.session_state["filename"]}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("### **Data Controls**")

    # CSV File Upload
    uploaded_file = st.file_uploader(
        "Upload Custom CSV",
        type=["csv"],
        key=f"uploader_{st.session_state['uploader_key']}",
        help="Upload an expense CSV (max 5MB). Required columns: date, description, category, amount.",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        # Verify 5MB limit
        if len(file_bytes) > 5 * 1024 * 1024:
            st.error("Uploaded file exceeds 5MB size limit. Please upload a smaller file.")
        else:
            if file_bytes != st.session_state["data_bytes"] or st.session_state["is_sample"]:
                st.session_state["data_bytes"] = file_bytes
                st.session_state["is_sample"] = False
                st.session_state["filename"] = uploaded_file.name
                st.rerun()

    # Reset Action Button
    if not st.session_state["is_sample"]:
        st.button("🔄 Reset to Sample Data", on_click=reset_to_sample_data, use_container_width=True)

    # Download Sample CSV
    sample_bytes = get_sample_csv_bytes()
    st.download_button(
        label="📥 Download Sample CSV",
        data=sample_bytes,
        file_name="sample_expenses.csv",
        mime="text/csv",
        use_container_width=True,
        help="Download the verified sample dataset to inspect its format or use as a template.",
    )

    st.divider()
    st.markdown("### **ML Model Controls**")
    sensitivity_pct = st.slider(
        "Anomaly Sensitivity",
        min_value=1,
        max_value=15,
        value=5,
        step=1,
        format="%d%%",
        help="Controls the Isolation Forest contamination rate. Higher values flag more borderline unusual transactions.",
    )
    contamination_rate = sensitivity_pct / 100.0

    st.caption(
        "💡 *Higher sensitivity = lower isolation threshold, flagging subtle anomalies. "
        "Default is 5%.*"
    )


# --- DATA PROCESSING & VALIDATION ---
try:
    df, validation_report = load_and_preprocess_file(
        st.session_state["data_bytes"],
        is_sample=st.session_state["is_sample"],
    )
except ValidationError as e:
    st.error(f"❌ **Data Validation Error:** {str(e)}")
    st.info("💡 Make sure your CSV contains columns: `date`, `description`, `category`, and `amount`.")
    if st.button("Revert to Default Sample Dataset"):
        reset_to_sample_data()
    st.stop()
except Exception as e:
    st.error(f"❌ **Unexpected Error:** {str(e)}")
    if st.button("Revert to Default Sample Dataset"):
        reset_to_sample_data()
    st.stop()


# Show Validation Summary if rows were rejected
if validation_report["rejected_rows"] > 0:
    with st.expander(f"⚠️ Notice: {validation_report['rejected_rows']} invalid rows were excluded from analysis"):
        st.markdown(
            f"""
            - **Total Rows Read:** {validation_report['total_rows_read']}
            - **Valid Rows Analyzed:** {validation_report['valid_rows']}
            - **Invalid Dates:** {validation_report['rejected_invalid_dates']}
            - **Non-Numeric Amounts:** {validation_report['rejected_invalid_amounts']}
            - **Zero or Negative Amounts:** {validation_report['rejected_non_positive_amounts']}
            - **Descriptions Imputed:** {validation_report['imputed_descriptions']}
            - **Categories Imputed:** {validation_report['imputed_categories']}
            """
        )

# Compute core analytics
summary_stats = compute_summary_statistics(df)
category_breakdown = compute_category_breakdown(df)
monthly_trends = compute_monthly_trends(df)
insights = generate_spending_insights(df)

# Run ML Anomaly Detection
# Convert DataFrame to records tuple for caching
records_tuple = tuple(df.to_dict(orient="records"))
anomaly_df, anomaly_metadata = run_cached_anomaly_detection(records_tuple, contamination=contamination_rate)


# --- MAIN APPLICATION HEADER ---
st.title("SpendWise: Smart Expense Analyzer")
st.markdown(
    "Explore personal spending patterns, understand category allocations, "
    "and identify statistically unusual transactions using machine learning."
)

# Top Banner with Active Source Indicator
col_banner_1, col_banner_2 = st.columns([3, 1])
with col_banner_1:
    if st.session_state["is_sample"]:
        st.info("ℹ️ Currently demonstrating with the **default sample dataset** (Jan – Apr 2026). Upload your CSV in the sidebar anytime!")
    else:
        st.success(f"✅ Analyzing **{st.session_state['filename']}** ({summary_stats['transaction_count']} valid expenses).")
with col_banner_2:
    st.metric(label="Total Expenses Analyzed", value=f"{summary_stats['transaction_count']}")


# --- APPLICATION TABS ---
tab_overview, tab_transactions, tab_insights, tab_anomaly, tab_about = st.tabs([
    "📊 Overview",
    "📋 Transactions",
    "💡 Spending Insights",
    "🔍 Anomaly Detection",
    "ℹ️ About the Model",
])


# ==============================================================================
# TAB A: OVERVIEW
# ==============================================================================
with tab_overview:
    # 1. KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Total Spending",
            value=f"${summary_stats['total_spend']:,.2f}",
        )
    with col2:
        st.metric(
            label="Valid Transactions",
            value=f"{summary_stats['transaction_count']:,}",
        )
    with col3:
        st.metric(
            label="Average Transaction",
            value=f"${summary_stats['mean_amount']:,.2f}",
            help=f"Median: ${summary_stats['median_amount']:,.2f}",
        )
    with col4:
        st.metric(
            label="Largest Single Expense",
            value=f"${summary_stats['max_amount']:,.2f}",
            help=f"Minimum: ${summary_stats['min_amount']:,.2f}",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Charts Row
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Category Breakdown")
        if not category_breakdown.empty:
            fig_cat = px.pie(
                category_breakdown,
                names="category",
                values="total",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Prism,
            )
            fig_cat.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>Total: $%{value:,.2f}<br>Share: %{percent}<extra></extra>",
            )
            fig_cat.update_layout(
                margin=dict(t=20, b=20, l=10, r=10),
                showlegend=False,
                height=350,
            )
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("No category data available.")

    with chart_col2:
        st.subheader("Monthly Spending Trend")
        if len(monthly_trends) >= 2:
            fig_month = px.line(
                monthly_trends,
                x="month_name",
                y="total",
                markers=True,
                text="total",
                color_discrete_sequence=["#0F766E"],
            )
            fig_month.update_traces(
                texttemplate="$%{text:,.0f}",
                textposition="top center",
                hovertemplate="<b>%{x}</b><br>Spend: $%{y:,.2f}<extra></extra>",
                line=dict(width=3),
                marker=dict(size=8),
            )
            fig_month.update_layout(
                margin=dict(t=20, b=20, l=10, r=10),
                xaxis_title="Month",
                yaxis_title="Total Spend ($)",
                height=350,
            )
            st.plotly_chart(fig_month, use_container_width=True)
        else:
            # Single month fallback
            fig_single = px.bar(
                monthly_trends,
                x="month_name",
                y="total",
                color_discrete_sequence=["#0F766E"],
                text="total",
            )
            fig_single.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            fig_single.update_layout(
                margin=dict(t=20, b=20, l=10, r=10),
                xaxis_title="Month",
                yaxis_title="Total Spend ($)",
                height=350,
            )
            st.plotly_chart(fig_single, use_container_width=True)
            st.caption("ℹ️ Only one calendar month detected. Upload multiple months of data to view multi-period trends.")

    # 3. Recent Transactions Snapshot
    st.subheader("Recent Expenses Snapshot")
    recent_df = df.sort_values(by="date", ascending=False).head(8)
    display_recent = recent_df[["formatted_date", "description", "category", "amount"]].copy()
    display_recent.columns = ["Date", "Description", "Category", "Amount ($)"]
    st.dataframe(
        display_recent.style.format({"Amount ($)": "${:,.2f}"}),
        use_container_width=True,
        hide_index=True,
    )


# ==============================================================================
# TAB B: TRANSACTIONS
# ==============================================================================
with tab_transactions:
    st.subheader("Expense Ledger & Search")
    st.caption("Filter, search, and sort through all valid transactions in the current dataset.")

    # Filter Controls
    f_col1, f_col2, f_col3, f_col4 = st.columns([1.5, 1.5, 1.2, 1.2])

    with f_col1:
        search_query = st.text_input("🔍 Search Description", placeholder="e.g. Groceries, Rent...")

    with f_col2:
        available_categories = sorted(df["category"].unique().tolist())
        selected_categories = st.multiselect(
            "Filter Category",
            options=available_categories,
            default=[],
            placeholder="All Categories",
        )

    with f_col3:
        min_date = df["date"].min().date()
        max_date = df["date"].max().date()
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

    with f_col4:
        sort_by = st.selectbox(
            "Sort By",
            options=["Date (Newest First)", "Date (Oldest First)", "Amount (Highest First)", "Amount (Lowest First)"],
        )

    # Apply Filters
    filtered_df = df.copy()

    if search_query.strip():
        filtered_df = filtered_df[
            filtered_df["description"].str.contains(search_query.strip(), case=False, na=False)
        ]

    if selected_categories:
        filtered_df = filtered_df[filtered_df["category"].isin(selected_categories)]

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
        filtered_df = filtered_df[
            (filtered_df["date"].dt.date >= start_d) & (filtered_df["date"].dt.date <= end_d)
        ]

    # Apply Sorting
    if sort_by == "Date (Newest First)":
        filtered_df = filtered_df.sort_values(by="date", ascending=False)
    elif sort_by == "Date (Oldest First)":
        filtered_df = filtered_df.sort_values(by="date", ascending=True)
    elif sort_by == "Amount (Highest First)":
        filtered_df = filtered_df.sort_values(by="amount", ascending=False)
    elif sort_by == "Amount (Lowest First)":
        filtered_df = filtered_df.sort_values(by="amount", ascending=True)

    # Results Display
    if filtered_df.empty:
        st.warning("⚠️ No transactions matched your filter criteria. Try adjusting the search query or date range.")
    else:
        st.markdown(
            f"**Showing {len(filtered_df)} of {len(df)} transactions** (Total: **${filtered_df['amount'].sum():,.2f}**)"
        )
        display_table = filtered_df[["formatted_date", "description", "category", "amount"]].copy()
        display_table.columns = ["Date", "Description", "Category", "Amount ($)"]
        st.dataframe(
            display_table.style.format({"Amount ($)": "${:,.2f}"}),
            use_container_width=True,
            hide_index=True,
            height=400,
        )


# ==============================================================================
# TAB C: SPENDING INSIGHTS
# ==============================================================================
with tab_insights:
    st.subheader("Data-Driven Spending Insights")
    st.markdown(
        "These transparent insights are derived mathematically from your actual spending figures, "
        "highlighting category weights, distribution skewness, and period comparisons."
    )

    if insights:
        ins_cols = st.columns(len(insights))
        for col, ins in zip(ins_cols, insights):
            with col:
                st.markdown(
                    f"""
                    <div class="spendwise-card">
                        <span class="badge-pill badge-sample">{ins['badge']}</span>
                        <h4 style="margin-top: 0.5rem; margin-bottom: 0.5rem; font-size: 1rem;">{ins['title']}</h4>
                        <p style="font-size: 0.9rem; color: #334155; line-height: 1.4;">{ins['description']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.divider()

    # Detailed Category Breakdown Table
    st.subheader("Category Expenditure Table")
    if not category_breakdown.empty:
        cat_display = category_breakdown.copy()
        cat_display.columns = ["Category", "Total Spend ($)", "Share of Wallet (%)", "Count", "Average ($)", "Median ($)"]
        st.dataframe(
            cat_display.style.format({
                "Total Spend ($)": "${:,.2f}",
                "Share of Wallet (%)": "{:.1f}%",
                "Average ($)": "${:,.2f}",
                "Median ($)": "${:,.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # Mathematical Deep Dive Box
    with st.expander("📐 Mathematical Notes: Mean vs. Median & Skewness"):
        st.markdown(
            f"""
            - **Mean (${summary_stats['mean_amount']:,.2f}):** The arithmetic average (total divided by count).
              It is sensitive to extreme values. A single large expense (e.g., ${summary_stats['max_amount']:,.2f})
              pulls the mean upwards.
            - **Median (${summary_stats['median_amount']:,.2f}):** The 50th percentile (middle transaction amount).
              It is robust against outliers because extreme purchases do not alter the center rank.
            - **Ratio (Mean / Median = {summary_stats['mean_median_ratio']}x):** In this dataset, the mean is
              **{summary_stats['mean_median_ratio']} times** the median, confirming a right-skewed distribution.
            """
        )


# ==============================================================================
# TAB D: ANOMALY DETECTION (ML)
# ==============================================================================
with tab_anomaly:
    st.subheader("Isolation Forest Anomaly Detection")
    st.markdown(
        "Isolation Forest identifies transactions that stand out from typical spending behavior. "
        "It partitions transactions using random decision trees; **outliers are isolated near the root "
        "in far fewer splits** than normal, dense transactions."
    )

    # Anomaly KPI Metrics
    anom_kpi1, anom_kpi2, anom_kpi3, anom_kpi4 = st.columns(4)
    with anom_kpi1:
        st.metric("Total Analyzed", f"{anomaly_metadata['total_analyzed']}")
    with anom_kpi2:
        st.metric(
            "Flagged Anomalies",
            f"{anomaly_metadata['anomaly_count']}",
            delta=f"{anomaly_metadata['anomaly_rate_pct']}% of total",
            delta_color="inverse",
        )
    with anom_kpi3:
        st.metric("Sensitivity Setting", f"{int(contamination_rate * 100)}%")
    with anom_kpi4:
        mad_res = compute_mad_threshold(df)
        st.metric("Statistical MAD Baseline", f"${mad_res['threshold']:,.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Scatter Plot: Anomalies vs Normal Expenses
    st.subheader("Expense Distribution & Anomaly Flags")
    fig_scatter = px.scatter(
        anomaly_df,
        x="date",
        y="amount",
        color="anomaly_label",
        color_discrete_map={
            "Flagged Anomaly": "#DC2626",  # Red
            "Normal Expense": "#0F766E",   # Teal
        },
        size=np.clip(anomaly_df["amount"], 10, 10000),
        hover_data={
            "description": True,
            "category": True,
            "amount": ":$.2f",
            "severity_score": ":.1f",
            "date": False,
        },
        labels={"amount": "Expense Amount ($)", "date": "Transaction Date", "anomaly_label": "Model Status"},
    )
    fig_scatter.update_layout(
        margin=dict(t=20, b=20, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Flagged Anomalies Table
    st.subheader("Flagged Transactions Review")
    flagged_df = anomaly_df[anomaly_df["is_anomaly"]].copy()

    if flagged_df.empty:
        st.success("✅ No transactions were flagged as anomalous at the current sensitivity setting.")
    else:
        st.markdown(
            f"The model flagged **{len(flagged_df)} transaction(s)**. Review each item below to determine "
            f"if it represents an expected expense (like planned rent) or an unexpected purchase."
        )

        display_flagged = flagged_df[
            ["formatted_date", "description", "category", "amount", "category_ratio", "severity_score"]
        ].sort_values(by="severity_score", ascending=False)

        display_flagged.columns = [
            "Date",
            "Description",
            "Category",
            "Amount ($)",
            "vs Category Median",
            "Severity Score (0-100)",
        ]

        st.dataframe(
            display_flagged.style.format({
                "Amount ($)": "${:,.2f}",
                "vs Category Median": "{:.1f}x",
                "Severity Score (0-100)": "{:.1f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # Educational Disclaimer Alert
    st.info(
        "⚠️ **Educational Disclaimer:** Anomaly detection identifies statistical deviations, not financial fraud. "
        "A large expense (such as rent or a planned laptop purchase) is an unusual transaction by magnitude, "
        "but is completely legitimate. Flagged transactions should always be reviewed by a human."
    )


# ==============================================================================
# TAB E: ABOUT THE MODEL
# ==============================================================================
with tab_about:
    st.subheader("Understanding the Machine Learning Pipeline")

    st.markdown(
        """
        SpendWise utilizes **Isolation Forest**, an unsupervised anomaly detection algorithm introduced
        by Liu, Ting, and Zhou (2008). Here is how the underlying mechanism functions:
        """
    )

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown(
            """
            ### 🌲 How Isolation Forest Works
            1. **Random Splitting:** The algorithm builds an ensemble of isolation decision trees (`iTrees`).
               At each node, it selects a feature at random and then selects a random split point between
               the minimum and maximum values of that feature.
            2. **Fewer Splits for Anomalies:** Because outliers have distinct values and sparse neighbors,
               they are quickly separated into leaf nodes near the top of the tree.
            3. **Path Length:** The average path length across all trees serves as the anomaly measure.
               Short path lengths indicate anomalies; long path lengths indicate normal points clustered together.
            """
        )

    with col_m2:
        st.markdown(
            """
            ### 🎯 Features Supplied to the Model
            We intentionally use a small, transparent feature set:
            - **`amount`:** Raw transaction magnitude in dollars.
            - **`log_amount` ($ln(1 + x)$):** Normalizes the heavy right tail of expense distributions so extreme
              values do not overpower linear distance metrics.
            - **`category_amount_ratio`:** Ratio of the expense to its category median. This provides contextual
              intelligence (e.g. flagging a $2,000 restaurant dinner while recognizing that $2,000 rent is normal).
            """
        )

    st.divider()

    st.markdown(
        """
        ### 🔍 Anomaly Detection vs. Supervised Classification
        | Aspect | Supervised Classification | Isolation Forest (Unsupervised) |
        | :--- | :--- | :--- |
        | **Labels Required** | Requires labelled data ("Fraud" vs. "Normal"). | **No labels required.** Learns purely from data structure. |
        | **Objective** | Predicts predefined class categories. | Measures **isolation depth** and statistical deviation. |
        | **Unknown Outliers**| Only identifies known patterns seen in training. | Can detect novel, unseen outlier patterns. |
        | **Output Meaning** | Class probability (e.g. 94% chance of fraud). | **Anomaly score / severity** (how isolated the observation is). |

        ### ⚙️ Sensitivity (Contamination) Setting
        - The **contamination parameter** represents the estimated proportion of outliers in the data.
        - In the sidebar, adjusting the slider sets this parameter between **1% and 15%** (default 5%).
        - A lower sensitivity (e.g., 2%) flags only extreme outliers.
        - A higher sensitivity (e.g., 10%) captures borderline unusual transactions.
        """
    )

    st.divider()
    st.caption("SpendWise v1.0 • Built with Python, Streamlit, Pandas, NumPy, and scikit-learn.")
