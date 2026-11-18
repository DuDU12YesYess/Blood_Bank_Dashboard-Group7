
## connect to streamlit and load the data

import calendar
import urllib.error

import pandas as pd
import streamlit as st

from data_utils import (
    clean_data,
    load_remote_data,
    load_uploaded_data,
    profile_data,
    validate_dashboard_columns,
)

st.set_page_config(page_title="Blood Supply Dashboard", page_icon="🩸", layout="wide")


@st.cache_data(ttl=3600, show_spinner="Loading blood supply data...")
def cached_remote_data() -> pd.DataFrame:
    return load_remote_data()


@st.cache_data(show_spinner=False)
def cached_uploaded_data(file_contents: bytes) -> pd.DataFrame:
    from io import BytesIO

    return load_uploaded_data(BytesIO(file_contents))


st.title("Blood Supply & Demand Dashboard")
st.caption("Inspect, clean, and explore monthly blood supply indicators.")

with st.sidebar:
    st.header("Data source")
    source = st.radio(
        "Choose a data source",
        ("Hugging Face dataset", "Upload a CSV"),
    )
    uploaded_file = (
        st.file_uploader("Choose a CSV file", type=["csv"])
        if source == "Upload a CSV"
        else None
    )

if source == "Upload a CSV" and uploaded_file is None:
    st.info("Upload a CSV with the blood supply dataset columns to continue.")
    st.stop()

try:
    if source == "Hugging Face dataset":
        raw_data = cached_remote_data()
    else:
        if uploaded_file is None:
            st.stop()
        raw_data = cached_uploaded_data(uploaded_file.getvalue())

    data, cleaning_summary = clean_data(raw_data)
    validate_dashboard_columns(data)
except (OSError, ValueError, pd.errors.ParserError, urllib.error.URLError) as error:
    st.error(f"Could not load or prepare the data: {error}")
    st.stop()

with st.sidebar:
    st.header("Filters")
    years = sorted(data["year"].dropna().unique().tolist())
    selected_years = st.multiselect("Year", years, default=years)
    months = list(range(1, 13))
    selected_months = st.multiselect(
        "Month",
        months,
        default=months,
        format_func=lambda month: calendar.month_name[month],
    )

    filter_columns = {
        "Region": "region_type",
        "Blood group": "blood_group",
        "Product category": "product_category",
    }
    selections: dict[str, list[object]] = {}
    for label, column in filter_columns.items():
        options = sorted(data[column].dropna().unique().tolist())
        selections[column] = st.multiselect(label, options, default=options)

filtered = data.loc[
    data["year"].isin(selected_years) & data["month"].isin(selected_months)
].copy()
for column, selected in selections.items():
    filtered = filtered.loc[filtered[column].isin(selected)]

if filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()

requested = int(filtered["units_requested_month"].sum())
collected = int(filtered["units_collected_month"].sum())
transfused = int(filtered["units_transfused"].sum())
expired = int(filtered["units_expired"].sum())
shortage_rate = filtered["shortage_in_last_month"].mean() * 100

metric_columns = st.columns(5)
metric_columns[0].metric("Records", f"{len(filtered):,}")
metric_columns[1].metric("Units collected", f"{collected:,}")
metric_columns[2].metric("Units transfused", f"{transfused:,}")
metric_columns[3].metric("Expired units", f"{expired:,}")
metric_columns[4].metric("Units requested", f"{requested:,}")
st.metric("Records reporting a shortage", f"{shortage_rate:.1f}%")

trend = (
    filtered.groupby(["year", "month"], as_index=False)[
        ["units_collected_month", "units_requested_month", "units_transfused"]
    ]
    .sum()
    .sort_values(["year", "month"])
)
trend["period"] = pd.to_datetime(
    dict(year=trend["year"], month=trend["month"], day=1), errors="coerce"
)
trend = trend.dropna(subset=["period"]).set_index("period")

left, right = st.columns(2)
with left:
    st.subheader("Monthly supply and demand")
    st.line_chart(
        trend[
            [
                "units_collected_month",
                "units_requested_month",
                "units_transfused",
            ]
        ]
    )

with right:
    st.subheader("Units requested by blood group")
    blood_group_demand = (
        filtered.groupby("blood_group")["units_requested_month"]
        .sum()
        .sort_values(ascending=False)
    )
    st.bar_chart(blood_group_demand)

st.subheader("Shortages by product category")
shortages = (
    filtered.groupby("product_category", as_index=False)
    .agg(
        shortage_rate=("shortage_in_last_month", "mean"),
        records=("shortage_in_last_month", "size"),
    )
    .sort_values("shortage_rate", ascending=False)
)
shortages["shortage_rate"] = (shortages["shortage_rate"] * 100).round(1)
st.dataframe(shortages, hide_index=True)

with st.expander("Data inspection and cleaning report"):
    st.write(
        f"Rows: {cleaning_summary['rows_before']:,} before cleaning, "
        f"{cleaning_summary['rows_after']:,} after; "
        f"{cleaning_summary['duplicates_removed']:,} exact duplicate rows removed; "
        f"{cleaning_summary['columns_renamed']:,} column labels normalized."
    )
    st.subheader("Column profile")
    st.dataframe(profile_data(data), hide_index=True)
    st.subheader("Cleaned data preview")
    st.dataframe(data.head(100), hide_index=True)

st.download_button(
    "Download cleaned data",
    data=data.to_csv(index=False).encode("utf-8"),
    file_name="blood_supply_cleaned.csv",
    mime="text/csv",
)
