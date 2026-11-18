
##use dataset from huggingface to load the data
from __future__ import annotations

import re
from typing import BinaryIO

import pandas as pd

DATA_URL = (
    "https://huggingface.co/datasets/xyz4013/blood-bank-supply-management/"
    "resolve/main/data/blood_national_blood_centre.csv"
)

REQUIRED_COLUMNS = {
    "year",
    "month",
    "units_collected_month",
    "units_requested_month",
    "units_transfused",
    "units_expired",
    "blood_group",
    "product_category",
    "region_type",
    "shortage_in_last_month",
    "available_on_survey_day",
}

NUMERIC_COLUMNS = {
    "id",
    "has_component_separation",
    "has_nat_testing",
    "has_blood_bank_fridge",
    "has_platelet_agitator",
    "blood_bank_staff_count",
    "storage_temp_c",
    "shelf_life_days",
    "unit_cost_usd",
    "year",
    "month",
    "units_collected_month",
    "units_requested_month",
    "units_crossmatched",
    "units_transfused",
    "tti_screened_pct",
    "tti_positive_pct",
    "units_discarded_tti",
    "units_expired",
    "available_on_survey_day",
    "shortage_in_last_month",
    "shortage_days_last_month",
    "crossmatch_turnaround_hours",
    "transfusion_reactions_month",
    "maternal_deaths_no_blood",
    "surgical_cancellations_no_blood",
    "cold_chain_maintained",
    "report_submitted",
}


def load_remote_data() -> pd.DataFrame:
    """Load the public CSV over HTTPS without requiring Hugging Face adapters."""
    return pd.read_csv(DATA_URL)


def load_uploaded_data(file: BinaryIO) -> pd.DataFrame:
    """Read an uploaded CSV file."""
    return pd.read_csv(file)


def clean_data(raw_data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Normalize labels and text, convert known numeric fields, and drop exact duplicates."""
    cleaned = raw_data.copy()
    original_columns = list(cleaned.columns)
    cleaned.columns = [
        re.sub(r"[^a-zA-Z0-9]+", "_", str(column).strip()).strip("_").lower()
        for column in cleaned.columns
    ]
    if cleaned.columns.has_duplicates:
        raise ValueError("Column names are duplicated after normalization.")

    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        cleaned[column] = cleaned[column].replace("", pd.NA)

    for column in NUMERIC_COLUMNS.intersection(cleaned.columns):
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    rows_before = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    summary = {
        "rows_before": rows_before,
        "rows_after": len(cleaned),
        "duplicates_removed": rows_before - len(cleaned),
        "columns_renamed": sum(
            original != normalized
            for original, normalized in zip(original_columns, cleaned.columns)
        ),
    }
    return cleaned, summary


def validate_dashboard_columns(data: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(data.columns))
    if missing:
        raise ValueError(
            "The CSV is missing columns required by this dashboard: "
            + ", ".join(missing)
        )


def profile_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return a compact per-column inspection table."""
    return pd.DataFrame(
        {
            "column": data.columns,
            "dtype": [str(dtype) for dtype in data.dtypes],
            "missing": data.isna().sum().to_numpy(),
            "missing_%": (data.isna().mean() * 100).round(2).to_numpy(),
            "unique_values": data.nunique(dropna=False).to_numpy(),
        }
    )
