# pageDiagrams.py — Data Exploration and Filtering

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Explore Data — Filters and Charts", layout="wide")

# ------------------ DATA LOADER ------------------
@st.cache_data
def load_data():
    """Load the speed dating dataset, preferring the raw version if available."""
    candidates = [
        "speeddating_raw.csv",
        os.path.join("raw", "speeddating.csv"),
        "speeddating.csv",
        os.path.join("proj", "speeddating.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df, path
    raise FileNotFoundError("No valid CSV file found in the expected locations.")

def decode_columns(df: pd.DataFrame, enable: bool) -> pd.DataFrame:
    """Convert binary or encoded columns into human-readable labels."""
    if not enable:
        return df

    out = df.copy()

    def is_binary_numeric(series: pd.Series) -> bool:
        num = pd.to_numeric(series, errors="coerce")
        unique = pd.unique(num.dropna())
        return set(unique).issubset({0, 1}) and len(unique) > 0

    # Gender mapping
    if "gender" in out.columns:
        if is_binary_numeric(out["gender"]):
            out["gender"] = (
                pd.to_numeric(out["gender"], errors="coerce")
                .map({0: "male", 1: "female"})
                .astype("category")
            )

    # Convert *_bin columns to "yes"/"no"
    for col in out.columns:
        if col.endswith("_bin") and is_binary_numeric(out[col]):
            out[col] = (
                pd.to_numeric(out[col], errors="coerce")
                .map({0: "no", 1: "yes"})
                .astype("category")
            )

    # Leave 'field' and other categorical columns unchanged
    return out


df_raw, loaded_path = load_data()

st.title("Data Exploration")
st.caption("Explore the dataset interactively using filters and visual summaries.")

with st.sidebar:
    st.markdown(f"**Loaded file:** `{loaded_path}`")
    decode = st.toggle("Convert encoded columns to readable labels", value=True)

df = decode_columns(df_raw, decode)

# ------------------ FILTER PANEL ------------------
with st.expander("Filters", expanded=True):
    col1, col2, col3 = st.columns(3)

    # Gender filter
    with col1:
        if "gender" in df.columns:
            gender_values = sorted([str(x) for x in df["gender"].dropna().unique()], key=str.lower)
            gender = st.selectbox("Gender", ["(all)"] + gender_values)
        else:
            gender = "(all)"

    # Age filter
    with col2:
        if "age" in df.columns:
            a_min, a_max = int(np.nanmin(df["age"])), int(np.nanmax(df["age"]))
            age_range = st.slider("Age Range", a_min, a_max, (a_min, a_max))
        else:
            age_range = (0, 200)

    # Field filter
    with col3:
        field_col = next((c for c in ["field", "field_text", "field_name"] if c in df.columns), None)
        if field_col:
            field_values = sorted([str(x) for x in df[field_col].dropna().unique()], key=str.lower)
            field_sel = st.selectbox("Field", ["(all)"] + field_values)
        else:
            field_sel = "(all)"
            field_col = None

# ------------------ APPLY FILTERS ------------------
filtered = df.copy()

if "gender" in filtered and gender != "(all)":
    filtered = filtered[filtered["gender"].astype(str) == gender]

if "age" in filtered:
    filtered = filtered[
        (filtered["age"].fillna(0) >= age_range[0])
        & (filtered["age"].fillna(0) <= age_range[1])
    ]

if field_col and field_sel != "(all)":
    filtered = filtered[filtered[field_col].astype(str) == field_sel]

st.markdown(f"**Rows after filters:** {len(filtered):,}")
st.dataframe(filtered.head(200), use_container_width=True)


# the histogram thing


# ------------------ HISTOGRAM (from filtered data, no fake numeric encoding) ------------------
st.markdown("---")
st.subheader("Histogram (from filtered rows)")

st.write(
    "The histogram below shows how the values of a selected numeric feature "
    "are distributed within the currently filtered dataset. Each bar represents "
    "how many participants fall into a specific value range. By adjusting the number "
    "of bins or toggling normalization, you can better observe patterns such as concentration "
    "around certain scores or how evenly spread the data is."
)

# only keep columns that are *purely numeric*, not categorical codes
num_cols = [
    c for c in filtered.columns
    if pd.api.types.is_numeric_dtype(filtered[c])
    and not pd.api.types.is_categorical_dtype(filtered[c])
    and not filtered[c].dropna().astype(str).str.isalpha().any()
]

if not num_cols:
    st.info("No purely numeric columns available in the filtered data.")
else:
    h_col = st.selectbox("Numeric column", num_cols)
    h_bins = st.slider("Bins", 5, 60, 20, 1)
    h_density = st.checkbox("Normalize (density)", value=True)

    # use numeric conversion safely (ignore categorical-looking columns)
    ser = pd.to_numeric(filtered[h_col], errors="coerce").dropna()
    if ser.empty:
        st.info("No numeric data for this column after filtering.")
    else:
        fig, ax = plt.subplots(figsize=(5.0, 2.8), dpi=130)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
        ax.hist(ser, bins=h_bins, density=h_density, edgecolor="black", linewidth=0.35, color="#4a90e2")
        ax.set_title(f"Histogram — {h_col}")
        ax.set_xlabel(h_col)
        ax.set_ylabel("Density" if h_density else "Count")
        plt.tight_layout(pad=1.0)
        st.pyplot(fig, use_container_width=False, bbox_inches="tight", width=350)

