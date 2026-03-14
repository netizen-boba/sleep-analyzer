from __future__ import annotations

import difflib
import pandas as pd
import streamlit as st

# Canonical field names and their known aliases (all lowercase, stripped)
CANONICAL_ALIASES = {
    "date": ["date", "night", "day", "recorded", "sleep date", "record date", "sleep night"],
    "bedtime": ["bedtime", "bed time", "sleep start", "lights out", "fell asleep", "sleep onset", "start time"],
    "wake_time": ["wake time", "wakeup", "wake up", "rise time", "alarm", "end time", "wake"],
    "total_sleep": ["total sleep", "sleep duration", "hours slept", "duration", "total sleep time", "sleep time", "time asleep"],
    "sleep_score": ["sleep score", "quality", "score", "sleep quality", "rating", "sleep rating", "overall score"],
    "deep_sleep": ["deep sleep", "deep", "slow wave", "sws", "deep sleep duration", "deep sleep time"],
    "rem_sleep": ["rem", "rem sleep", "dream sleep", "rapid eye movement", "rem duration"],
    "light_sleep": ["light sleep", "light", "nrem light", "light sleep duration", "light sleep time"],
    "awakenings": ["awakenings", "wake ups", "wakeups", "arousals", "disruptions", "times awake", "number of awakenings"],
    "heart_rate": ["heart rate", "avg hr", "resting hr", "bpm", "avg heart rate", "average heart rate", "resting heart rate"],
    "hrv": ["hrv", "heart rate variability", "rmssd", "sdnn"],
    "respiratory_rate": ["respiratory rate", "breathing rate", "breaths", "breaths per minute", "breath rate"],
}

REQUIRED_COLUMNS = {"date"}
USEFUL_COLUMNS = {"sleep_score", "total_sleep"}


def _normalize(s: str) -> str:
    """Lowercase, strip, remove punctuation for comparison."""
    import re
    return re.sub(r"[^\w\s]", "", s.lower()).strip()


def detect_column_mapping(df_columns: list[str]) -> dict[str, str | None]:
    """
    Auto-detect which actual column corresponds to each canonical field.
    Returns a dict: {canonical_field: actual_column_name or None}
    """
    normalized_actual = {col: _normalize(col) for col in df_columns}
    mapping = {field: None for field in CANONICAL_ALIASES}

    # Track which actual columns have already been claimed
    claimed = set()

    for canonical, aliases in CANONICAL_ALIASES.items():
        best_match = None
        best_score = 0

        for actual_col, norm_actual in normalized_actual.items():
            if actual_col in claimed:
                continue
            for alias in aliases:
                norm_alias = _normalize(alias)
                # Exact match
                if norm_actual == norm_alias:
                    best_match = actual_col
                    best_score = 100
                    break
                # Substring match
                if norm_alias in norm_actual or norm_actual in norm_alias:
                    score = 80
                    if score > best_score:
                        best_score = score
                        best_match = actual_col
                # Fuzzy match
                ratio = difflib.SequenceMatcher(None, norm_actual, norm_alias).ratio()
                score = int(ratio * 60)
                if score > best_score and score > 35:
                    best_score = score
                    best_match = actual_col
            if best_score == 100:
                break

        if best_match:
            mapping[canonical] = best_match
            claimed.add(best_match)

    return mapping


def render_column_mapper(df_columns: list[str], detected: dict[str, str | None]) -> dict[str, str | None]:
    """
    Render sidebar selectboxes for user to confirm/correct the column mapping.
    Returns the confirmed mapping.
    """
    st.sidebar.subheader("Column Mapping")
    st.sidebar.caption("Confirm the detected columns. Select 'None' if not in your data.")

    options = ["None"] + df_columns
    confirmed = {}

    for canonical, detected_col in detected.items():
        label = canonical.replace("_", " ").title()
        default_idx = options.index(detected_col) if detected_col in options else 0
        selected = st.sidebar.selectbox(
            label,
            options=options,
            index=default_idx,
            key=f"col_map_{canonical}",
        )
        confirmed[canonical] = selected if selected != "None" else None

    return confirmed


def load_excel(uploaded_file) -> tuple[pd.DataFrame | None, dict[str, str | None]]:
    """
    Load an uploaded Excel file. Handles sheet selection if multiple sheets exist.
    Returns (dataframe, confirmed_column_mapping) or (None, {}) on failure.
    """
    file_size = getattr(uploaded_file, "size", None) or len(uploaded_file.getvalue())
    if file_size > 10 * 1024 * 1024:
        st.warning("File is larger than 10 MB. Large files may slow down parsing.")

    try:
        xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(f"Could not open Excel file: {e}")
        return None, {}

    sheet_names = xls.sheet_names

    if len(sheet_names) > 1:
        selected_sheet = st.sidebar.selectbox(
            "Select sheet",
            options=sheet_names,
            key="sheet_selector",
        )
    else:
        selected_sheet = sheet_names[0]

    try:
        df = pd.read_excel(xls, sheet_name=selected_sheet)
    except Exception as e:
        st.error(f"Could not read sheet '{selected_sheet}': {e}")
        return None, {}

    if df.empty:
        st.error("The selected sheet appears to be empty.")
        return None, {}

    detected = detect_column_mapping(list(df.columns))
    confirmed = render_column_mapper(list(df.columns), detected)

    # Validate minimum requirements
    if not confirmed.get("date"):
        st.sidebar.error("A Date column is required. Please map it above.")
        return None, confirmed

    has_useful = any(confirmed.get(f) for f in USEFUL_COLUMNS)
    if not has_useful:
        st.sidebar.warning(
            "Neither a Sleep Score nor Total Sleep column was detected. "
            "Some charts and AI insights will be limited."
        )

    return df, confirmed
