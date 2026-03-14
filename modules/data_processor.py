from __future__ import annotations

import numpy as np
import pandas as pd


def _parse_dates(series: pd.Series) -> pd.Series:
    """Try multiple date formats, return a datetime Series."""
    formats = [
        "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y",
        "%b %d, %Y", "%B %d, %Y", "%d-%b-%Y",
        "%m-%d-%Y", "%Y/%m/%d",
    ]
    for fmt in formats:
        try:
            parsed = pd.to_datetime(series, format=fmt, errors="coerce")
            if parsed.notna().sum() > len(series) * 0.5:
                return parsed
        except Exception:
            continue
    # Last resort: pandas infer
    return pd.to_datetime(series, infer_datetime_format=True, errors="coerce")


def _parse_time_to_minutes(series: pd.Series) -> pd.Series:
    """
    Convert time column to minutes past midnight.
    Handles: '11:30 PM', '23:30', datetime objects, time objects.
    Bedtimes after 6 PM that are actually the prior night are kept as-is;
    caller adjusts midnight crossing if needed.
    """
    def to_minutes(val):
        if pd.isna(val):
            return np.nan
        if hasattr(val, "hour"):
            return val.hour * 60 + val.minute
        s = str(val).strip()
        # Handle HH:MM AM/PM
        for fmt in ["%I:%M %p", "%I:%M%p", "%H:%M", "%H:%M:%S"]:
            try:
                import datetime
                t = datetime.datetime.strptime(s, fmt)
                return t.hour * 60 + t.minute
            except ValueError:
                continue
        return np.nan

    return series.apply(to_minutes)


def _detect_sleep_stage_unit(series: pd.Series) -> str:
    """Return 'minutes' or 'percent' based on value range."""
    max_val = series.dropna().max()
    if max_val <= 1.0:
        return "fraction"
    if max_val <= 100:
        return "percent"
    return "minutes"


def process(df: pd.DataFrame, col_map: dict[str, str | None]) -> pd.DataFrame:
    """
    Clean and enrich the raw dataframe.
    Returns a new dataframe with canonical column names and derived features.
    """
    out = pd.DataFrame()

    # --- Date ---
    date_col = col_map.get("date")
    if not date_col or date_col not in df.columns:
        raise ValueError("No date column found.")
    parsed_dates = _parse_dates(df[date_col])
    valid_mask = parsed_dates.notna()
    df = df[valid_mask].sort_values(by=date_col).reset_index(drop=True)
    out["date"] = _parse_dates(df[date_col]).reset_index(drop=True)

    # --- Sleep score ---
    if col_map.get("sleep_score") and col_map["sleep_score"] in df.columns:
        raw = pd.to_numeric(df[col_map["sleep_score"]], errors="coerce")
        # Normalise fractions to 0-100
        if raw.dropna().max() <= 1.0:
            raw = raw * 100
        out["sleep_score"] = raw.values

    # --- Total sleep (store in minutes) ---
    if col_map.get("total_sleep") and col_map["total_sleep"] in df.columns:
        raw = pd.to_numeric(df[col_map["total_sleep"]], errors="coerce")
        # Detect hours vs minutes: if median < 16 assume hours
        if raw.dropna().median() < 16:
            raw = raw * 60
        out["total_sleep_min"] = raw.values

    # --- Bedtime / Wake time ---
    if col_map.get("bedtime") and col_map["bedtime"] in df.columns:
        out["bedtime_min"] = _parse_time_to_minutes(df[col_map["bedtime"]]).values

    if col_map.get("wake_time") and col_map["wake_time"] in df.columns:
        out["wake_time_min"] = _parse_time_to_minutes(df[col_map["wake_time"]]).values

    # Fix midnight-crossing bedtimes: if bedtime > wake_time, shift bedtime back by 1440
    if "bedtime_min" in out.columns and "wake_time_min" in out.columns:
        mask = out["bedtime_min"] > out["wake_time_min"]
        out.loc[mask, "bedtime_min"] = out.loc[mask, "bedtime_min"] - 1440

    # --- Sleep stages ---
    stage_cols = ["deep_sleep", "rem_sleep", "light_sleep"]
    for stage in stage_cols:
        mapped = col_map.get(stage)
        if mapped and mapped in df.columns:
            raw = pd.to_numeric(df[mapped], errors="coerce")
            unit = _detect_sleep_stage_unit(raw)
            if unit == "fraction":
                raw = raw * 100  # store as percent
            elif unit == "minutes":
                # Convert to percent of total sleep if available
                if "total_sleep_min" in out.columns:
                    pct = (raw.values / out["total_sleep_min"]) * 100
                    out[f"{stage}_pct"] = pct
                out[f"{stage}_min"] = raw.values
                continue
            out[f"{stage}_pct"] = raw.values

    # --- Other numeric metrics ---
    for field in ["awakenings", "heart_rate", "hrv", "respiratory_rate"]:
        mapped = col_map.get(field)
        if mapped and mapped in df.columns:
            out[field] = pd.to_numeric(df[mapped], errors="coerce").values

    # --- Derived features ---
    out["day_of_week"] = out["date"].dt.day_name()
    out["day_of_week_num"] = out["date"].dt.dayofweek  # 0=Monday
    out["week_number"] = out["date"].dt.isocalendar().week.astype(int)
    out["month"] = out["date"].dt.to_period("M").astype(str)
    out["is_weekend"] = out["day_of_week_num"] >= 5

    if "sleep_score" in out.columns and out["sleep_score"].notna().sum() >= 7:
        out["score_7day_avg"] = (
            out["sleep_score"]
            .rolling(window=7, min_periods=3)
            .mean()
        )

    if "total_sleep_min" in out.columns and out["total_sleep_min"].notna().sum() >= 7:
        out["sleep_min_7day_avg"] = (
            out["total_sleep_min"]
            .rolling(window=7, min_periods=3)
            .mean()
        )

    # Outlier flag (for display purposes, not dropped)
    if "total_sleep_min" in out.columns:
        out["is_outlier"] = (out["total_sleep_min"] < 120) | (out["total_sleep_min"] > 960)

    return out


def format_minutes(minutes: float) -> str:
    """Format a minute value as 'Xh Ym'."""
    if pd.isna(minutes):
        return "N/A"
    h = int(minutes) // 60
    m = int(minutes) % 60
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m"
