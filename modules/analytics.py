"""
Pure analytical functions. No Streamlit or Plotly imports here.
All functions take a processed DataFrame and return dicts or DataFrames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ── Helpers ──────────────────────────────────────────────────────────────────

def _col(df: pd.DataFrame, name: str) -> pd.Series | None:
    """Return column if it exists and has at least one non-NaN value."""
    if name in df.columns and df[name].notna().any():
        return df[name]
    return None


def _recent(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty:
        return df
    cutoff = df["date"].max() - pd.Timedelta(days=days)
    return df[df["date"] >= cutoff]


DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ── Public API ────────────────────────────────────────────────────────────────

def compute_summary_stats(df: pd.DataFrame) -> dict:
    """Return summary statistics for all key numeric columns."""
    results = {"n_nights": len(df)}

    if not df.empty:
        results["date_start"] = df["date"].min().strftime("%b %d, %Y")
        results["date_end"] = df["date"].max().strftime("%b %d, %Y")
    else:
        results["date_start"] = results["date_end"] = "N/A"

    for col_name, label in [
        ("sleep_score", "score"),
        ("total_sleep_min", "total_sleep_min"),
        ("deep_sleep_pct", "deep_sleep_pct"),
        ("rem_sleep_pct", "rem_sleep_pct"),
        ("light_sleep_pct", "light_sleep_pct"),
        ("awakenings", "awakenings"),
        ("heart_rate", "heart_rate"),
        ("hrv", "hrv"),
    ]:
        s = _col(df, col_name)
        if s is not None:
            results[f"{label}_mean"] = round(s.mean(), 1)
            results[f"{label}_median"] = round(s.median(), 1)
            results[f"{label}_std"] = round(s.std(), 1)
            results[f"{label}_min"] = round(s.min(), 1)
            results[f"{label}_max"] = round(s.max(), 1)

    # Nights above 80 score
    score = _col(df, "sleep_score")
    if score is not None:
        above_80 = (score >= 80).sum()
        results["nights_above_80"] = int(above_80)
        results["nights_above_80_pct"] = round(above_80 / len(df) * 100, 1)

    return results


def detect_day_of_week_patterns(df: pd.DataFrame) -> dict:
    """Return avg sleep score (and total sleep) by day of week."""
    result = {}
    for metric, col_name in [("score", "sleep_score"), ("total_sleep_min", "total_sleep_min")]:
        s = _col(df, col_name)
        if s is None:
            continue
        tmp = df.copy()
        tmp["_metric"] = s
        avg_by_dow = (
            tmp.groupby("day_of_week")["_metric"]
            .mean()
            .reindex(DOW_ORDER)
            .dropna()
        )
        if avg_by_dow.empty:
            continue
        result[f"{metric}_by_dow"] = avg_by_dow.round(1).to_dict()
        result[f"{metric}_best_day"] = avg_by_dow.idxmax()
        result[f"{metric}_worst_day"] = avg_by_dow.idxmin()
        result[f"{metric}_best_day_avg"] = round(avg_by_dow.max(), 1)
        result[f"{metric}_worst_day_avg"] = round(avg_by_dow.min(), 1)
    return result


def compute_streaks(df: pd.DataFrame, threshold: float = 80.0) -> dict:
    """Compute streaks above/below a sleep score threshold."""
    score = _col(df, "sleep_score")
    if score is None:
        return {}

    above = (score >= threshold).values

    def longest_streak(arr: np.ndarray) -> int:
        if not arr.any():
            return 0
        lengths = []
        cur = 0
        for v in arr:
            if v:
                cur += 1
                lengths.append(cur)
            else:
                cur = 0
        return max(lengths) if lengths else 0

    # Current streak (from end of data)
    current = 0
    for v in reversed(above):
        if v:
            current += 1
        else:
            break

    result = {
        "threshold": threshold,
        "longest_good_streak": longest_streak(above),
        "longest_poor_streak": longest_streak(~above),
        "current_streak_above": current,
    }
    return result


def detect_best_worst_periods(df: pd.DataFrame, window: int = 7) -> dict:
    """Find the best and worst consecutive-window-day periods by avg sleep score."""
    score = _col(df, "sleep_score")
    if score is None or len(df) < window:
        return {}

    rolling_avg = score.rolling(window=window, min_periods=window).mean()
    if rolling_avg.dropna().empty:
        return {}

    best_idx = rolling_avg.idxmax()
    worst_idx = rolling_avg.idxmin()

    return {
        "window_days": window,
        "best_period_end": df.loc[best_idx, "date"].strftime("%b %d, %Y"),
        "best_period_avg": round(rolling_avg[best_idx], 1),
        "worst_period_end": df.loc[worst_idx, "date"].strftime("%b %d, %Y"),
        "worst_period_avg": round(rolling_avg[worst_idx], 1),
    }


def compute_bedtime_consistency(df: pd.DataFrame) -> dict:
    """Std dev of bedtime in minutes (lower = more consistent)."""
    bedtime = _col(df, "bedtime_min")
    if bedtime is None:
        return {}
    std = bedtime.std()
    return {
        "bedtime_std_min": round(std, 1),
        "bedtime_mean_min": round(bedtime.mean(), 1),
        "consistency_rating": (
            "very consistent" if std < 15
            else "consistent" if std < 30
            else "somewhat inconsistent" if std < 60
            else "inconsistent"
        ),
    }


def detect_trends(df: pd.DataFrame) -> dict:
    """Linear regression slope on sleep score over last 30 and 90 days."""
    result = {}
    score = _col(df, "sleep_score")
    if score is None:
        return result

    for days, label in [(30, "30d"), (90, "90d")]:
        subset = _recent(df, days)
        s = _col(subset, "sleep_score")
        if s is None or s.dropna().count() < 5:
            continue
        x = np.arange(len(s))
        valid = s.notna()
        if valid.sum() < 5:
            continue
        slope, _, r_value, p_value, _ = stats.linregress(x[valid], s[valid])
        result[f"slope_{label}"] = round(slope, 3)  # points per night
        result[f"r2_{label}"] = round(r_value ** 2, 3)
        result[f"trend_{label}"] = (
            "improving" if slope > 0.05
            else "declining" if slope < -0.05
            else "stable"
        )

    return result


def compute_stage_ratios(df: pd.DataFrame) -> dict:
    """Avg % per sleep stage vs healthy benchmarks."""
    BENCHMARKS = {
        "deep_sleep_pct": (13, 23),
        "rem_sleep_pct": (20, 25),
        "light_sleep_pct": (50, 60),
    }
    result = {}
    for col_name, (low, high) in BENCHMARKS.items():
        s = _col(df, col_name)
        if s is None:
            continue
        avg = round(s.mean(), 1)
        result[col_name.replace("_pct", "_avg_pct")] = avg
        result[col_name.replace("_pct", "_benchmark")] = f"{low}-{high}%"
        result[col_name.replace("_pct", "_status")] = (
            "below target" if avg < low
            else "above target" if avg > high
            else "healthy"
        )
    return result


def find_correlations(df: pd.DataFrame) -> dict:
    """Pearson correlations between key pairs."""
    pairs = [
        ("bedtime_min", "sleep_score", "bedtime_vs_score"),
        ("total_sleep_min", "sleep_score", "total_sleep_vs_score"),
        ("awakenings", "sleep_score", "awakenings_vs_score"),
        ("hrv", "sleep_score", "hrv_vs_score"),
    ]
    result = {}
    for col_a, col_b, label in pairs:
        a = _col(df, col_a)
        b = _col(df, col_b)
        if a is None or b is None:
            continue
        combined = pd.DataFrame({"a": a, "b": b}).dropna()
        if len(combined) < 5:
            continue
        r, p = stats.pearsonr(combined["a"], combined["b"])
        result[label] = round(r, 3)
        result[f"{label}_p"] = round(p, 4)
    return result


def build_full_summary(df: pd.DataFrame, scope_days: int | None = None) -> dict:
    """Aggregate all analytics into a single dict for the AI prompt."""
    if scope_days:
        df = _recent(df, scope_days)

    return {
        "summary": compute_summary_stats(df),
        "dow_patterns": detect_day_of_week_patterns(df),
        "streaks": compute_streaks(df),
        "best_worst_periods": detect_best_worst_periods(df),
        "bedtime_consistency": compute_bedtime_consistency(df),
        "trends": detect_trends(df),
        "stage_ratios": compute_stage_ratios(df),
        "correlations": find_correlations(df),
    }
