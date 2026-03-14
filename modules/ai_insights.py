"""
Claude API integration for AI-generated sleep insights.
"""

from __future__ import annotations

import os
from typing import Generator

import streamlit as st

from .analytics import build_full_summary
from .data_processor import format_minutes


def _get_api_key() -> str | None:
    """Read API key from Streamlit secrets, then env vars."""
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass
    return os.environ.get("ANTHROPIC_API_KEY")


def _build_prompt(summary: dict, scope_label: str) -> str:
    """Convert analytics summary dict into a structured prompt string."""
    s = summary.get("summary", {})
    dow = summary.get("dow_patterns", {})
    streaks = summary.get("streaks", {})
    bw = summary.get("best_worst_periods", {})
    bedtime = summary.get("bedtime_consistency", {})
    trends = summary.get("trends", {})
    stages = summary.get("stage_ratios", {})
    corr = summary.get("correlations", {})

    lines = [
        f"Here is a statistical summary of my sleep data ({scope_label}):",
        "",
        "OVERVIEW",
        f"- Nights tracked: {s.get('n_nights', 'N/A')}",
        f"- Date range: {s.get('date_start', 'N/A')} to {s.get('date_end', 'N/A')}",
    ]

    if "score_mean" in s:
        lines += [
            f"- Average sleep score: {s['score_mean']} (range: {s.get('score_min')}–{s.get('score_max')})",
            f"- Nights scoring ≥80: {s.get('nights_above_80', 'N/A')} ({s.get('nights_above_80_pct', 'N/A')}%)",
        ]

    if "total_sleep_min_mean" in s:
        lines.append(
            f"- Average sleep duration: {format_minutes(s['total_sleep_min_mean'])} "
            f"(range: {format_minutes(s.get('total_sleep_min_min', 0))}–"
            f"{format_minutes(s.get('total_sleep_min_max', 0))})"
        )

    if stages:
        lines += ["", "SLEEP STAGE AVERAGES"]
        for stage, label in [
            ("deep_sleep_avg_pct", "Deep sleep"),
            ("rem_sleep_avg_pct", "REM sleep"),
            ("light_sleep_avg_pct", "Light sleep"),
        ]:
            if stage in stages:
                bench_key = stage.replace("_avg_pct", "_benchmark")
                status_key = stage.replace("_avg_pct", "_status")
                lines.append(
                    f"- {label}: {stages[stage]}% "
                    f"(healthy target: {stages.get(bench_key, 'N/A')}, "
                    f"status: {stages.get(status_key, 'N/A')})"
                )

    if dow:
        lines += ["", "DAY-OF-WEEK PATTERNS"]
        if "score_best_day" in dow:
            lines.append(
                f"- Best night: {dow['score_best_day']} (avg score: {dow.get('score_best_day_avg')})"
            )
            lines.append(
                f"- Worst night: {dow['score_worst_day']} (avg score: {dow.get('score_worst_day_avg')})"
            )

    if trends:
        lines += ["", "RECENT TRENDS"]
        for period in ["30d", "90d"]:
            if f"trend_{period}" in trends:
                lines.append(
                    f"- {period} trend: {trends[f'trend_{period}']} "
                    f"(slope: {trends.get(f'slope_{period}', 'N/A')} pts/night)"
                )

    if bedtime:
        lines += ["", "BEDTIME CONSISTENCY"]
        lines.append(
            f"- Bedtime variability: {bedtime.get('bedtime_std_min')} min std dev "
            f"({bedtime.get('consistency_rating', 'N/A')})"
        )

    if streaks:
        lines += ["", "STREAKS"]
        lines.append(f"- Longest good streak (score ≥{streaks.get('threshold', 80)}): "
                     f"{streaks.get('longest_good_streak', 'N/A')} consecutive nights")
        lines.append(f"- Current streak: {streaks.get('current_streak_above', 0)} nights")
        lines.append(f"- Longest poor streak: {streaks.get('longest_poor_streak', 'N/A')} nights")

    if bw:
        lines += ["", "BEST/WORST PERIODS"]
        lines.append(f"- Best {bw.get('window_days', 7)}-day period: ending {bw.get('best_period_end')}, "
                     f"avg score {bw.get('best_period_avg')}")
        lines.append(f"- Worst {bw.get('window_days', 7)}-day period: ending {bw.get('worst_period_end')}, "
                     f"avg score {bw.get('worst_period_avg')}")

    if corr:
        lines += ["", "CORRELATIONS (Pearson r)"]
        label_map = {
            "bedtime_vs_score": "Later bedtime → score",
            "total_sleep_vs_score": "More sleep → score",
            "awakenings_vs_score": "More awakenings → score",
            "hrv_vs_score": "Higher HRV → score",
        }
        for key, label in label_map.items():
            if key in corr:
                lines.append(f"- {label}: r = {corr[key]}")

    lines += [
        "",
        "Please provide a thorough analysis and personalized, actionable recommendations based on this data.",
    ]

    return "\n".join(lines)


SYSTEM_PROMPT = """\
You are a sleep health analyst reviewing personal sleep tracking data.
Be specific — reference the actual numbers provided in the data summary.
Write in second person ("Your deep sleep averages...").
Organize your response with these sections:
1. Key Findings
2. Patterns Detected
3. Concerning Trends (if any)
4. Recommendations

Keep the total response under 650 words. Be direct and helpful, not generic.\
"""


def _build_raw_sample(df, scope_days: int | None = None) -> str:
    """
    Build a plain-text table of the most recent rows from the processed dataframe.
    Used as a fallback so Claude always has real data to work with.
    """
    import pandas as pd

    if scope_days:
        cutoff = df["date"].max() - pd.Timedelta(days=scope_days)
        df = df[df["date"] >= cutoff]

    # Pick only columns that have data
    useful = [c for c in ["date", "sleep_score", "total_sleep_min", "bedtime_min",
                           "wake_time_min", "deep_sleep_pct", "rem_sleep_pct",
                           "light_sleep_pct", "awakenings", "heart_rate", "hrv"]
              if c in df.columns and df[c].notna().any()]

    if not useful:
        return ""

    sample = df[useful].tail(30).copy()

    # Make it readable: convert minutes to hours where sensible
    if "total_sleep_min" in sample.columns:
        sample["total_sleep_hrs"] = (sample["total_sleep_min"] / 60).round(2)
        sample = sample.drop(columns=["total_sleep_min"])
    if "bedtime_min" in sample.columns:
        sample = sample.drop(columns=["bedtime_min"])
    if "wake_time_min" in sample.columns:
        sample = sample.drop(columns=["wake_time_min"])

    return sample.to_string(index=False)


def generate_insights(df, scope_days: int | None = None) -> Generator[str, None, None]:
    """
    Stream AI insights as a generator of text chunks.
    Caller should use st.write_stream() with this generator.
    """
    import anthropic

    api_key = _get_api_key()
    if not api_key:
        yield (
            "**API key not configured.**\n\n"
            "To enable AI insights:\n"
            "- **Local:** add `ANTHROPIC_API_KEY = 'sk-ant-...'` to `.streamlit/secrets.toml`\n"
            "- **Streamlit Cloud:** go to your app's Settings → Secrets and add the key there\n\n"
            "Get an API key at https://console.anthropic.com"
        )
        return

    scope_label = (
        f"last {scope_days} days" if scope_days else "all available data"
    )

    summary = build_full_summary(df, scope_days=scope_days)
    stats_prompt = _build_prompt(summary, scope_label)

    # Always append a raw sample so Claude has real numbers even if stats are sparse
    raw_sample = _build_raw_sample(df, scope_days=scope_days)
    if raw_sample:
        user_message = (
            stats_prompt.rstrip()
            + "\n\nRAW DATA SAMPLE (most recent nights):\n"
            + raw_sample
            + "\n\nPlease use both the statistics above AND the raw data sample to provide a thorough analysis."
        )
    else:
        user_message = stats_prompt

    client = anthropic.Anthropic(api_key=api_key)

    try:
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                yield text
    except anthropic.APIConnectionError:
        yield "**Connection error.** Could not reach the Anthropic API. Check your internet connection."
    except anthropic.RateLimitError:
        yield "**Rate limit reached.** Please wait a moment and try again."
    except anthropic.APIStatusError as e:
        yield f"**API error {e.status_code}:** {e.message}"
    except Exception as e:
        yield f"**Unexpected error:** {e}"
