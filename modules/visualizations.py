"""
All Plotly chart functions. Each returns a plotly Figure.
No Streamlit calls here — charts are rendered in app.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data_processor import format_minutes

TEMPLATE = "plotly_dark"
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
STAGE_COLORS = {
    "deep_sleep_min": "#4C6EF5",
    "rem_sleep_min": "#7950F2",
    "light_sleep_min": "#74C0FC",
    "deep_sleep_pct": "#4C6EF5",
    "rem_sleep_pct": "#7950F2",
    "light_sleep_pct": "#74C0FC",
}


def _col_exists(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and df[col].notna().any()


# ── Overview ──────────────────────────────────────────────────────────────────

def score_timeline(df: pd.DataFrame) -> go.Figure:
    """Line chart of sleep score over time with 7-day rolling average."""
    fig = go.Figure()
    if not _col_exists(df, "sleep_score"):
        fig.add_annotation(text="No sleep score data available", showarrow=False)
        return fig

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["sleep_score"],
        mode="lines+markers",
        name="Sleep Score",
        line=dict(color="#74C0FC", width=1.5),
        marker=dict(size=4),
        hovertemplate="%{x|%b %d, %Y}<br>Score: %{y:.0f}<extra></extra>",
    ))

    if _col_exists(df, "score_7day_avg"):
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["score_7day_avg"],
            mode="lines",
            name="7-Day Avg",
            line=dict(color="#FF922B", width=2.5, dash="dash"),
            hovertemplate="%{x|%b %d, %Y}<br>7-Day Avg: %{y:.1f}<extra></extra>",
        ))

    fig.update_layout(
        template=TEMPLATE,
        title="Sleep Score Over Time",
        xaxis_title="Date",
        yaxis_title="Sleep Score",
        yaxis=dict(range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def calendar_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap calendar colored by sleep score."""
    if not _col_exists(df, "sleep_score"):
        fig = go.Figure()
        fig.add_annotation(text="No sleep score data available", showarrow=False)
        return fig

    tmp = df[["date", "sleep_score"]].dropna().copy()
    tmp["week"] = tmp["date"].dt.isocalendar().week.astype(int)
    tmp["year"] = tmp["date"].dt.isocalendar().year.astype(int)
    tmp["year_week"] = tmp["year"].astype(str) + "-" + tmp["week"].astype(str).str.zfill(2)
    tmp["dow"] = tmp["date"].dt.dayofweek  # 0=Mon

    # Build unique week positions sorted chronologically
    week_labels = tmp.sort_values("date")["year_week"].unique()
    week_pos = {yw: i for i, yw in enumerate(week_labels)}
    tmp["week_pos"] = tmp["year_week"].map(week_pos)

    z_matrix = np.full((7, len(week_labels)), np.nan)
    text_matrix = [[""] * len(week_labels) for _ in range(7)]

    for _, row in tmp.iterrows():
        wi = row["week_pos"]
        di = int(row["dow"])
        z_matrix[di][wi] = row["sleep_score"]
        text_matrix[di][wi] = f"{row['date'].strftime('%b %d')}<br>Score: {row['sleep_score']:.0f}"

    # x-tick labels: show month name at first week of each month
    x_labels = []
    last_month = None
    for yw in week_labels:
        yr, wk = yw.split("-")
        d = pd.Timestamp.fromisocalendar(int(yr), int(wk), 1)
        if d.month != last_month:
            x_labels.append(d.strftime("%b %Y"))
            last_month = d.month
        else:
            x_labels.append("")

    fig = go.Figure(go.Heatmap(
        z=z_matrix,
        x=x_labels,
        y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        text=text_matrix,
        hoverinfo="text",
        colorscale="RdYlGn",
        zmin=0, zmax=100,
        showscale=True,
        colorbar=dict(title="Score"),
    ))
    fig.update_layout(
        template=TEMPLATE,
        title="Sleep Score Calendar",
        xaxis=dict(tickangle=-45),
        margin=dict(l=40, r=20, t=50, b=60),
    )
    return fig


def duration_timeline(df: pd.DataFrame) -> go.Figure:
    """Line chart of total sleep duration over time."""
    fig = go.Figure()
    if not _col_exists(df, "total_sleep_min"):
        fig.add_annotation(text="No sleep duration data available", showarrow=False)
        return fig

    hours = df["total_sleep_min"] / 60
    fig.add_trace(go.Scatter(
        x=df["date"], y=hours,
        mode="lines+markers",
        name="Sleep Duration",
        line=dict(color="#69DB7C", width=1.5),
        marker=dict(size=4),
        hovertemplate="%{x|%b %d, %Y}<br>Duration: %{y:.1f}h<extra></extra>",
    ))

    if _col_exists(df, "sleep_min_7day_avg"):
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["sleep_min_7day_avg"] / 60,
            mode="lines",
            name="7-Day Avg",
            line=dict(color="#FF922B", width=2.5, dash="dash"),
            hovertemplate="%{x|%b %d, %Y}<br>7-Day Avg: %{y:.1f}h<extra></extra>",
        ))

    # 8-hour reference line
    fig.add_hline(y=8, line_dash="dot", line_color="gray",
                  annotation_text="8h target", annotation_position="bottom right")

    fig.update_layout(
        template=TEMPLATE,
        title="Sleep Duration Over Time",
        xaxis_title="Date",
        yaxis_title="Hours",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# ── Trends ────────────────────────────────────────────────────────────────────

def sleep_stages_stacked_bar(df: pd.DataFrame, n_days: int = 60) -> go.Figure:
    """Stacked bar chart of sleep stage composition over recent nights."""
    stage_col_pairs = [
        ("deep_sleep_min", "Deep"),
        ("rem_sleep_min", "REM"),
        ("light_sleep_min", "Light"),
    ]
    # Fall back to pct columns
    pct_pairs = [
        ("deep_sleep_pct", "Deep"),
        ("rem_sleep_pct", "REM"),
        ("light_sleep_pct", "Light"),
    ]

    use_pairs = [p for p in stage_col_pairs if _col_exists(df, p[0])]
    unit_label = "Minutes"
    if not use_pairs:
        use_pairs = [p for p in pct_pairs if _col_exists(df, p[0])]
        unit_label = "%"

    if not use_pairs:
        fig = go.Figure()
        fig.add_annotation(text="No sleep stage data available", showarrow=False)
        return fig

    subset = df.tail(n_days).copy()
    fig = go.Figure()
    colors = ["#4C6EF5", "#7950F2", "#74C0FC"]

    for (col, label), color in zip(use_pairs, colors):
        if not _col_exists(subset, col):
            continue
        fig.add_trace(go.Bar(
            x=subset["date"], y=subset[col],
            name=label,
            marker_color=color,
            hovertemplate=f"%{{x|%b %d}}<br>{label}: %{{y:.0f}} {unit_label}<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        template=TEMPLATE,
        title=f"Sleep Stage Composition (Last {n_days} Nights)",
        xaxis_title="Date",
        yaxis_title=unit_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def bedtime_trend(df: pd.DataFrame) -> go.Figure:
    """Line chart of bedtime over time (in hours past midnight)."""
    fig = go.Figure()
    if not _col_exists(df, "bedtime_min"):
        fig.add_annotation(text="No bedtime data available", showarrow=False)
        return fig

    # Convert to hours for readability; negative = before midnight
    hours = df["bedtime_min"] / 60

    def mins_to_label(m: float) -> str:
        if pd.isna(m):
            return ""
        total = int(m)
        h = (total // 60) % 24
        mn = abs(total % 60)
        period = "AM" if h < 12 else "PM"
        h12 = h % 12 or 12
        return f"{h12}:{mn:02d} {period}"

    hover_text = df["bedtime_min"].apply(mins_to_label)

    fig.add_trace(go.Scatter(
        x=df["date"], y=hours,
        mode="lines+markers",
        name="Bedtime",
        line=dict(color="#DA77F2", width=1.5),
        marker=dict(size=4),
        text=hover_text,
        hovertemplate="%{x|%b %d, %Y}<br>Bedtime: %{text}<extra></extra>",
    ))

    fig.update_layout(
        template=TEMPLATE,
        title="Bedtime Trend Over Time",
        xaxis_title="Date",
        yaxis_title="Hours relative to midnight",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def dow_avg_bar(df: pd.DataFrame) -> go.Figure:
    """Bar chart of average sleep score by day of week."""
    if not _col_exists(df, "sleep_score"):
        fig = go.Figure()
        fig.add_annotation(text="No sleep score data available", showarrow=False)
        return fig

    avg = (
        df.groupby("day_of_week")["sleep_score"]
        .mean()
        .reindex(DOW_ORDER)
        .dropna()
    )

    colors = ["#FF6B6B" if v == avg.min() else "#69DB7C" if v == avg.max() else "#74C0FC"
              for v in avg.values]

    fig = go.Figure(go.Bar(
        x=avg.index, y=avg.values,
        marker_color=colors,
        hovertemplate="%{x}<br>Avg Score: %{y:.1f}<extra></extra>",
    ))
    fig.update_layout(
        template=TEMPLATE,
        title="Average Sleep Score by Day of Week",
        xaxis_title="Day",
        yaxis_title="Avg Sleep Score",
        yaxis=dict(range=[max(0, avg.min() - 10), min(105, avg.max() + 10)]),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def monthly_avg_bar(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: average score and sleep duration per month."""
    metrics = []
    if _col_exists(df, "sleep_score"):
        metrics.append(("sleep_score", "Avg Score", "#74C0FC"))
    if not metrics:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False)
        return fig

    grouped = df.groupby("month")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for col, label, color in metrics:
        avg = grouped[col].mean()
        fig.add_trace(go.Bar(
            x=avg.index, y=avg.values,
            name=label, marker_color=color,
            hovertemplate="%{x}<br>" + label + ": %{y:.1f}<extra></extra>",
        ), secondary_y=False)

    if _col_exists(df, "total_sleep_min"):
        avg_hrs = grouped["total_sleep_min"].mean() / 60
        fig.add_trace(go.Scatter(
            x=avg_hrs.index, y=avg_hrs.values,
            name="Avg Duration (hrs)", mode="lines+markers",
            line=dict(color="#FF922B", width=2),
            hovertemplate="%{x}<br>Avg Duration: %{y:.1f}h<extra></extra>",
        ), secondary_y=True)

    fig.update_layout(
        template=TEMPLATE,
        title="Monthly Averages",
        xaxis_title="Month",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=50, b=60),
    )
    fig.update_yaxes(title_text="Avg Score", secondary_y=False)
    fig.update_yaxes(title_text="Avg Duration (hrs)", secondary_y=True)
    return fig


# ── Distributions ─────────────────────────────────────────────────────────────

def score_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogram of sleep scores."""
    if not _col_exists(df, "sleep_score"):
        fig = go.Figure()
        fig.add_annotation(text="No sleep score data", showarrow=False)
        return fig

    scores = df["sleep_score"].dropna()
    fig = go.Figure(go.Histogram(
        x=scores, nbinsx=20, marker_color="#74C0FC",
        hovertemplate="Score: %{x}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=scores.mean(), line_dash="dash", line_color="#FF922B",
                  annotation_text=f"Mean: {scores.mean():.0f}", annotation_position="top right")
    fig.update_layout(
        template=TEMPLATE,
        title="Sleep Score Distribution",
        xaxis_title="Sleep Score",
        yaxis_title="Nights",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def duration_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogram of total sleep duration in hours."""
    if not _col_exists(df, "total_sleep_min"):
        fig = go.Figure()
        fig.add_annotation(text="No duration data", showarrow=False)
        return fig

    hours = df["total_sleep_min"].dropna() / 60
    fig = go.Figure(go.Histogram(
        x=hours, nbinsx=20, marker_color="#69DB7C",
        hovertemplate="Duration: %{x:.1f}h<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=8, line_dash="dot", line_color="gray",
                  annotation_text="8h target", annotation_position="top left")
    fig.add_vline(x=hours.mean(), line_dash="dash", line_color="#FF922B",
                  annotation_text=f"Mean: {hours.mean():.1f}h", annotation_position="top right")
    fig.update_layout(
        template=TEMPLATE,
        title="Sleep Duration Distribution",
        xaxis_title="Hours",
        yaxis_title="Nights",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def score_boxplot_by_dow(df: pd.DataFrame) -> go.Figure:
    """Box plot of sleep score by day of week."""
    if not _col_exists(df, "sleep_score"):
        fig = go.Figure()
        fig.add_annotation(text="No sleep score data", showarrow=False)
        return fig

    fig = go.Figure()
    for day in DOW_ORDER:
        scores = df[df["day_of_week"] == day]["sleep_score"].dropna()
        if scores.empty:
            continue
        fig.add_trace(go.Box(
            y=scores, name=day[:3], marker_color="#74C0FC",
            boxpoints="outliers",
            hovertemplate=f"{day}<br>%{{y:.0f}}<extra></extra>",
        ))

    fig.update_layout(
        template=TEMPLATE,
        title="Sleep Score Spread by Day of Week",
        xaxis_title="Day",
        yaxis_title="Sleep Score",
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=False,
    )
    return fig


def scatter_bedtime_vs_score(df: pd.DataFrame) -> go.Figure:
    """Scatter plot: bedtime (hours) vs sleep score."""
    if not _col_exists(df, "bedtime_min") or not _col_exists(df, "sleep_score"):
        fig = go.Figure()
        fig.add_annotation(text="Bedtime or score data not available", showarrow=False)
        return fig

    tmp = df[["date", "bedtime_min", "sleep_score"]].dropna()
    hours = tmp["bedtime_min"] / 60

    fig = px.scatter(
        tmp, x=hours, y="sleep_score",
        trendline="ols",
        labels={"x": "Bedtime (hrs past midnight)", "sleep_score": "Sleep Score"},
        template=TEMPLATE,
        title="Bedtime vs Sleep Score",
        color_discrete_sequence=["#DA77F2"],
    )
    fig.update_traces(
        marker=dict(size=6, opacity=0.7),
        selector=dict(mode="markers"),
    )
    fig.update_layout(margin=dict(l=40, r=20, t=50, b=40))
    return fig


def scatter_duration_vs_score(df: pd.DataFrame) -> go.Figure:
    """Scatter plot: total sleep hours vs sleep score."""
    if not _col_exists(df, "total_sleep_min") or not _col_exists(df, "sleep_score"):
        fig = go.Figure()
        fig.add_annotation(text="Duration or score data not available", showarrow=False)
        return fig

    tmp = df[["date", "total_sleep_min", "sleep_score"]].dropna()
    hours = tmp["total_sleep_min"] / 60

    fig = px.scatter(
        tmp, x=hours, y="sleep_score",
        trendline="ols",
        labels={"x": "Total Sleep (hours)", "sleep_score": "Sleep Score"},
        template=TEMPLATE,
        title="Sleep Duration vs Sleep Score",
        color_discrete_sequence=["#69DB7C"],
    )
    fig.update_traces(
        marker=dict(size=6, opacity=0.7),
        selector=dict(mode="markers"),
    )
    fig.update_layout(margin=dict(l=40, r=20, t=50, b=40))
    return fig
