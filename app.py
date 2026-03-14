import streamlit as st
import pandas as pd

from modules.data_loader import load_excel
from modules.data_processor import process, format_minutes
from modules import analytics, visualizations as viz
from modules.ai_insights import generate_insights

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sleep Analyzer",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("Sleep Analyzer")
st.sidebar.markdown("Upload your sleep data Excel file to get started.")

uploaded_file = st.sidebar.file_uploader(
    "Choose an Excel file",
    type=["xlsx", "xls"],
    help="Supports .xlsx and .xls files from Oura, Garmin, Fitbit, WHOOP, or any app that exports to Excel.",
)

# ── Session state ─────────────────────────────────────────────────────────────

if "df_processed" not in st.session_state:
    st.session_state.df_processed = None
if "col_map" not in st.session_state:
    st.session_state.col_map = {}
if "ai_insights_text" not in st.session_state:
    st.session_state.ai_insights_text = None
if "ai_scope" not in st.session_state:
    st.session_state.ai_scope = None

# ── Load and process data ─────────────────────────────────────────────────────

@st.cache_data(show_spinner="Parsing your sleep data...")
def cached_load(file_bytes: bytes, file_name: str):
    """Cache parsed dataframe keyed on file content hash."""
    import io
    return load_excel(io.BytesIO(file_bytes))


@st.cache_data(show_spinner="Processing data...")
def cached_process(df_json: str, col_map_json: str):
    import json
    df = pd.read_json(df_json)
    col_map = json.loads(col_map_json)
    return process(df, col_map)


if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    # Reset AI cache when a new file is uploaded
    file_hash = hash(file_bytes)
    if st.session_state.get("file_hash") != file_hash:
        st.session_state.ai_insights_text = None
        st.session_state.file_hash = file_hash

    # We need to re-run load_excel (not cached) because it renders sidebar widgets
    import io
    df_raw, col_map = load_excel(io.BytesIO(file_bytes))

    if df_raw is not None and col_map.get("date"):
        try:
            import json
            df = process(df_raw, col_map)
            st.session_state.df_processed = df
            st.session_state.col_map = col_map
        except Exception as e:
            st.error(f"Error processing data: {e}")
            st.session_state.df_processed = None

# ── Landing screen ────────────────────────────────────────────────────────────

if st.session_state.df_processed is None:
    st.title("Sleep Analyzer")
    st.markdown("""
Welcome! Upload an Excel file in the sidebar to begin.

**Supported data columns include:**
- Date, Bedtime, Wake time
- Total sleep duration
- Sleep score / quality rating
- Sleep stages (Deep, REM, Light)
- Awakenings, Heart rate, HRV, Respiratory rate

The app will automatically detect your column names — even if they differ from device to device.

---
**Supported exports from:** Oura Ring, Garmin Connect, Fitbit, WHOOP, Apple Health (via third-party export), or any custom Excel spreadsheet.
    """)
    st.stop()

# ── Main content ──────────────────────────────────────────────────────────────

df = st.session_state.df_processed
n = len(df)

st.title("Sleep Analyzer")
if n < 7:
    st.info(f"Only {n} nights of data found. Some features (rolling averages, trend analysis) need at least 7 nights.")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_overview, tab_trends, tab_distributions, tab_ai = st.tabs([
    "Overview", "Trends", "Distributions", "AI Insights"
])

# ── TAB 1: Overview ───────────────────────────────────────────────────────────

with tab_overview:
    stats = analytics.compute_summary_stats(df)

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nights Tracked", stats.get("n_nights", "—"))
    with col2:
        score_mean = stats.get("score_mean")
        st.metric(
            "Avg Sleep Score",
            f"{score_mean:.0f}" if score_mean else "—",
            delta=None,
        )
    with col3:
        dur = stats.get("total_sleep_min_mean")
        st.metric("Avg Sleep Duration", format_minutes(dur) if dur else "—")
    with col4:
        above = stats.get("nights_above_80_pct")
        st.metric("Nights Scoring ≥80", f"{above}%" if above is not None else "—")

    st.markdown("---")

    # Charts
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.plotly_chart(viz.score_timeline(df), use_container_width=True, key="score_timeline")
    with col_right:
        st.plotly_chart(viz.duration_timeline(df), use_container_width=True, key="duration_timeline")

    st.plotly_chart(viz.calendar_heatmap(df), use_container_width=True, key="calendar_heatmap")

# ── TAB 2: Trends ─────────────────────────────────────────────────────────────

with tab_trends:
    trends = analytics.detect_trends(df)
    dow = analytics.detect_day_of_week_patterns(df)
    streaks = analytics.compute_streaks(df)

    # Trend summary cards
    col1, col2, col3 = st.columns(3)
    with col1:
        t30 = trends.get("trend_30d", "N/A")
        slope = trends.get("slope_30d")
        st.metric("30-Day Trend", t30.title() if t30 != "N/A" else "N/A",
                  delta=f"{slope:+.2f} pts/night" if slope else None)
    with col2:
        best_day = dow.get("score_best_day", "—")
        best_avg = dow.get("score_best_day_avg", "")
        st.metric("Best Night of Week", best_day, delta=f"avg {best_avg}" if best_avg else None)
    with col3:
        streak = streaks.get("longest_good_streak", "—")
        current = streaks.get("current_streak_above", 0)
        st.metric("Best Streak (≥80)", f"{streak} nights",
                  delta=f"Current: {current}" if current else None)

    st.markdown("---")

    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(viz.dow_avg_bar(df), use_container_width=True, key="dow_avg_bar")
        st.plotly_chart(viz.bedtime_trend(df), use_container_width=True, key="bedtime_trend")
    with col_right:
        n_days = st.selectbox("Stage chart window", [30, 60, 90], index=1, key="stage_window")
        st.plotly_chart(viz.sleep_stages_stacked_bar(df, n_days=n_days), use_container_width=True, key="sleep_stages_bar")
        st.plotly_chart(viz.monthly_avg_bar(df), use_container_width=True, key="monthly_avg_bar")

# ── TAB 3: Distributions ──────────────────────────────────────────────────────

with tab_distributions:
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(viz.score_histogram(df), use_container_width=True, key="score_histogram")
        st.plotly_chart(viz.scatter_bedtime_vs_score(df), use_container_width=True, key="scatter_bedtime")
    with col_right:
        st.plotly_chart(viz.duration_histogram(df), use_container_width=True, key="duration_histogram")
        st.plotly_chart(viz.scatter_duration_vs_score(df), use_container_width=True, key="scatter_duration")

    st.plotly_chart(viz.score_boxplot_by_dow(df), use_container_width=True, key="score_boxplot_dow")

# ── TAB 4: AI Insights ────────────────────────────────────────────────────────

with tab_ai:
    st.subheader("AI Sleep Insights")
    st.caption("Powered by Claude. Insights are generated from your actual data — not generic advice.")

    # Show which columns were successfully detected
    detected_cols = [c for c in ["sleep_score", "total_sleep_min", "bedtime_min",
                                  "wake_time_min", "deep_sleep_pct", "rem_sleep_pct",
                                  "light_sleep_pct", "awakenings", "heart_rate", "hrv"]
                     if c in df.columns and df[c].notna().any()]
    missing_key = [c for c in ["sleep_score", "total_sleep_min"] if c not in detected_cols]

    if missing_key:
        st.warning(
            f"⚠️ The following important columns were **not mapped**: "
            f"{', '.join(c.replace('_', ' ').replace(' min', '').title() for c in missing_key)}. "
            f"Check the **Column Mapping** section in the sidebar and make sure your Sleep Score "
            f"and/or Sleep Duration columns are assigned. Re-upload the file if needed."
        )

    with st.expander("Columns detected in your data", expanded=False):
        if detected_cols:
            st.success(f"✅ Found {len(detected_cols)} metric column(s): " +
                       ", ".join(c.replace("_", " ").replace(" min", "").replace(" pct", " %").title()
                                 for c in detected_cols))
        else:
            st.error("No metric columns detected beyond date/bedtime. Check column mapping in the sidebar.")

    scope_options = {"Last 30 days": 30, "Last 90 days": 90, "All time": None}
    scope_label = st.selectbox("Analysis scope", list(scope_options.keys()), key="ai_scope_select")
    scope_days = scope_options[scope_label]

    # Check if we need to regenerate (scope changed or no cached result)
    needs_regen = (
        st.session_state.ai_insights_text is None
        or st.session_state.ai_scope != scope_days
    )

    col_gen, col_clear = st.columns([2, 1])
    with col_gen:
        generate_btn = st.button("Generate Insights", type="primary")
    with col_clear:
        if st.button("Clear") and st.session_state.ai_insights_text:
            st.session_state.ai_insights_text = None
            st.rerun()

    if generate_btn or (needs_regen and st.session_state.ai_insights_text is None and generate_btn):
        st.session_state.ai_scope = scope_days
        placeholder = st.empty()
        full_text = ""
        with placeholder.container():
            full_text = st.write_stream(generate_insights(df, scope_days=scope_days))
        st.session_state.ai_insights_text = full_text
    elif st.session_state.ai_insights_text:
        st.markdown(st.session_state.ai_insights_text)
    else:
        st.info("Click **Generate Insights** to get a personalized AI analysis of your sleep patterns.")
