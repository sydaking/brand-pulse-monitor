import streamlit as st
import pandas as pd
from pathlib import Path
import altair as alt


st.set_page_config(page_title="Brand Pulse Monitor", layout="wide")

# Title
st.title("📊 Brand Pulse Monitor")
st.write("Sentiment & topic analysis on customer feedback.")


# -----------------------
# Load Processed Data
# -----------------------

project_root = Path(__file__).resolve().parent.parent
processed_path = project_root / "data" / "processed_reviews.csv"

@st.cache_data
def load_data():
    if not processed_path.exists():
        st.error("Processed data not found. Run the pipeline first!")
        return pd.DataFrame()
    return pd.read_csv(processed_path)

df = load_data()

if df.empty:
    st.stop()
# Normalize source names to avoid duplicates like 'google_play ' vs 'google_play'
if "source" in df.columns:
    df["source"] = (
        df["source"]
        .astype(str)      # make sure it's string
        .str.strip()      # remove leading/trailing spaces
        .str.lower()      # make everything lowercase
        .replace({
            "trustpilot": "trust_pilot",
            "trust pilot": "trust_pilot",
            "trust_pilot": "trust_pilot",
            "app store": "app_store",
            "app_store": "app_store",
            "google play": "google_play",
            "google_play": "google_play",
            "reddit": "reddit",
        })
    )
SEVERITY_ORDER = ["Severe", "Moderate", "Mild", "Praise"]
SEVERITY_WEIGHTS = {
    "Severe": -2,
    "Moderate": -1,
    "Mild": 1,
    "Praise": 2,
}
SEVERITY_COLORS = {
    "Severe": "#d73027",    # strong red
    "Moderate": "#fc8d59",  # orange
    "Mild": "#fee08b",      # yellow
    "Praise": "#1a9850",    # green
}

if "severity" in df.columns:
    # Make severity a categorical type with order for nicer sorting
    df["severity"] = pd.Categorical(
        df["severity"],
        categories=SEVERITY_ORDER,
        ordered=True,
    )

# Ensure correct data types
if "rating" in df.columns:
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

if "created_at" in df.columns:
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")


# -----------------------
# Sidebar Filters
# -----------------------

st.sidebar.header("🔍 Filters")

sentiment_filter = st.sidebar.multiselect(
    "Sentiment",
    options=df["sentiment_label"].unique().tolist(),
    default=df["sentiment_label"].unique().tolist(),
)

topic_filter = st.sidebar.multiselect(
    "Topic",
    options=df["topic_name"].unique().tolist(),
    default=df["topic_name"].unique().tolist(),
)

source_filter = st.sidebar.multiselect(
    "Source",
    options=df["source"].unique().tolist(),
    default=df["source"].unique().tolist(),
)

search_text = st.sidebar.text_input("Search Text")

# Severity filter (if available)
if "severity" in df.columns:
    available_severity = [
        s for s in SEVERITY_ORDER if s in df["severity"].dropna().unique()
    ]
    selected_severity = st.sidebar.multiselect(
        "Severity",
        options=available_severity,
        default=available_severity,
    )
else:
    selected_severity = None


# -----------------------
# Apply Filters
# -----------------------

filtered_df = df[
    (df["sentiment_label"].isin(sentiment_filter)) &
    (df["topic_name"].isin(topic_filter)) &
    (df["source"].isin(source_filter))
]
# Apply severity filter if available
if selected_severity is not None:
    filtered_df = filtered_df[filtered_df["severity"].isin(selected_severity)]
if search_text:
    filtered_df = filtered_df[filtered_df["text"].str.contains(search_text, case=False)]


# -----------------------
# Summary Metrics
# -----------------------

st.subheader("📌 Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Feedback", len(filtered_df))
col2.metric("Positive", (filtered_df["sentiment_label"] == "POSITIVE").sum())
col3.metric("Negative", (filtered_df["sentiment_label"] == "NEGATIVE").sum())

# Brand health metric (overall on current filtered data)
score_df = filtered_df.copy()

# Map sentiment to +1 / -1
score_df["sentiment_numeric"] = score_df["sentiment_label"].map(
    {"POSITIVE": 1, "NEGATIVE": -1}
)

# Add severity weight if available
if "severity" in score_df.columns:
    mapped = score_df["severity"].map(SEVERITY_WEIGHTS)
    score_df["severity_weight"] = pd.to_numeric(mapped, errors="coerce").fillna(0)
else:
    score_df["severity_weight"] = 0


# Brand health score = sentiment + severity weight
score_df["brand_health_score"] = score_df["sentiment_numeric"] + score_df["severity_weight"]

if not score_df["brand_health_score"].dropna().empty:
    brand_health_value = score_df["brand_health_score"].mean()
    st.metric("Brand Health (overall)", f"{brand_health_value:.2f}")
else:
    st.metric("Brand Health (overall)", "N/A")

# -----------------------
# Source Distribution
# -----------------------

if "source" in filtered_df.columns and not filtered_df.empty:
    st.subheader("🌐 Feedback by Source")

    source_counts = filtered_df["source"].value_counts()
    st.bar_chart(source_counts)

# -----------------------
# Severity Distribution
# -----------------------

if "severity" in filtered_df.columns and not filtered_df.empty:
    st.subheader("🚨 Severity Distribution")

    severity_counts = (
        filtered_df["severity"]
        .value_counts()
        .reindex(SEVERITY_ORDER)
        .dropna()
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.bar_chart(severity_counts)

    with col_b:
        st.write(severity_counts.to_frame("count"))

# -----------------------
# Topic Distribution Chart
# -----------------------

st.subheader("📈 Topic Distribution")

topic_counts = filtered_df["topic_name"].value_counts()
st.bar_chart(topic_counts)


# -----------------------
# Sentiment by Topic
# -----------------------

st.subheader("📊 Sentiment by Topic")

sent_topic_pivot = (
    filtered_df
    .groupby(["topic_name", "sentiment_label"])
    .size()
    .unstack(fill_value=0)
)

st.bar_chart(sent_topic_pivot)


# -----------------------
# Severity by Topic
# -----------------------

if "severity" in filtered_df.columns:
    st.subheader("🚦 Severity by Topic")

    sev_topic_pivot = (
        filtered_df
        .dropna(subset=["topic_name", "severity"])
        .groupby(["topic_name", "severity"])
        .size()
        .unstack(fill_value=0)
    )

    # If we have a defined severity order, re-order the columns
    existing_cols = [c for c in SEVERITY_ORDER if c in sev_topic_pivot.columns]
    if existing_cols:
        sev_topic_pivot = sev_topic_pivot[existing_cols]

    st.dataframe(sev_topic_pivot)

# -----------------------
# Severity by Source
# -----------------------

if "severity" in filtered_df.columns and "source" in filtered_df.columns:
    st.subheader("🌍 Severity by Source")

    sev_source_pivot = (
        filtered_df
        .dropna(subset=["source", "severity"])
        .groupby(["source", "severity"])
        .size()
        .unstack(fill_value=0)
    )

    existing_cols_src = [c for c in SEVERITY_ORDER if c in sev_source_pivot.columns]
    if existing_cols_src:
        sev_source_pivot = sev_source_pivot[existing_cols_src]

        sev_long = sev_source_pivot.reset_index().melt(
        id_vars="source",
        var_name="severity",
        value_name="count"
    )

    chart = (
        alt.Chart(sev_long)
        .mark_bar()
        .encode(
            x=alt.X("source:N", title="Source"),
            y=alt.Y("count:Q", title="Number of Reviews"),
            color=alt.Color(
                "severity:N",
                scale=alt.Scale(
                    domain=list(SEVERITY_COLORS.keys()),
                    range=list(SEVERITY_COLORS.values()),
                ),
                legend=alt.Legend(title="Severity"),
            ),
            tooltip=["source", "severity", "count"],
        )
    )

    st.altair_chart(chart, use_container_width=True)


# -----------------------
# Sentiment by Source
# -----------------------

if "source" in filtered_df.columns and not filtered_df.empty:
    st.subheader("💬 Sentiment by Source")

    sent_source_pivot = (
        filtered_df
        .groupby(["source", "sentiment_label"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    st.bar_chart(sent_source_pivot)

# -----------------------
# Severity by Topic
# -----------------------

if "severity" in filtered_df.columns and not filtered_df.empty:
    st.subheader("🧊 Severity by Topic")

    sev_topic = (
        filtered_df
        .groupby(["topic_name", "severity"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=SEVERITY_ORDER, fill_value=0)
        .sort_index()
    )

    st.bar_chart(sev_topic)

# -----------------------
# Sentiment by Rating
# -----------------------

if "rating" in filtered_df.columns:
    st.subheader("⭐ Sentiment by Rating")

    sent_rating_pivot = (
        filtered_df
        .dropna(subset=["rating"])
        .groupby(["rating", "sentiment_label"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    st.bar_chart(sent_rating_pivot)
# -----------------------
# Sentiment Over Time
# -----------------------

if "created_at" in filtered_df.columns:
    st.subheader("⏱️ Average Sentiment Over Time")

    # Map POSITIVE/NEGATIVE to +1 / -1 just for a rough trend
    sentiment_numeric = filtered_df["sentiment_label"].map({"POSITIVE": 1, "NEGATIVE": -1})
    time_df = filtered_df.copy()
    time_df["sentiment_numeric"] = sentiment_numeric

    # Resample by week to smooth things a bit (you can change to 'D' for daily)
    time_df = (
        time_df
        .set_index("created_at")
        .resample("W")["sentiment_numeric"]
        .mean()
        .to_frame(name="avg_sentiment")
    )

    st.line_chart(time_df)
# -----------------------
# Brand Health Over Time
# -----------------------

bh_df = filtered_df.copy()

# Map sentiment to +1 / -1
bh_df["sentiment_numeric"] = bh_df["sentiment_label"].map(
    {"POSITIVE": 1, "NEGATIVE": -1}
)

# Add severity weight if available
if "severity" in bh_df.columns:
    mapped_bh = bh_df["severity"].map(SEVERITY_WEIGHTS)
    bh_df["severity_weight"] = pd.to_numeric(mapped_bh, errors="coerce").fillna(0)
else:
    bh_df["severity_weight"] = 0

bh_df["brand_health_score"] = bh_df["sentiment_numeric"] + bh_df["severity_weight"]

# Clean up dates
bh_df = bh_df.dropna(subset=["created_at"])
bh_df["created_at"] = pd.to_datetime(bh_df["created_at"], errors="coerce")
bh_df = bh_df.dropna(subset=["created_at"])

if not bh_df.empty:
    bh_time = (
        bh_df
        .set_index("created_at")
        .resample("W")["brand_health_score"]
        .mean()
        .to_frame(name="brand_health")
    )

    st.subheader("📉 Brand Health Over Time")
    st.line_chart(bh_time)


# -----------------------
# New / Rising Issues This Week
# -----------------------

if "created_at" in filtered_df.columns and "topic_name" in filtered_df.columns:
    st.subheader("🔥 New / Rising Issues This Week")

    # Work on a copy so we don't mutate filtered_df
    issues_df = filtered_df.copy()

    # Focus on "issues" if severity is available
    if "severity" in issues_df.columns:
        issues_df = issues_df[issues_df["severity"].isin(["Severe", "Moderate"])]

    # Need at least some dated rows
    issues_df = issues_df.dropna(subset=["created_at"])
    if issues_df.empty:
        st.info("No dated feedback available for issue tracking yet.")
    else:
        # Ensure datetime type
        issues_df["created_at"] = pd.to_datetime(issues_df["created_at"], errors="coerce")
        issues_df = issues_df.dropna(subset=["created_at"])

        if issues_df.empty:
            st.info("No valid dates after parsing for issue tracking.")
        else:
            # Determine the "current week" based on the latest date in the filtered set
            max_date = issues_df["created_at"].max().normalize()  # midnight
            # Monday as the start of the week (0=Monday, 6=Sunday)
            current_week_start = max_date - pd.Timedelta(days=max_date.weekday())
            last_week_start = current_week_start - pd.Timedelta(days=7)
            last_week_end = current_week_start - pd.Timedelta(seconds=1)

            # Filter into week buckets
            this_week_mask = (issues_df["created_at"] >= current_week_start) & (issues_df["created_at"] <= max_date)
            last_week_mask = (issues_df["created_at"] >= last_week_start) & (issues_df["created_at"] <= last_week_end)

            this_week_df = issues_df[this_week_mask]
            last_week_df = issues_df[last_week_mask]

            if this_week_df.empty:
                st.info(
                    f"No issues found for the current week window "
                    f"({current_week_start.date()} → {max_date.date()})."
                )
            else:
                this_counts = (
                    this_week_df
                    .groupby("topic_name")
                    .size()
                    .rename("this_week")
                )
                last_counts = (
                    last_week_df
                    .groupby("topic_name")
                    .size()
                    .rename("last_week")
                )

                summary = pd.concat([this_counts, last_counts], axis=1).fillna(0).astype(int)
                summary["change"] = summary["this_week"] - summary["last_week"]

                # Avoid division by zero for pct change
                def pct(row):
                    if row["last_week"] == 0:
                        return None
                    return (row["change"] / row["last_week"]) * 100.0

                summary["pct_change"] = summary.apply(pct, axis=1)

                # New topics: no occurrences last week, some this week
                new_topics = summary[(summary["last_week"] == 0) & (summary["this_week"] > 0)]
                # Rising topics: had some last week, more this week
                rising_topics = summary[
                    (summary["last_week"] > 0) &
                    (summary["this_week"] > summary["last_week"])
                ].sort_values("change", ascending=False)

                st.caption(
                    f"Week window: {current_week_start.date()} → {max_date.date()} "
                    f"(compared to previous week {last_week_start.date()} → {last_week_end.date()})"
                )

                col_new, col_rising = st.columns(2)

                with col_new:
                    st.markdown("**🆕 New issue topics this week**")
                    if new_topics.empty:
                        st.write("No brand-new topics appeared this week.")
                    else:
                        st.dataframe(
                            new_topics[["this_week"]]
                            .sort_values("this_week", ascending=False)
                            .rename(columns={"this_week": "count_this_week"})
                        )

                with col_rising:
                    st.markdown("**📈 Rising issue topics (week-over-week)**")
                    if rising_topics.empty:
                        st.write("No topics increased compared to last week.")
                    else:
                        st.dataframe(
                            rising_topics[["last_week", "this_week", "change", "pct_change"]]
                            .round({"pct_change": 1})
                        )


# -----------------------
# Table of Comments
# -----------------------

st.subheader("📝 Feedback Comments")
st.dataframe(
    filtered_df[
        ["id", "source", "created_at", "text", "sentiment_label", "severity", "topic_name"]
    ]
)
