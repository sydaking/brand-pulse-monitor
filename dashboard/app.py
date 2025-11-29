import streamlit as st
import pandas as pd
from pathlib import Path

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
SEVERITY_ORDER = ["Severe", "Moderate", "Mild", "Praise"]

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
# Table of Comments
# -----------------------

st.subheader("📝 Feedback Comments")
st.dataframe(
    filtered_df[
        ["id", "source", "created_at", "text", "sentiment_label", "severity", "topic_name"]
    ]
)
