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


# -----------------------
# Apply Filters
# -----------------------

filtered_df = df[
    (df["sentiment_label"].isin(sentiment_filter)) &
    (df["topic_name"].isin(topic_filter)) &
    (df["source"].isin(source_filter))
]

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
# Table of Comments
# -----------------------

st.subheader("📝 Feedback Comments")
st.dataframe(
    filtered_df[
        ["id", "source", "created_at", "text", "sentiment_label", "topic_name"]
    ]
)
