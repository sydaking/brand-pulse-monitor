import pandas as pd
import numpy as np
from pathlib import Path
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from collections import Counter


# -----------------------------
#   LOAD RAW DATA
# -----------------------------

def load_raw_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find raw data at {path}")

    # More tolerant CSV reader – skips totally broken rows
    df = pd.read_csv(
        path,
        engine="python",
        on_bad_lines="warn",
    )
    return df


# -----------------------------
#   SENTIMENT ANALYSIS
# -----------------------------

def init_sentiment_model():
    # DistilBERT sentiment model (free, from Hugging Face)
    return pipeline(
        task="sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


def add_sentiment(df: pd.DataFrame, sentiment_pipe) -> pd.DataFrame:
    """
    Run sentiment analysis on the 'text' column and add:
    - sentiment_label (POSITIVE/NEGATIVE)
    - sentiment_score (0-1)

    We truncate long reviews to avoid the 512-token limit crash.
    """
    texts = df["text"].astype(str).tolist()

    results = sentiment_pipe(
        texts,
        truncation=True,
        max_length=512,
    )

    df = df.copy()
    df["sentiment_label"] = [r["label"] for r in results]
    df["sentiment_score"] = [r["score"] for r in results]
    return df


# -----------------------------
#   TOPIC MODELING (BERT-ish)
# -----------------------------

def init_embedding_model():
    # MiniLM = small, fast, surprisingly powerful
    return SentenceTransformer("all-MiniLM-L6-v2")


# Custom extra stopwords you’ve been adding
CUSTOM_STOPWORDS = {
    "embrace", "pet", "pets", "insurance", "company",
    "has", "been", "also", "just", "really", "very",
    "claim", "claims", "policy", "policies",
    "dog", "dogs", "cat", "cats", "puppy", "kitten",
    "year", "years", "month", "months",
}

ALL_STOPWORDS = ENGLISH_STOP_WORDS.union(CUSTOM_STOPWORDS)


def add_topics(df: pd.DataFrame, embedder) -> pd.DataFrame:
    """
    1) Create semantic embeddings for each review
    2) Reduce dimensions with TruncatedSVD
    3) Cluster with KMeans
    4) Generate topic names using class-based keyword extraction
    """

    df = df.copy()
    texts = df["text"].astype(str).tolist()

    if len(df) < 5:
        # Too few reviews to cluster meaningfully
        df["topic_cluster"] = 0
        df["topic_name"] = "All feedback"
        return df

    # 1) Embeddings
    print("Creating embeddings for topic modeling...")
    embeddings = embedder.encode(texts, show_progress_bar=True)
    X = np.array(embeddings)

    # 2) Dimensionality reduction (like a light UMAP)
    n_samples = X.shape[0]
    n_components = min(50, X.shape[1], max(2, n_samples - 1))
    print(f"Reducing embeddings to {n_components} dimensions with SVD...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X)

    # 3) KMeans clustering
    # Choose number of topics based on dataset size
    # e.g., ~1 topic per 25 reviews, capped between 3 and 12
    suggested_k = max(3, min(12, n_samples // 25))
    n_clusters = max(3, suggested_k)

    print(f"Clustering into {n_clusters} topics with KMeans...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_reduced)
    df["topic_cluster"] = clusters

    # 4) Class-based keyword extraction per topic
    print("Extracting keywords for topic names...")

    # One big document per topic
    docs_per_topic = []
    for topic_id in range(n_clusters):
        docs = df.loc[df["topic_cluster"] == topic_id, "text"].astype(str)
        if docs.empty:
            docs_per_topic.append("")
        else:
            docs_per_topic.append(" ".join(docs))

    # Vectorize big docs
    vectorizer = CountVectorizer(
        stop_words=list(ALL_STOPWORDS),
        max_features=2000,
        ngram_range=(1, 2),  # allow 1-2 word phrases
    )
    doc_term_matrix = vectorizer.fit_transform(docs_per_topic)
    terms = np.array(vectorizer.get_feature_names_out())

    topic_names = {}
    for topic_id in range(n_clusters):
        row = doc_term_matrix[topic_id].toarray().ravel()

        if row.sum() == 0:
            topic_names[topic_id] = f"Topic {topic_id}"
            continue

        # Take top 3 words/phrases for the topic
        top_indices = row.argsort()[::-1][:3]
        top_terms = [terms[i] for i in top_indices]

        # Make it a little nicer to read
        topic_label = " / ".join(top_terms)
        topic_names[topic_id] = topic_label

    df["topic_name"] = df["topic_cluster"].map(topic_names)

    print("Topic names:")
    print(df[["topic_cluster", "topic_name"]].drop_duplicates().sort_values("topic_cluster"))

    return df


# -----------------------------
#   SAVE / MAIN
# -----------------------------

def save_processed_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved processed data to {path}")


def main():
    project_root = Path(__file__).resolve().parent.parent
    raw_path = project_root / "data" / "raw_reviews.csv"
    processed_path = project_root / "data" / "processed_reviews.csv"

    print(f"Loading raw data from {raw_path}...")
    df = load_raw_data(raw_path)
    print(f"Loaded {len(df)} rows.")

    print("Initializing sentiment model...")
    sentiment_pipe = init_sentiment_model()

    print("Running sentiment analysis...")
    df = add_sentiment(df, sentiment_pipe)

    print("Initializing embedding model...")
    embedder = init_embedding_model()

    print("Adding topics (semantic embeddings + SVD + KMeans)...")
    df = add_topics(df, embedder)

    print("Sample with sentiment & topics:")
    print(df[["id", "rating", "sentiment_label", "topic_cluster", "topic_name"]].head())

    save_processed_data(df, processed_path)


if __name__ == "__main__":
    main()

