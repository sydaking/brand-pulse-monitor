import json
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
from transformers import pipeline


# -----------------------------
#   DATA LOADING
# -----------------------------

def load_raw_data(path: Path) -> pd.DataFrame:
    """
    Load the raw reviews CSV. Uses a tolerant parser so that
    slightly messy rows are skipped instead of crashing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Could not find raw data at {path}")

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
    """
    Initialize a small DistilBERT sentiment pipeline.
    """
    return pipeline(
        task="sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


def add_sentiment(df: pd.DataFrame, sentiment_pipe) -> pd.DataFrame:
    """
    Run sentiment analysis on the 'text' column and add:
    - sentiment_label (POSITIVE/NEGATIVE)
    - sentiment_score (0-1)

    Long reviews are truncated to avoid the 512-token model limit.
    """
    texts = df["text"].astype(str).tolist()

    results = sentiment_pipe(
        texts,
        truncation=True,
        max_length=512,
    )

    out = df.copy()
    out["sentiment_label"] = [r["label"] for r in results]
    out["sentiment_score"] = [r["score"] for r in results]
    return out


# -----------------------------
#   EMBEDDINGS & TOPICS
# -----------------------------

def init_embedding_model():
    """
    Sentence transformer for semantic embeddings.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")


def cluster_texts(texts: List[str], embedder) -> np.ndarray:
    """
    Create embeddings, reduce dimension, and cluster with KMeans.
    Returns an array of cluster ids for each text.
    """
    if len(texts) < 5:
        # not enough to cluster meaningfully, put everything in topic 0
        return np.zeros(len(texts), dtype=int)

    print("Creating embeddings for topic modeling...")
    embeddings = embedder.encode(texts, show_progress_bar=True)
    X = np.array(embeddings)

    n_samples, n_features = X.shape
    n_components = min(50, n_features, max(2, n_samples - 1))
    print(f"Reducing embeddings to {n_components} dimensions with SVD...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X)

    # simple heuristic for number of clusters
    if n_samples < 80:
        n_clusters = 4
    elif n_samples < 160:
        n_clusters = 5
    else:
        n_clusters = 6

    print(f"Clustering into {n_clusters} topics with KMeans...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_reduced)
    return clusters


# -----------------------------
#   MANUAL GPT LABEL WORKFLOW
# -----------------------------

def export_topic_summaries(df: pd.DataFrame, path: Path, max_examples: int = 8) -> None:
    """
    Write a human-readable summary file with example reviews per topic.
    You can copy/paste each section into ChatGPT to get a topic label.
    """
    lines: List[str] = []
    topic_ids = sorted(df["topic_cluster"].unique())

    for topic_id in topic_ids:
        subset = df[df["topic_cluster"] == topic_id]

        lines.append(f"===== Topic {topic_id} =====")
        lines.append(f"Total reviews: {len(subset)}")

        if "rating" in subset.columns:
            try:
                avg_rating = subset["rating"].astype(float).mean()
                lines.append(f"Average rating: {avg_rating:.2f}")
            except Exception:
                pass

        if "sentiment_label" in subset.columns:
            sent_counts = subset["sentiment_label"].value_counts().to_dict()
            sent_str = ", ".join(f"{k}: {v}" for k, v in sent_counts.items())
            lines.append(f"Sentiment counts: {sent_str}")

        lines.append("Example reviews:")
        for text in subset["text"].astype(str).head(max_examples):
            clean = text.replace("\n", " ").strip()
            lines.append(f"- {clean}")

        lines.append("")  # blank line between topics

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote topic summaries to {path}")


def load_or_create_topic_labels(topic_ids: List[int], path: Path) -> dict:
    """
    If a labels file exists, load it. Otherwise, create a stub file
    with generic labels and return that.
    """
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            labels = json.load(f)
        print(f"Loaded topic labels from {path}")
        return labels

    # create stub labels
    labels = {str(tid): f"Topic {tid}" for tid in topic_ids}
    with path.open("w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print(f"Created stub topic labels file at {path}")
    print("Edit this file to replace the values with better labels, e.g.:")
    print('{ "0": "Fast claim processing", "1": "Premium price hikes", ... }')
    return labels


def apply_topic_labels(df: pd.DataFrame, labels: dict) -> pd.DataFrame:
    """
    Map numeric topic_cluster ids to human-readable topic_name
    using a labels dict: { "0": "Some label", ... }.
    """
    out = df.copy()
    out["topic_name"] = out["topic_cluster"].map(
        lambda tid: labels.get(str(tid), f"Topic {tid}")
    )
    return out


# -----------------------------
#   SAVE & MAIN
# -----------------------------

def save_processed_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved processed data to {path}")


def main():
    project_root = Path(__file__).resolve().parent.parent
    raw_path = project_root / "data" / "raw_reviews.csv"
    processed_path = project_root / "data" / "processed_reviews.csv"
    summaries_path = project_root / "data" / "topic_summaries.txt"
    labels_path = project_root / "data" / "topic_labels.json"

    print(f"Loading raw data from {raw_path}...")
    df = load_raw_data(raw_path)
    print(f"Loaded {len(df)} rows.")

    if "text" not in df.columns:
        raise ValueError("Expected a 'text' column in raw_reviews.csv")

    print("Initializing sentiment model...")
    sentiment_pipe = init_sentiment_model()

    print("Running sentiment analysis...")
    df = add_sentiment(df, sentiment_pipe)

    print("Initializing embedding model...")
    embedder = init_embedding_model()

    print("Clustering reviews into topics...")
    clusters = cluster_texts(df["text"].astype(str).tolist(), embedder)
    df["topic_cluster"] = clusters

    # Export summaries for manual GPT labeling
    export_topic_summaries(df, summaries_path)

    # Load or create labels mapping
    topic_ids = sorted(df["topic_cluster"].unique())
    labels = load_or_create_topic_labels(topic_ids, labels_path)

    # Apply labels to dataframe
    df = apply_topic_labels(df, labels)

    print("Sample with sentiment & topics:")
    print(df[["id", "rating", "sentiment_label", "topic_cluster", "topic_name"]].head())

    save_processed_data(df, processed_path)


if __name__ == "__main__":
    main()


