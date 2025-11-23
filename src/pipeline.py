import pandas as pd
from pathlib import Path
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import Counter



def load_raw_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find raw data at {path}")

    # Use the Python engine and be forgiving with slightly messy CSV rows
    df = pd.read_csv(
        path,
        engine="python",       # more tolerant than the default C engine
        on_bad_lines="warn"    # skip lines that are totally broken
    )
    return df



def init_sentiment_model():
    return pipeline(
        task="sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


def add_sentiment(df: pd.DataFrame, sentiment_pipe) -> pd.DataFrame:
    results = sentiment_pipe(df["text"].tolist())
    df = df.copy()
    df["sentiment_label"] = [r["label"] for r in results]
    df["sentiment_score"] = [r["score"] for r in results]
    return df


# -----------------------------
#   TOPIC MODELING SECTION
# -----------------------------

def init_embedding_model():
    # Lightweight, fast embedding model
    return SentenceTransformer("all-MiniLM-L6-v2")


def add_topics(df: pd.DataFrame, embedder) -> pd.DataFrame:
    """
    Create embeddings for each text, cluster them with KMeans,
    and assign simple topic names based on common words.
    """
    df = df.copy()

    texts = df["text"].tolist()
    print("Creating embeddings for topic clustering...")
    embeddings = embedder.encode(texts, show_progress_bar=True)

    # Choose a small number of clusters for our tiny dataset
    n_clusters = min(4, len(df))  # up to 4 topics
    print(f"Clustering into {n_clusters} topics with KMeans...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    df["topic_cluster"] = cluster_labels

    # Name topics using simple word frequency within each cluster
    topic_names = {}
    stopwords = {
        "the","and","to","it","is","of","i","a","in","for","this","that","was","are","but",
        "not","very","really","just","like","much","im","i’m","i'm","you","we","they",
        "so","be","my","your","their","our","on","with","at","too","also","have","has","had",
        "as","if","or","all","from","by","an","me","he","she","his","her","its","what","when",
        "which","who","about","there","more","no","one","out","up","would","could","should"
    }

    for cluster_id in sorted(df["topic_cluster"].unique()):
        subset = df[df["topic_cluster"] == cluster_id]["text"]
        words = " ".join(subset).lower().split()
        words = [
            w.strip(".,!?\"'()")
            for w in words
            if w not in stopwords and len(w) > 2
        ]

        if not words:
            topic_names[cluster_id] = f"Topic {cluster_id}"
        else:
            # Take up to 3 most common words and join them
            top_words = Counter(words).most_common(3)
            topic_names[cluster_id] = " / ".join([w for w, _ in top_words])

    df["topic_name"] = df["topic_cluster"].map(topic_names)

    return df


# -----------------------------
#   SAVE / MAIN SECTION
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

    print("Adding topics (embeddings + KMeans)...")
    df = add_topics(df, embedder)

    print("Sample with sentiment & topics:")
    print(df[["id", "text", "sentiment_label", "topic_name"]].head())

    save_processed_data(df, processed_path)


if __name__ == "__main__":
    main()
