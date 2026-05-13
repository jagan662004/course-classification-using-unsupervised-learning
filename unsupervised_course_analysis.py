import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix, hstack
from sklearn.cluster import AgglomerativeClustering, DBSCAN, MiniBatchKMeans
from sklearn.decomposition import LatentDirichletAllocation, PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.copy()

    text_cols = [
        "Course Title",
        "What you will learn",
        "Skill gain",
        "Keyword",
        "Level",
        "Offered By",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Not specified").astype(str)

    numeric_cols = ["Rating", "Duration to complete (Approx.)", "Number of Review"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["combined_text"] = (
        df["Course Title"]
        + " "
        + df["What you will learn"]
        + " "
        + df["Skill gain"]
        + " "
        + df["Keyword"]
        + " "
        + df["Level"]
        + " "
        + df["Offered By"]
    )
    return df


def create_output_folders(base_dir: Path) -> Tuple[Path, Path]:
    plots_dir = base_dir / "outputs" / "plots"
    tables_dir = base_dir / "outputs" / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir, tables_dir


def run_eda_visualizations(df: pd.DataFrame, plots_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 5))
    sns.histplot(df["Rating"].dropna(), bins=30, kde=True, color="royalblue")
    plt.title("Distribution of Course Ratings")
    plt.xlabel("Rating")
    plt.tight_layout()
    plt.savefig(plots_dir / "rating_distribution.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    duration = df["Duration to complete (Approx.)"].dropna()
    sns.histplot(duration, bins=30, kde=True, color="darkorange")
    plt.title("Distribution of Course Duration")
    plt.xlabel("Duration (hours approx.)")
    plt.tight_layout()
    plt.savefig(plots_dir / "duration_distribution.png", dpi=220)
    plt.close()

    plt.figure(figsize=(11, 6))
    keyword_counts = df["Keyword"].value_counts().head(15)
    sns.barplot(
        x=keyword_counts.values,
        y=keyword_counts.index,
        hue=keyword_counts.index,
        palette="viridis",
        legend=False,
    )
    plt.title("Top 15 Course Categories (Keyword)")
    plt.xlabel("Count")
    plt.ylabel("Keyword")
    plt.tight_layout()
    plt.savefig(plots_dir / "top_keywords.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 6))
    corr_df = df[["Rating", "Duration to complete (Approx.)", "Number of Review"]]
    sns.heatmap(corr_df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig(plots_dir / "correlation_heatmap.png", dpi=220)
    plt.close()


def build_features(df: pd.DataFrame) -> Tuple[np.ndarray, TfidfVectorizer]:
    tfidf = TfidfVectorizer(
        max_features=1800,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
    )
    text_features_sparse = tfidf.fit_transform(df["combined_text"])

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    numeric_features = numeric_pipeline.fit_transform(
        df[["Rating", "Duration to complete (Approx.)", "Number of Review"]]
    )

    numeric_sparse = csr_matrix(numeric_features)
    combined_sparse = hstack([text_features_sparse, numeric_sparse]).tocsr()

    svd = TruncatedSVD(n_components=120, random_state=42)
    combined_dense = svd.fit_transform(combined_sparse)
    return combined_dense, tfidf


def find_best_kmeans(X: np.ndarray, k_values: List[int]) -> Tuple[MiniBatchKMeans, Dict[int, float]]:
    scores = {}
    best_model = None
    best_score = -1.0

    for k in k_values:
        model = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=512, n_init=10)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels, sample_size=min(2000, len(X)), random_state=42)
        scores[k] = score
        if score > best_score:
            best_score = score
            best_model = model

    if best_model is None:
        raise RuntimeError("KMeans model training failed.")
    return best_model, scores


def run_other_clusterers(X: np.ndarray, n_clusters: int) -> Dict[str, np.ndarray]:
    sample_size = min(2500, len(X))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X), size=sample_size, replace=False)
    X_sample = X[sample_idx]

    agg = AgglomerativeClustering(n_clusters=n_clusters)
    agg_sample_labels = agg.fit_predict(X_sample)
    agg_centroids = np.vstack([X_sample[agg_sample_labels == c].mean(axis=0) for c in range(n_clusters)])
    agg_dists = ((X[:, None, :] - agg_centroids[None, :, :]) ** 2).sum(axis=2)
    agg_labels = agg_dists.argmin(axis=1)

    dbscan = DBSCAN(eps=2.8, min_samples=10)
    dbscan_sample_labels = dbscan.fit_predict(X_sample)
    non_noise = dbscan_sample_labels != -1
    if non_noise.sum() > 0:
        unique_clusters = np.unique(dbscan_sample_labels[non_noise])
        db_centroids = np.vstack([X_sample[dbscan_sample_labels == c].mean(axis=0) for c in unique_clusters])
        db_dists = ((X[:, None, :] - db_centroids[None, :, :]) ** 2).sum(axis=2)
        nearest_idx = db_dists.argmin(axis=1)
        dbscan_labels = unique_clusters[nearest_idx]
    else:
        dbscan_labels = np.full(len(X), -1)

    return {
        "Agglomerative": agg_labels,
        "DBSCAN": dbscan_labels,
    }


def plot_cluster_projection(X: np.ndarray, labels: np.ndarray, title: str, out_path: Path) -> None:
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(X)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels,
        cmap="tab20",
        alpha=0.75,
        s=22,
    )
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def run_topic_modeling(df: pd.DataFrame, tables_dir: Path, n_topics: int = 8) -> None:
    count_vectorizer = CountVectorizer(
        stop_words="english",
        max_features=2200,
        min_df=2,
        max_df=0.9,
    )
    doc_term = count_vectorizer.fit_transform(df["combined_text"])
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method="batch")
    lda.fit(doc_term)

    feature_names = np.array(count_vectorizer.get_feature_names_out())
    topic_rows = []
    for i, topic in enumerate(lda.components_):
        top_idx = topic.argsort()[-12:][::-1]
        words = ", ".join(feature_names[top_idx])
        topic_rows.append({"Topic": f"Topic_{i}", "Top Words": words})

    pd.DataFrame(topic_rows).to_csv(tables_dir / "lda_topics.csv", index=False)


def build_recommendations(df: pd.DataFrame, labels: np.ndarray, tables_dir: Path) -> None:
    rec_df = df.copy()
    rec_df["Cluster"] = labels

    summary = (
        rec_df.groupby("Cluster", dropna=False)
        .agg(
            course_count=("Course Title", "count"),
            avg_rating=("Rating", "mean"),
            avg_duration=("Duration to complete (Approx.)", "mean"),
            avg_reviews=("Number of Review", "mean"),
        )
        .reset_index()
        .sort_values(by="course_count", ascending=False)
    )
    summary.to_csv(tables_dir / "cluster_summary.csv", index=False)

    top_courses = (
        rec_df.sort_values(by=["Cluster", "Rating", "Number of Review"], ascending=[True, False, False])
        .groupby("Cluster")
        .head(10)[
            ["Cluster", "Course Title", "Keyword", "Level", "Rating", "Number of Review", "Course Url"]
        ]
    )
    top_courses.to_csv(tables_dir / "top_courses_by_cluster.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unsupervised learning + visualization pipeline for Coursera-style course dataset."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="CourseraDataset-Clean.csv",
        help="Path to clean dataset csv file.",
    )
    args = parser.parse_args()

    base_dir = Path(".")
    plots_dir, tables_dir = create_output_folders(base_dir)
    df = load_and_prepare_data(Path(args.input))

    run_eda_visualizations(df, plots_dir)
    X, _ = build_features(df)

    kmeans_model, k_scores = find_best_kmeans(X, list(range(4, 13)))
    kmeans_labels = kmeans_model.labels_

    pd.DataFrame({"k": list(k_scores.keys()), "silhouette_score": list(k_scores.values())}).to_csv(
        tables_dir / "kmeans_silhouette_scores.csv", index=False
    )

    plot_cluster_projection(
        X, kmeans_labels, "KMeans Clusters (PCA Projection)", plots_dir / "kmeans_clusters_pca.png"
    )

    other_models = run_other_clusterers(X, n_clusters=kmeans_model.n_clusters)
    for model_name, labels in other_models.items():
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1 and (model_name != "DBSCAN" or len(unique_labels[unique_labels != -1]) > 1):
            score = silhouette_score(X, labels) if len(set(labels)) > 1 else np.nan
        else:
            score = np.nan
        pd.DataFrame(
            [{"model": model_name, "silhouette_score": score, "n_unique_labels": len(unique_labels)}]
        ).to_csv(tables_dir / f"{model_name.lower()}_metrics.csv", index=False)
        plot_cluster_projection(
            X,
            labels,
            f"{model_name} Clusters (PCA Projection)",
            plots_dir / f"{model_name.lower()}_clusters_pca.png",
        )

    run_topic_modeling(df, tables_dir, n_topics=8)
    build_recommendations(df, kmeans_labels, tables_dir)

    print("Pipeline completed.")
    print(f"Plots saved to: {plots_dir}")
    print(f"Tables saved to: {tables_dir}")


if __name__ == "__main__":
    main()

