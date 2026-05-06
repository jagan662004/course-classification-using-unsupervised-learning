import numpy as np
import pandas as pd
import requests
import streamlit as st
import re
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD


st.set_page_config(page_title="Course Recommender", page_icon="🎓", layout="wide")


DATA_PATH = "CourseraDataset-Clean.csv"
FALLBACK_IMAGE = "https://placehold.co/640x360/png?text=Course+Image+Not+Available"


def get_course_image(url: str) -> str:
    if not isinstance(url, str) or not url.startswith("http"):
        return FALLBACK_IMAGE
    try:
        response = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            return FALLBACK_IMAGE
        html = response.text
        if BeautifulSoup is not None:
            soup = BeautifulSoup(html, "html.parser")
            og_image = soup.find("meta", property="og:image")
            if og_image and og_image.get("content"):
                return og_image["content"]

            twitter_image = soup.find("meta", attrs={"name": "twitter:image"})
            if twitter_image and twitter_image.get("content"):
                return twitter_image["content"]
        else:
            og_match = re.search(
                r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
                html,
                flags=re.IGNORECASE,
            )
            if og_match:
                return og_match.group(1)
    except Exception:
        return FALLBACK_IMAGE

    return FALLBACK_IMAGE


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    text_cols = [
        "Course Title",
        "What you will learn",
        "Skill gain",
        "Keyword",
        "Level",
        "Offered By",
    ]
    for col in text_cols:
        df[col] = df[col].fillna("Not specified").astype(str)

    numeric_cols = ["Rating", "Duration to complete (Approx.)", "Number of Review"]
    for col in numeric_cols:
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


@st.cache_resource(show_spinner=True)
def build_model_features(df: pd.DataFrame):
    tfidf = TfidfVectorizer(
        max_features=1800,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
    )
    text_features = tfidf.fit_transform(df["combined_text"])

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    numeric_features = numeric_pipeline.fit_transform(
        df[["Rating", "Duration to complete (Approx.)", "Number of Review"]]
    )

    combined_sparse = hstack([text_features, csr_matrix(numeric_features)]).tocsr()
    svd = TruncatedSVD(n_components=120, random_state=42)
    X = svd.fit_transform(combined_sparse)

    model = MiniBatchKMeans(n_clusters=8, random_state=42, batch_size=512, n_init=10)
    labels = model.fit_predict(X)
    return X, labels


def get_top_recommendations(df: pd.DataFrame, X: np.ndarray, labels: np.ndarray, selected_course: str) -> pd.DataFrame:
    selected_idx = df.index[df["Course Title"] == selected_course][0]
    selected_cluster = labels[selected_idx]

    cluster_indices = np.where(labels == selected_cluster)[0]
    cluster_vectors = X[cluster_indices]
    target_vector = X[selected_idx].reshape(1, -1)

    sims = cosine_similarity(target_vector, cluster_vectors).flatten()
    cluster_df = df.iloc[cluster_indices].copy()
    cluster_df["similarity"] = sims

    recs = (
        cluster_df[cluster_df["Course Title"] != selected_course]
        .sort_values(by=["similarity", "Rating", "Number of Review"], ascending=[False, False, False])
        .head(10)
    )
    return recs


def render_analytics_dashboard(df: pd.DataFrame, labels: np.ndarray) -> None:
    analytics_df = df.copy()
    analytics_df["Cluster"] = labels
    sns.set_theme(style="whitegrid")

    st.markdown("## Data Visualization and Analysis")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Courses", f"{len(analytics_df):,}")
    m2.metric("Avg Rating", f"{analytics_df['Rating'].mean():.2f}")
    m3.metric("Median Duration (hrs)", f"{analytics_df['Duration to complete (Approx.)'].median():.1f}")
    m4.metric("Total Reviews", f"{int(analytics_df['Number of Review'].fillna(0).sum()):,}")

    t1, t2, t3, t4 = st.tabs(["Rating & Duration", "Category Insights", "Providers", "Cluster Analysis"])

    with t1:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(analytics_df["Rating"].dropna(), bins=25, kde=True, color="royalblue", ax=ax)
            ax.set_title("Rating Distribution")
            ax.set_xlabel("Rating")
            st.pyplot(fig, clear_figure=True)
        with c2:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(
                analytics_df["Duration to complete (Approx.)"].dropna(),
                bins=25,
                kde=True,
                color="darkorange",
                ax=ax,
            )
            ax.set_title("Duration Distribution")
            ax.set_xlabel("Duration (hours)")
            st.pyplot(fig, clear_figure=True)

    with t2:
        top_keywords = analytics_df["Keyword"].value_counts().head(12).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(top_keywords.index, top_keywords.values, color="#3b82f6")
        ax.set_title("Top 12 Course Categories")
        ax.set_xlabel("Number of Courses")
        st.pyplot(fig, clear_figure=True)

    with t3:
        top_providers = analytics_df["Offered By"].value_counts().head(10).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(top_providers.index, top_providers.values, color="#14b8a6")
        ax.set_title("Top 10 Providers by Course Count")
        ax.set_xlabel("Number of Courses")
        st.pyplot(fig, clear_figure=True)

        provider_rating = (
            analytics_df.groupby("Offered By", dropna=False)["Rating"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .sort_values(ascending=True)
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(provider_rating.index, provider_rating.values, color="#a855f7")
        ax.set_title("Top 10 Providers by Average Rating")
        ax.set_xlabel("Average Rating")
        st.pyplot(fig, clear_figure=True)

    with t4:
        cluster_counts = analytics_df["Cluster"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(cluster_counts.index.astype(str), cluster_counts.values, color="#ef4444")
        ax.set_title("Cluster Size Distribution")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Courses")
        st.pyplot(fig, clear_figure=True)

        cluster_summary = (
            analytics_df.groupby("Cluster", dropna=False)
            .agg(
                courses=("Course Title", "count"),
                avg_rating=("Rating", "mean"),
                avg_duration=("Duration to complete (Approx.)", "mean"),
                avg_reviews=("Number of Review", "mean"),
            )
            .round(2)
            .sort_values(by="courses", ascending=False)
        )
        st.dataframe(cluster_summary, use_container_width=True)

    st.markdown("### Quick Insights")
    top_keyword = analytics_df["Keyword"].value_counts().idxmax()
    top_provider = analytics_df["Offered By"].value_counts().idxmax()
    best_cluster = (
        analytics_df.groupby("Cluster", dropna=False)["Rating"].mean().sort_values(ascending=False).index[0]
    )
    st.write(
        f"- Most common category is **{top_keyword}**.\n"
        f"- Provider with the largest catalog is **{top_provider}**.\n"
        f"- Highest average-rated cluster is **Cluster {best_cluster}**."
    )


def recommendation_card(row: pd.Series):
    image_url = get_course_image(row.get("Course Url", ""))
    with st.container(border=True):
        c1, c2 = st.columns([1, 2], gap="large")
        with c1:
            try:
                st.image(image_url, use_container_width=True)
            except TypeError:
                st.image(image_url, use_column_width=True)
        with c2:
            st.subheader(row["Course Title"])
            st.write(f"**Provider:** {row.get('Offered By', 'N/A')}")
            st.write(f"**Category:** {row.get('Keyword', 'N/A')} | **Level:** {row.get('Level', 'N/A')}")
            st.write(
                f"**Rating:** {row.get('Rating', np.nan):.2f} | "
                f"**Reviews:** {int(row.get('Number of Review', 0)) if pd.notna(row.get('Number of Review', np.nan)) else 0}"
            )
            st.write(f"**Duration (hrs):** {row.get('Duration to complete (Approx.)', np.nan)}")
            course_url = row.get("Course Url", "")
            if isinstance(course_url, str) and course_url.startswith("http"):
                st.markdown(f"[Open Course Page]({course_url})")


def main():
    st.title(" Course Recommender ")
    st.caption("Get top 10 similar courses using clustering + vector similarity.")

    df = load_data(DATA_PATH)
    X, labels = build_model_features(df)

    with st.sidebar:
        st.header("Filters")
        keyword_filter = st.multiselect(
            "Category (Keyword)",
            sorted(df["Keyword"].dropna().unique().tolist()),
        )
        level_filter = st.multiselect(
            "Level",
            sorted(df["Level"].dropna().unique().tolist()),
        )
        min_rating = st.slider("Minimum Rating", 0.0, 5.0, 4.0, 0.1)

    filtered_df = df.copy()
    if keyword_filter:
        filtered_df = filtered_df[filtered_df["Keyword"].isin(keyword_filter)]
    if level_filter:
        filtered_df = filtered_df[filtered_df["Level"].isin(level_filter)]
    filtered_df = filtered_df[filtered_df["Rating"].fillna(0) >= min_rating]

    render_analytics_dashboard(df, labels)

    st.markdown("### Select a course you like")
    if filtered_df.empty:
        st.warning("No courses match the current filters. Try relaxing filters.")
        return

    selected_course = st.selectbox("Course", sorted(filtered_df["Course Title"].unique().tolist()))
    try:
        recommend_clicked = st.button("Get Top 10 Recommendations", type="primary", use_container_width=True)
    except TypeError:
        recommend_clicked = st.button("Get Top 10 Recommendations", type="primary")

    if recommend_clicked:
        recs = get_top_recommendations(df, X, labels, selected_course)
        st.markdown("## Top 10 Recommendations")
        if recs.empty:
            st.info("Could not find recommendations for this course. Try another one.")
            return
        for _, row in recs.iterrows():
            recommendation_card(row)


if __name__ == "__main__":
    main()
