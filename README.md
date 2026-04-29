# Unsupervised Course Recommendation and Analysis

This project builds an unsupervised-learning pipeline on the Coursera dataset to:

- analyze data with clear visualizations,
- discover hidden course groups using clustering,
- extract latent topics from course text,
- generate cluster-based course recommendation tables.

## Dataset Used

- `CourseraDataset-Clean.csv` (primary input)
- `CourseraDataset-Unclean.csv` (optional, for your own cleaning experiments)

## Models and Methods

- **KMeans** (primary clustering model, best `k` selected using silhouette score)
- **Agglomerative Clustering** (hierarchical clustering baseline)
- **DBSCAN** (density-based clustering baseline)
- **LDA Topic Modeling** (unsupervised topic extraction from text)
- **PCA** (2D projection for cluster visualization)

## Visualizations Generated

The script saves all plots in `outputs/plots/`:

- Rating distribution
- Duration distribution
- Top keyword/category counts
- Numeric correlation heatmap
- PCA cluster plots for KMeans, Agglomerative, and DBSCAN

## Output Tables Generated

The script saves all CSV outputs in `outputs/tables/`:

- `kmeans_silhouette_scores.csv`
- `cluster_summary.csv`
- `top_courses_by_cluster.csv`
- `lda_topics.csv`
- `agglomerative_metrics.csv`
- `dbscan_metrics.csv`

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python unsupervised_course_analysis.py --input CourseraDataset-Clean.csv
```

## Streamlit App (Nice UI + Top 10 recommendations + Images)

Run:

```bash
streamlit run streamlit_app.py
```

Features:

- Clean interface with sidebar filters (category, level, min rating)
- Course picker and top 10 similar recommendations
- Course front-page image fetched from course metadata (`og:image`) with fallback image
- Direct link to each recommended course page

## Notes

- If your dataset has slightly different column names, update the column names inside `unsupervised_course_analysis.py`.
- You can tune clustering quality by changing KMeans range, DBSCAN `eps`, and feature settings.
