"""Streamlit dashboard: enter a MovieLens user ID, see their rating history and
side-by-side SVD vs PMF recommendations. Run with: streamlit run app.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")  # noqa: E402 -- must precede pyplot import; standard Agg-backend pattern
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from models.ensemble_model import apply_blend, ensemble_full_matrix, load_blend, load_ensemble
from models.pmf_model import load_prediction_matrix
from utils.data_loader import load_movies, load_ratings, load_users
from utils.interpretability import explain_recommendation
from utils.matrix_creation import build_user_item_matrix
from utils.recommendation import RecommenderModel, generate_recommendations, top_rated_movies
from utils.side_features import build_item_features, build_user_features

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(ROOT, "reports")

st.set_page_config(page_title="Movie Recommender: SVD vs PMF", layout="wide")


@st.cache_resource
def load_app_data():
    ratings = load_ratings()
    movies = load_movies()
    users = load_users()
    all_users = np.sort(ratings["UserID"].unique())
    all_movies = np.sort(ratings["MovieID"].unique())
    user_index = {uid: i for i, uid in enumerate(all_users)}
    movie_index = {mid: i for i, mid in enumerate(all_movies)}
    user_features = build_user_features(users, user_index)
    item_features = build_item_features(movies, movie_index)

    seen_matrix, _, _ = build_user_item_matrix(ratings, all_users, all_movies)

    svd_path = os.path.join(REPORTS_DIR, "svd_predictions.npy")
    pmf_factors_dir = os.path.join(REPORTS_DIR, "pmf_factors")
    if not os.path.exists(svd_path) or not os.path.isdir(pmf_factors_dir):
        return None

    svd_predictions = np.load(svd_path)
    classical_pmf_predictions = load_prediction_matrix(pmf_factors_dir)
    item_factors = np.load(os.path.join(pmf_factors_dir, "V.npy"))

    ensemble_dir = os.path.join(REPORTS_DIR, "neural_ensemble")
    blend_path = os.path.join(REPORTS_DIR, "blend_weights.json")
    if os.path.isdir(ensemble_dir) and os.path.exists(blend_path):
        # Final reported "PMF" is a blend of classical PMF + SVD + a NeuralMF
        # ensemble (see README) -- reconstruct it exactly as run_pipeline.py did.
        ensemble = load_ensemble(
            len(all_users), len(all_movies), ensemble_dir,
            user_features=user_features, item_features=item_features,
        )
        neural_predictions = ensemble_full_matrix(ensemble, len(all_users), len(all_movies))
        blend = load_blend(blend_path)
        components = {
            "svd": svd_predictions,
            "pmf": classical_pmf_predictions,
            "neural": neural_predictions,
        }
        pmf_predictions = np.clip(apply_blend(blend, components), 1.0, 5.0).astype("float32")
    else:
        pmf_predictions = classical_pmf_predictions

    metrics = {}
    metrics_path = os.path.join(REPORTS_DIR, "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    svd_model = RecommenderModel(
        "SVD", svd_predictions, user_index, movie_index, movies, seen_matrix
    )
    pmf_model = RecommenderModel(
        "PMF", pmf_predictions, user_index, movie_index, movies, seen_matrix
    )

    return dict(
        ratings=ratings,
        movies=movies,
        user_index=user_index,
        movie_index=movie_index,
        item_factors=item_factors,
        svd_model=svd_model,
        pmf_model=pmf_model,
        metrics=metrics,
        min_user=int(all_users.min()),
        max_user=int(all_users.max()),
    )


def render_comparison_chart(user_id, svd_model, pmf_model, movies):
    svd_recs = generate_recommendations(user_id, svd_model, top_n=10)
    pmf_recs = generate_recommendations(user_id, pmf_model, top_n=10)
    union_ids = list(dict.fromkeys(list(svd_recs["MovieID"]) + list(pmf_recs["MovieID"])))

    row = svd_model.user_index[user_id]
    titles, svd_vals, pmf_vals = [], [], []
    for mid in union_ids:
        col = svd_model.movie_index[mid]
        title = movies.loc[movies["MovieID"] == mid, "Title"].values[0]
        titles.append(title[:25])
        svd_vals.append(svd_model.predictions[row, col])
        pmf_vals.append(pmf_model.predictions[row, col])

    x = np.arange(len(union_ids))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(union_ids) * 0.6), 5))
    ax.bar(x - width / 2, svd_vals, width, label="SVD")
    ax.bar(x + width / 2, pmf_vals, width, label="PMF")
    ax.set_xticks(x)
    ax.set_xticklabels(titles, rotation=60, ha="right")
    ax.set_xlabel("Movie")
    ax.set_ylabel("Predicted rating")
    ax.set_title(f"SVD vs PMF predictions for user {user_id}")
    ax.legend()
    fig.tight_layout()
    return fig, svd_recs, pmf_recs


def parse_user_id(raw_value, min_user, max_user, user_index):
    """Returns (user_id, error_message). error_message is None on success."""
    raw_value = raw_value.strip()
    if not raw_value:
        return None, "Please enter a user ID."
    try:
        user_id = int(raw_value)
    except ValueError:
        return None, f"'{raw_value}' is not a valid integer user ID."
    if user_id not in user_index:
        return None, f"User {user_id} was not found. Valid range: {min_user}-{max_user}."
    return user_id, None


def render_why_recommended(user_id, pmf_recs, data):
    st.subheader("Why was this recommended?")
    if pmf_recs.empty:
        st.info("No PMF recommendations available to explain.")
        return

    selected_title = st.selectbox("Explain a PMF recommendation:", pmf_recs["Title"])
    movie_id = int(pmf_recs.loc[pmf_recs["Title"] == selected_title, "MovieID"].iloc[0])

    similar_df, explanation = explain_recommendation(
        user_id,
        movie_id,
        data["item_factors"],
        data["movie_index"],
        data["ratings"],
        data["movies"],
    )
    st.write(explanation)
    if similar_df is not None:
        fig, ax = plt.subplots(figsize=(6, 3))
        labels = [t[:30] for t in similar_df["Title"]][::-1]
        ax.barh(labels, similar_df["Similarity"][::-1], color="#55A868")
        ax.set_xlabel("Latent-factor similarity")
        ax.set_title("Movies you rated highly that resemble this pick")
        fig.tight_layout()
        st.pyplot(fig)


def main():
    st.title("🎬 Movie Recommender: SVD vs PMF")
    st.caption(
        "Matrix-factorization movie recommender trained on MovieLens 1M. "
        "Enter a user ID to see their rating history and compare recommendations "
        "from an SVD model against the project's \"PMF\" model -- classical PMF "
        "blended with a NeuralMF ensemble (see README for why)."
    )

    data = load_app_data()
    if data is None:
        st.error(
            "No trained model artifacts found under reports/. Run `python run_pipeline.py` first."
        )
        return

    if data["metrics"]:
        cols = st.columns(3)
        cols[0].metric("SVD RMSE", data["metrics"].get("SVD_RMSE"))
        cols[1].metric("PMF RMSE", data["metrics"].get("PMF_RMSE"))
        cols[2].metric("PMF improvement", f"{data['metrics'].get('PMF_vs_SVD_improvement_%')}%")

    st.divider()

    raw_user_id = st.text_input(
        f"User ID ({data['min_user']}-{data['max_user']})", value=str(data["min_user"])
    )
    user_id, error = parse_user_id(
        raw_user_id, data["min_user"], data["max_user"], data["user_index"]
    )
    if error:
        st.error(error)
        return

    top_movies = top_rated_movies(user_id, data["ratings"], data["movies"], top_n=10)

    st.subheader(f"User {user_id}'s top-rated movies")
    if top_movies.empty:
        st.info("This user has no ratings on record.")
    else:
        st.dataframe(top_movies, use_container_width=True, hide_index=True)

    st.subheader("Recommendations")
    fig, svd_recs, pmf_recs = render_comparison_chart(
        user_id, data["svd_model"], data["pmf_model"], data["movies"]
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**SVD top 10**")
        st.dataframe(svd_recs, use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**PMF top 10**")
        st.dataframe(pmf_recs, use_container_width=True, hide_index=True)

    st.subheader("SVD vs PMF prediction comparison")
    st.pyplot(fig)

    render_why_recommended(user_id, pmf_recs, data)


if __name__ == "__main__":
    main()
