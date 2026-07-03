"""Recommendation generation shared by both the SVD and PMF models."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RecommenderModel:
    """Bundles everything generate_recommendations() needs for one trained model."""

    name: str
    predictions: np.ndarray  # users x movies predicted ratings (dense)
    user_index: dict  # UserID -> row index
    movie_index: dict  # MovieID -> column index
    movies_df: pd.DataFrame  # MovieID, Title, Genres
    seen_matrix: np.ndarray  # users x movies, >0 where the user has rated the movie (any split)


def generate_recommendations(user_id, model, top_n=10):
    """Top-N unseen movies for user_id, ranked by predicted rating, from `model`."""
    if user_id not in model.user_index:
        raise ValueError(f"Unknown user_id: {user_id}")

    row = model.user_index[user_id]
    preds = model.predictions[row].copy()
    preds[model.seen_matrix[row] > 0] = -np.inf

    top_n = min(top_n, len(preds))
    top_cols = np.argpartition(-preds, top_n - 1)[:top_n]
    top_cols = top_cols[np.argsort(-preds[top_cols])]

    movie_id_by_col = {col: mid for mid, col in model.movie_index.items()}
    records = []
    for col in top_cols:
        movie_id = movie_id_by_col[col]
        movie_row = model.movies_df.loc[model.movies_df["MovieID"] == movie_id]
        title = movie_row["Title"].values[0] if len(movie_row) else f"Movie {movie_id}"
        genres = movie_row["Genres"].values[0] if len(movie_row) else ""
        records.append(
            {
                "MovieID": int(movie_id),
                "Title": title,
                "Genres": genres,
                "PredictedRating": float(preds[col]),
            }
        )
    return pd.DataFrame(records)


def top_rated_movies(user_id, ratings_df, movies_df, top_n=10):
    """A user's own highest-rated movies, from the ratings history."""
    user_ratings = ratings_df[ratings_df["UserID"] == user_id]
    merged = user_ratings.merge(movies_df, on="MovieID")
    merged = merged.sort_values("Rating", ascending=False).head(top_n)
    return merged[["MovieID", "Title", "Genres", "Rating"]].reset_index(drop=True)


def save_user_recommendations(user_id, svd_model, pmf_model, top_n, out_path):
    """Side-by-side SVD vs PMF top-N recommendations for one user, written to CSV."""
    svd_recs = generate_recommendations(user_id, svd_model, top_n).add_prefix("SVD_")
    pmf_recs = generate_recommendations(user_id, pmf_model, top_n).add_prefix("PMF_")
    combined = pd.concat([svd_recs, pmf_recs], axis=1)
    combined.to_csv(out_path, index=False)
    return combined
