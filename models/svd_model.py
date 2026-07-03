"""SVD-based matrix factorization recommender using scipy.sparse.linalg.svds."""

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds


def train_svd(normalized_matrix, k=50):
    """Truncated SVD of the normalized user-item matrix. Returns U, sigma, Vt
    sorted by descending singular value (svds returns them ascending)."""
    U, sigma, Vt = svds(csc_matrix(normalized_matrix), k=k)
    order = np.argsort(-sigma)
    return U[:, order], sigma[order], Vt[order, :]


def reconstruct_predictions(U, sigma, Vt, baseline, clip=(1.0, 5.0)):
    """Full predicted rating matrix: baseline (global + user + item bias) plus
    the low-rank reconstruction of the residual signal."""
    predicted = (U * sigma) @ Vt + baseline
    if clip is not None:
        predicted = np.clip(predicted, clip[0], clip[1])
    return predicted.astype("float32")


def rmse_on_ratings(predicted, ratings_df, user_index, movie_index):
    """RMSE of `predicted` against a ratings table (e.g. the held-out test split)."""
    known = ratings_df["UserID"].isin(user_index) & ratings_df["MovieID"].isin(movie_index)
    subset = ratings_df[known]
    rows = subset["UserID"].map(user_index).to_numpy()
    cols = subset["MovieID"].map(movie_index).to_numpy()
    preds = predicted[rows, cols]
    actual = subset["Rating"].to_numpy()
    rmse = float(np.sqrt(np.mean((preds - actual) ** 2)))
    return rmse, len(subset)
