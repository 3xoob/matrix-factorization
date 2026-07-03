"""Train/test splitting and user-item matrix construction/normalization."""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "processed"
)


def split_ratings(ratings_df, test_size=0.2, random_state=42):
    """Row-wise split of the ratings table. random_state is fixed for reproducibility."""
    train_df, test_df = train_test_split(
        ratings_df, test_size=test_size, random_state=random_state
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_user_item_matrix(train_df, all_user_ids, all_movie_ids):
    """Dense user x movie ratings matrix (0 = missing), built from TRAIN ratings only.

    all_user_ids/all_movie_ids should come from the full dataset so that users/movies
    which only appear in the test split still get a valid row/column (cold-start).
    """
    user_index = {uid: i for i, uid in enumerate(all_user_ids)}
    movie_index = {mid: i for i, mid in enumerate(all_movie_ids)}

    matrix = np.zeros((len(all_user_ids), len(all_movie_ids)), dtype="float32")
    rows = train_df["UserID"].map(user_index).to_numpy()
    cols = train_df["MovieID"].map(movie_index).to_numpy()
    matrix[rows, cols] = train_df["Rating"].to_numpy(dtype="float32")

    return matrix, user_index, movie_index


def normalize_matrix(matrix, save_path=None, damping=25.0):
    """Normalize the matrix by removing damped user/item baseline effects
    (Koren-style shrunk means), leaving unobserved cells at 0.

    Plain zero-filling biases a low-rank reconstruction toward predicting
    near-zero ratings everywhere, since the decomposition minimizes error
    over the whole dense matrix, most of which is unobserved. Subtracting a
    baseline (global mean + per-user effect + per-item effect, each shrunk
    toward 0 by `damping` pseudo-observations so sparse users/movies don't
    get noisy estimates) leaves only the genuine interaction signal for the
    factorization to model, which is what's saved to
    processed/user_item_matrix.csv and what SVD is trained on.
    """
    mask = matrix > 0
    global_mean = float(matrix[mask].mean())

    item_counts = mask.sum(axis=0)
    item_dev_sum = np.where(mask, matrix - global_mean, 0).sum(axis=0)
    item_bias = item_dev_sum / (item_counts + damping)

    resid_after_item = np.where(mask, matrix - global_mean - item_bias[np.newaxis, :], 0)
    user_counts = mask.sum(axis=1)
    user_bias = resid_after_item.sum(axis=1) / (user_counts + damping)

    baseline = global_mean + user_bias[:, np.newaxis] + item_bias[np.newaxis, :]
    normalized = np.where(mask, matrix - baseline, 0).astype("float32")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pd.DataFrame(normalized).to_csv(save_path, index=False, float_format="%.3f")

    return normalized, baseline.astype("float32"), global_mean


if __name__ == "__main__":
    from utils.data_loader import load_ratings

    ratings = load_ratings()
    train_df, test_df = split_ratings(ratings)
    all_users = np.sort(ratings["UserID"].unique())
    all_movies = np.sort(ratings["MovieID"].unique())
    matrix, uidx, midx = build_user_item_matrix(train_df, all_users, all_movies)
    print("raw matrix:", matrix.shape, "nnz:", int((matrix > 0).sum()))
    norm, baseline, gmean = normalize_matrix(
        matrix, save_path=os.path.join(PROCESSED_DIR, "user_item_matrix.csv")
    )
    print("global mean:", gmean, "normalized shape:", norm.shape, "baseline shape:", baseline.shape)
