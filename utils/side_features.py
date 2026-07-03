"""Precomputed side-feature matrices (genre for movies, demographics for
users), used to augment NeuralMF with content signal a pure ID-based
collaborative model never sees -- e.g. two comedies a user has never
encountered still share a "Comedy" signal a plain embedding lookup can't."""

import numpy as np


def build_item_features(movies_df, movie_index):
    """(n_items x n_genres) genre multi-hot, aligned to movie_index order."""
    n_items = len(movie_index)
    genre_dummies = movies_df["Genres"].str.get_dummies(sep="|")
    features = np.zeros((n_items, genre_dummies.shape[1]), dtype="float32")
    movie_ids = movies_df["MovieID"].to_numpy()
    genre_values = genre_dummies.to_numpy(dtype="float32")
    for row, movie_id in enumerate(movie_ids):
        if movie_id in movie_index:
            features[movie_index[movie_id]] = genre_values[row]
    return features


def build_user_features(users_df, user_index):
    """(n_users x (2 + n_age_buckets + n_occupations)) one-hot demographics:
    gender, age bucket, occupation."""
    n_users = len(user_index)
    age_buckets = sorted(users_df["Age"].unique())
    age_to_idx = {a: i for i, a in enumerate(age_buckets)}
    occupations = sorted(users_df["Occupation"].unique())
    occ_to_idx = {o: i for i, o in enumerate(occupations)}

    n_dims = 2 + len(age_buckets) + len(occupations)
    features = np.zeros((n_users, n_dims), dtype="float32")
    for row in users_df.itertuples():
        if row.UserID not in user_index:
            continue
        idx = user_index[row.UserID]
        features[idx, 0 if row.Gender == "M" else 1] = 1.0
        features[idx, 2 + age_to_idx[row.Age]] = 1.0
        features[idx, 2 + len(age_buckets) + occ_to_idx[row.Occupation]] = 1.0
    return features
