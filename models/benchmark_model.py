"""Item-based collaborative filtering benchmark (cosine similarity), used as
the classic memory-based CF baseline that the matrix-factorization models
(SVD, PMF) are expected to beat."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def train_item_based_cf(train_matrix):
    """Item-item cosine similarity over raw rating vectors (0 = missing)."""
    similarity = cosine_similarity(train_matrix.T)
    np.fill_diagonal(similarity, 0.0)  # an item is not its own neighbor
    return similarity.astype("float32")


def predict_item_based_cf(train_matrix, similarity, clip=(1.0, 5.0)):
    """Predicted rating = similarity-weighted average of a user's own ratings
    on the movies most like the target movie. Falls back to the global mean
    when a user/movie pair has no similar, already-rated movies to lean on."""
    mask = (train_matrix > 0).astype("float32")
    numerator = train_matrix @ similarity
    denominator = mask @ np.abs(similarity)
    global_mean = float(train_matrix[train_matrix > 0].mean())

    predicted = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, global_mean),
        where=denominator > 1e-8,
    )
    if clip is not None:
        predicted = np.clip(predicted, clip[0], clip[1])
    return predicted.astype("float32")
