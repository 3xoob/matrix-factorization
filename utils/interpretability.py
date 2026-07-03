"""Global and local interpretability for the latent-factor models.

Global: which genres each latent factor tends to activate for (a per-factor
"theme"), summarized as a factor x genre heatmap.

Local: why one specific movie was recommended to one specific user, found by
comparing the recommended movie's latent vector against the latent vectors of
movies the user has already rated highly (nearest neighbor in factor space).
"""

import os

import numpy as np
import pandas as pd


def genre_matrix(movies_df):
    """Multi-hot (movies x genres) encoding, aligned to movies_df's row order."""
    return movies_df["Genres"].str.get_dummies(sep="|")


def _rated_movies(movies_df, movie_index):
    """movies_df restricted to movies that actually appear in movie_index --
    the catalog (movies.dat) includes titles with zero ratings, which have no
    row in the trained factor matrices."""
    return movies_df[movies_df["MovieID"].isin(movie_index)].reset_index(drop=True)


def factor_genre_affinity(item_factors, movies_df, movie_index):
    """Average latent-factor value per genre: rows = factors, columns = genres.

    A large positive/negative value means that factor consistently activates
    for movies of that genre -- the closest thing to a human-readable "meaning"
    a matrix-factorization latent dimension has.
    """
    rated = _rated_movies(movies_df, movie_index)
    genres = genre_matrix(rated)
    row_for_movie = rated["MovieID"].map(movie_index).to_numpy()
    factors_aligned = item_factors[row_for_movie]  # rated's row order

    genre_counts = genres.sum(axis=0).to_numpy()
    affinity = (genres.to_numpy().T @ factors_aligned) / np.maximum(genre_counts[:, None], 1)
    factor_names = [f"factor_{k}" for k in range(item_factors.shape[1])]
    return pd.DataFrame(affinity.T, columns=genres.columns, index=factor_names)


def top_movies_for_factor(item_factors, movies_df, movie_index, factor_idx, top_k=10):
    """The movies most strongly (positively) associated with one latent factor,
    used to sanity-check what a factor-genre affinity value actually means."""
    rated = _rated_movies(movies_df, movie_index)
    row_for_movie = rated["MovieID"].map(movie_index).to_numpy()
    loadings = item_factors[row_for_movie, factor_idx]
    ranked = rated.assign(loading=loadings).sort_values("loading", ascending=False)
    return ranked.head(top_k)[["MovieID", "Title", "Genres", "loading"]].reset_index(drop=True)


def plot_factor_genre_heatmap(item_factors, movies_df, movie_index, save_path, n_factors=12):
    import matplotlib

    matplotlib.use("Agg")  # noqa: E402
    import matplotlib.pyplot as plt

    affinity = factor_genre_affinity(item_factors[:, :n_factors], movies_df, movie_index)

    fig, ax = plt.subplots(figsize=(max(10, len(affinity.columns) * 0.5), max(6, n_factors * 0.4)))
    im = ax.imshow(affinity.values, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(affinity.columns)))
    ax.set_xticklabels(affinity.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(affinity.index)))
    ax.set_yticklabels(affinity.index)
    ax.set_xlabel("Genre")
    ax.set_ylabel("Latent factor")
    ax.set_title("Global interpretability: average latent-factor value by genre")
    fig.colorbar(im, ax=ax, label="Mean factor value")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return affinity


def explain_recommendation(
    user_id, movie_id, item_factors, movie_index, ratings_df, movies_df, top_k=3
):
    """Why was `movie_id` recommended to `user_id`? Finds the user's own highly
    rated movies whose latent vectors are most similar (cosine) to the
    recommended movie's latent vector -- i.e. "this was recommended because it
    looks, in latent taste-space, like movies you already rated highly."
    """
    target_vec = item_factors[movie_index[movie_id]]
    target_norm = np.linalg.norm(target_vec) + 1e-8

    user_ratings = ratings_df[(ratings_df["UserID"] == user_id) & (ratings_df["Rating"] >= 4)]
    if user_ratings.empty:
        return None, "This user has no highly-rated movies to compare against."

    rows = []
    for _, row in user_ratings.iterrows():
        mid = row["MovieID"]
        if mid not in movie_index or mid == movie_id:
            continue
        vec = item_factors[movie_index[mid]]
        sim = float(np.dot(target_vec, vec) / ((np.linalg.norm(vec) + 1e-8) * target_norm))
        rows.append({"MovieID": mid, "Rating": row["Rating"], "Similarity": sim})

    if not rows:
        return None, "No comparable rated movies found for this user."

    similar = pd.DataFrame(rows).sort_values("Similarity", ascending=False).head(top_k)
    similar = similar.merge(movies_df[["MovieID", "Title"]], on="MovieID")

    rec_title = movies_df.loc[movies_df["MovieID"] == movie_id, "Title"].values[0]
    lines = [
        f'Because you rated "{r.Title}" {r.Rating:.0f}/5 (similarity {r.Similarity:.2f})'
        for r in similar.itertuples()
    ]
    explanation = f'"{rec_title}" was recommended mainly ' + "; and ".join(lines)
    return similar.reset_index(drop=True), explanation


def plot_recommendation_explanation(user_id, movie_id, similar_df, movies_df, save_path):
    import matplotlib

    matplotlib.use("Agg")  # noqa: E402
    import matplotlib.pyplot as plt

    rec_title = movies_df.loc[movies_df["MovieID"] == movie_id, "Title"].values[0]
    labels = [t[:30] for t in similar_df["Title"]]
    plt.figure(figsize=(8, 5))
    plt.barh(labels[::-1], similar_df["Similarity"][::-1], color="#55A868")
    plt.xlabel("Latent-factor cosine similarity to recommended movie")
    plt.title(f'Why "{rec_title[:40]}" was recommended to user {user_id}')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close()
