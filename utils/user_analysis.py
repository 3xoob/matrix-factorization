"""Per-user accuracy analysis: find a well-predicted training user, a poorly
predicted training user, and simulate a genuine cold-start (test-only) user.

MovieLens 1M requires every user to have >=20 ratings, so with a random
row-wise 80/20 split the odds of any real user landing entirely in the test
set are astronomically small (0.2^20). A true "test-only" user doesn't occur
naturally in this dataset -- so to demonstrate that scenario honestly, we
simulate it: pick a user and exclude every one of their ratings from training,
then see how the (otherwise identical) model performs for someone it has never
seen a single rating from.
"""

import numpy as np


def per_user_rmse(predictions, test_df, user_index, movie_index, min_test_ratings=5):
    """RMSE broken out per user, restricted to users with enough test ratings
    for the number to be meaningful, sorted best (lowest RMSE) first."""
    known = test_df["UserID"].isin(user_index) & test_df["MovieID"].isin(movie_index)
    sub = test_df[known].copy()
    sub["row"] = sub["UserID"].map(user_index)
    sub["col"] = sub["MovieID"].map(movie_index)
    sub["pred"] = predictions[sub["row"].to_numpy(), sub["col"].to_numpy()]
    sub["sq_err"] = (sub["pred"] - sub["Rating"]) ** 2

    grouped = sub.groupby("UserID").agg(
        n_test_ratings=("Rating", "size"),
        rmse=("sq_err", lambda x: np.sqrt(x.mean())),
        rating_std=("Rating", "std"),
    )
    grouped = grouped[grouped["n_test_ratings"] >= min_test_ratings]
    return grouped.reset_index().sort_values("rmse")


def pick_accurate_and_inaccurate_users(per_user_df):
    """Best- and worst-predicted training users, by per-user test RMSE."""
    accurate_user = int(per_user_df.iloc[0]["UserID"])
    inaccurate_user = int(per_user_df.iloc[-1]["UserID"])
    return accurate_user, inaccurate_user


def pick_coldstart_user(ratings_df, min_ratings=20, max_ratings=30, random_state=42):
    """A real user with a modest rating count, chosen to simulate cold start on."""
    counts = ratings_df.groupby("UserID").size()
    candidates = counts[(counts >= min_ratings) & (counts <= max_ratings)].index
    return int(np.random.RandomState(random_state).choice(candidates))


def exclude_user_ratings(ratings_df, user_id):
    """Ratings with `user_id` entirely removed, for simulating a user the
    model has never trained on."""
    return ratings_df[ratings_df["UserID"] != user_id].reset_index(drop=True)


def summarize_user_profile(user_id, ratings_df, movies_df):
    """Plain-English-ready stats about a user's rating history, used to ground
    the accurate-vs-inaccurate writeup in real numbers rather than guesses."""
    user_ratings = ratings_df[ratings_df["UserID"] == user_id].merge(movies_df, on="MovieID")
    if user_ratings.empty:
        return {"user_id": user_id, "n_ratings": 0}

    genre_dummies = user_ratings["Genres"].str.get_dummies(sep="|")
    genre_counts = genre_dummies.sum().sort_values(ascending=False)
    top3_share = genre_counts.head(3).sum() / genre_counts.sum() if genre_counts.sum() else 0.0
    return {
        "user_id": user_id,
        "n_ratings": len(user_ratings),
        "rating_mean": float(user_ratings["Rating"].mean()),
        "rating_std": float(user_ratings["Rating"].std()) if len(user_ratings) > 1 else 0.0,
        "top_genres": genre_counts.head(3).to_dict(),
        "genre_concentration": float(top3_share),
    }
