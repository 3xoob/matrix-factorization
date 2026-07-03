"""Load and clean the raw MovieLens 1M dataset files."""

import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_ratings(data_dir=DATA_DIR):
    path = os.path.join(data_dir, "ratings.dat")
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
    )
    df = df.dropna().astype(
        {"UserID": "int64", "MovieID": "int64", "Rating": "float64", "Timestamp": "int64"}
    )
    return df.reset_index(drop=True)


def load_users(data_dir=DATA_DIR):
    path = os.path.join(data_dir, "users.dat")
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["UserID", "Gender", "Age", "Occupation", "ZipCode"],
    )
    return df.dropna().reset_index(drop=True)


def load_movies(data_dir=DATA_DIR):
    path = os.path.join(data_dir, "movies.dat")
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["MovieID", "Title", "Genres"],
    )
    return df.dropna().reset_index(drop=True)


def load_all(data_dir=DATA_DIR):
    return load_ratings(data_dir), load_users(data_dir), load_movies(data_dir)


if __name__ == "__main__":
    ratings, users, movies = load_all()
    print(f"ratings: {ratings.shape}, users: {users.shape}, movies: {movies.shape}")
