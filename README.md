# Movie Recommender System — SVD vs PMF

A movie recommendation engine trained on the [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)
dataset, comparing a memory-based benchmark, SVD, and a PMF-based model, with an
interactive Streamlit dashboard for side-by-side recommendations.

## Contents

- [Requirements](#requirements)
- [Setup](#setup)
- [Running the pipeline](#running-the-pipeline)
- [Running the dashboard](#running-the-dashboard)
- [Architecture](#architecture)
- [Results](#results)
- [Overfitting prevention and the stopping criterion](#overfitting-prevention-and-the-stopping-criterion)
- [Interpretability](#interpretability)
- [Per-user accuracy analysis](#per-user-accuracy-analysis)
- [Project structure](#project-structure)
- [Notes & limitations](#notes--limitations)
- [License](#license)

## Requirements

- Python 3.10+
- ~1GB disk for the dataset + generated reports, ~6GB for the virtual environment (PyTorch)
- A CUDA GPU is strongly recommended. The NeuralMF ensemble step is trained twice
  per pipeline run; on CPU only, expect the pipeline to take substantially longer
  than the ~30-45 minutes it takes on a GPU (see [Running the pipeline](#running-the-pipeline))

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

The MovieLens 1M dataset (`ratings.dat`, `users.dat`, `movies.dat`) should be under
`data/`. Download it directly if it isn't already there:

```bash
curl -o /tmp/ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip -j /tmp/ml-1m.zip 'ml-1m/*' -d data/
```

## Running the pipeline

```bash
python run_pipeline.py
```

Loads and splits the data (80/20, `random_state=42`), builds and normalizes the
user-item matrix, trains the benchmark/SVD/classical-PMF models, trains a NeuralMF
ensemble and blends it in, evaluates everything, runs the interpretability and
per-user analyses, and writes every artifact under `processed/` and `reports/`.

Expect **30-45 minutes** end to end on a GPU — classical PMF's early-stopping
search plus final retrain is ~20-25 minutes of that, and the NeuralMF ensemble
(trained twice: once on a 90% split to fit honest blend weights, once on the
full split for deployment) is most of the rest.

## Running the dashboard

```bash
streamlit run app.py
```

Enter any user ID from the dataset to see their rating history, SVD and PMF
recommendations, a chart comparing the two models' predicted ratings, and an
explanation of why a specific PMF recommendation was made. Non-numeric or
out-of-range IDs are rejected with an inline error rather than crashing the app.

## Architecture

- **Benchmark** — memory-based item-item collaborative filtering (cosine similarity).
- **SVD** — truncated singular value decomposition (`scipy.sparse.linalg.svds`)
  of a bias-corrected user-item matrix. A damped user/item baseline (shrunk
  toward the global mean so sparse users/movies don't get noisy estimates) is
  subtracted before decomposition — plain zero-filling of missing ratings
  biases a low-rank reconstruction toward predicting near-zero ratings
  everywhere, since the decomposition otherwise has to fit the sparsity
  pattern itself rather than genuine rating deviations.
- **PMF** — a blend of three components:
  1. *Classical PMF*: trained from scratch with mini-batch gradient descent
     (global mean + user/item bias terms + latent factors), extending the
     Gaussian-prior formulation (Salakhutdinov & Mnih, 2007) the same way most
     production-style implementations do (e.g. Koren et al.'s Netflix Prize
     work). Training length is chosen by early stopping on a validation curve,
     not guessed.
  2. *NeuralMF ensemble*: 20 independently-trained NeuralMF models (He et al.,
     2017 — a GMF bilinear path fused with an MLP interaction path), each
     augmented with genre (movies) and demographic (users) side features fed
     into the MLP path, averaged together to cancel out each member's fast
     overfitting.
  3. A linear blend of (1), (2), and SVD, with weights fit by ordinary least
     squares on a held-out validation slice of the training data.

  `reports/model_metrics.json`'s `PMF_RMSE` is this final blended model;
  `Classical_PMF_RMSE` and `NeuralMF_ensemble_RMSE` are the first two
  ingredients on their own, kept for comparison. **Why this three-part
  design, and what didn't work along the way, is documented in
  [`docs/experiments.md`](docs/experiments.md).**

## Results

Measured on an 80/20 train/test split (`random_state=42`) of MovieLens 1M
(1,000,209 ratings, 6,040 users, 3,706 rated movies):

| Model | Test RMSE | Target |
|---|---|---|
| Benchmark (item-based CF) | 1.0057 | — (reference point) |
| SVD (k=25) | **0.8877** | ≤ 0.90 ✅ |
| Classical PMF (80 factors, 224 epochs, early-stopped) | 0.8630 | — |
| NeuralMF ensemble (20 members, + side features) alone | 0.8446 | — |
| **PMF (SVD + classical PMF + NeuralMF ensemble, blended)** | **0.8417** | ≤ 0.85 ✅ |
| PMF improvement over SVD | **5.18%** | ≥ 5% ✅ |

All three matrix-factorization-family models clear the memory-based
collaborative-filtering benchmark by a wide margin.

**On the blend-weight methodology:** weights are fit via ordinary least
squares on a genuine held-out 10% slice of the *train* split (never the test
set) — the classical PMF's own early-stopping search model is reused here for
free, and SVD/NeuralMF are separately trained on that same 90%/10% split
specifically to produce honest validation predictions. Those weights are then
applied to predictions from SVD/PMF/NeuralMF trained on the *full* train split
(the actual deployed models), and evaluated once against test.
`reports/blend_weights.json` holds the exact coefficients.

## Overfitting prevention and the stopping criterion

Classical PMF is regularized (L2 penalty on every user/item bias and latent
factor, `reg` in `PMF_PARAMS`), and the number of training epochs is not a
guess: `run_pipeline.py` first trains on 90% of the train split while tracking
MSE on the held-out 10% each epoch, then picks the epoch with the lowest
validation MSE (`best_epoch` in `models/pmf_model.py`) and retrains on the
*full* train split for exactly that many epochs. `reports/pmf_convergence.png`
plots both curves — training MSE keeps falling well past the marked stopping
point, and that growing gap is the overfitting the early stop protects against.

NeuralMF's much higher capacity overfits far faster (within 10-15 epochs on
this dataset) — dropout, L2 weight decay, and small-scale embedding
initialization all help but don't remove the effect. Rather than fight that
with a single model, each ensemble member trains for a short, fixed,
validation-checked epoch count, and averaging many independently-trained
members (different seeds and architectures) cancels out most of each one's
overfitting noise — the same bias-variance logic as random forests bagging
over-fit trees.

## Interpretability

- **Global** (`reports/factor_genre_heatmap.png`, `reports/latent_factor_examples.csv`):
  average PMF latent-factor value per genre, plus the top movies loading on each of
  the first 5 factors. Some factors have a clearly recoverable theme (e.g. one
  factor activates strongly for Animation/Children's/Musical); this is the closest
  a matrix-factorization latent dimension gets to a human-readable meaning. This
  applies to the classical-PMF ingredient only — the NeuralMF ensemble's MLP
  path mixes its embeddings nonlinearly and isn't similarly decomposable.
- **Local** (`reports/why_recommended_user_<id>.png`, also live in the dashboard):
  a specific recommendation is explained by finding the user's own highly rated
  movies whose latent vectors are most similar (cosine similarity) to the
  recommended movie's vector — "recommended because it resembles movies you already
  rated highly."

## Per-user accuracy analysis

`reports/interpretability_analysis.md` compares three users using the tuned PMF model:

1. **The most accurately predicted training user** — low per-user test RMSE.
2. **The least accurately predicted training user** — high per-user test RMSE.
3. **A simulated cold-start user** — every one of their ratings excluded from
   training, then evaluated as if they were new. (MovieLens 1M requires ≥20 ratings
   per user, so a real user landing entirely in test via random splitting has
   near-zero probability — a true "test-only" user doesn't occur naturally in this
   dataset, hence the simulation.)

The accurate/inaccurate split tracks concrete, measurable differences: rating
variance and how concentrated a user's taste is across genres (see the file for
the exact numbers). The cold-start user's predictions collapse toward the global
mean and item biases, with no personalized signal — the textbook cold-start
limitation of matrix factorization.

## Project structure

```
data/                              raw MovieLens 1M files
processed/user_item_matrix.csv     normalized (bias-removed) train user-item matrix
models/benchmark_model.py          item-based CF benchmark (cosine similarity)
models/svd_model.py                SVD training, reconstruction, RMSE
models/pmf_model.py                classical PMF (fit/predict), SVD++ option, early stopping
models/neural_pmf_model.py         NeuralMF (GMF + MLP, PyTorch) single-model training/inference
models/ensemble_model.py           NeuralMF ensemble + linear blend with SVD/PMF
utils/data_loader.py               load & clean ratings/users/movies
utils/matrix_creation.py           train/test split, matrix build + normalization
utils/recommendation.py            generate_recommendations(user_id, model, top_n=10)
utils/interpretability.py          global (factor-genre) and local (why-recommended) analysis
utils/user_analysis.py             per-user RMSE, accurate/inaccurate/cold-start user selection
utils/side_features.py             genre (movies) + demographic (users) feature matrices for NeuralMF
reports/                           metrics, plots, per-user recommendation CSVs, analysis writeup
reports/neural_ensemble/           saved NeuralMF ensemble member weights + architectures
reports/blend_weights.json         fitted SVD/PMF/NeuralMF blend coefficients
run_pipeline.py                    orchestrates the full pipeline end-to-end
app.py                             Streamlit dashboard
Movie_Recommender_System.ipynb     walkthrough notebook (EDA, all of the above)
docs/experiments.md                the investigation behind the final PMF architecture
```

## Notes & limitations

- All three models are evaluated on the same 80/20 split (`random_state=42`); the
  user-item matrix used to train SVD/the benchmark is built from the train split
  only to avoid test leakage.
- Cold-start users/movies (present in test but absent from train) fall back to the
  global/baseline mean rather than a learned signal — see the per-user analysis
  above for a direct demonstration.
- Recommendations exclude every movie a user has rated in either split, not just
  the training split, so the dashboard never resurfaces a movie the user has
  already seen.

## License

MIT — see [LICENSE](LICENSE).
