"""A small ensemble of NeuralMF models (different seeds/architectures) plus a
linear blend with the classical SVD/PMF predictions.

Individually, PMF/SVD++ plateau around RMSE 0.86 on MovieLens 1M (consistent
with published benchmarks -- see README), and a single NeuralMF model
overfits fast on this dataset size. Averaging several independently-trained
NeuralMF models cancels out their (largely uncorrelated) overfitting noise,
and blending the result with the classical PMF/SVD predictions adds a
genuinely different error pattern on top -- this combination is what closes
the gap to the project's stricter accuracy target.
"""

import os

import numpy as np

from models.neural_pmf_model import (
    fit_neural_mf,
    full_prediction_matrix,
    load_model,
    predict,
    save_model,
)

# A handful of distinct architectures/learning rates, each trained under
# several random seeds -- diversity in *architecture*, not just init, is what
# keeps ensemble members' errors from all being correlated with each other.
ENSEMBLE_CONFIGS = [
    dict(
        lr=7e-4, weight_decay=1e-5, dropout=0.25,
        gmf_dim=32, mlp_dim=32, hidden_dims=(128, 64, 32), n_epochs=10,
    ),
    dict(
        lr=5e-4, weight_decay=2e-5, dropout=0.2,
        gmf_dim=24, mlp_dim=24, hidden_dims=(96, 48, 24), n_epochs=14,
    ),
    dict(
        lr=1e-3, weight_decay=1e-5, dropout=0.3,
        gmf_dim=40, mlp_dim=40, hidden_dims=(160, 80, 40), n_epochs=8,
    ),
    dict(
        lr=7e-4, weight_decay=1e-5, dropout=0.3,
        gmf_dim=64, mlp_dim=64, hidden_dims=(256, 128, 64), n_epochs=9,
    ),
]
ENSEMBLE_REPEATS = 5  # -> 20 members total: validated during tuning as enough to reach the plateau


def train_ensemble(
    n_users, n_items, user_ids, item_ids, ratings, seed_offset=0,
    user_features=None, item_features=None,
):
    """Trains one NeuralMF per (config, repeat) combination. user_features/
    item_features (genre/demographic side content, see utils/side_features.py)
    are passed identically to every member.

    Returns (models, architectures) -- architectures holds each member's
    gmf_dim/mlp_dim/hidden_dims, since members intentionally differ and that
    has to be known again to reconstruct them with load_ensemble()."""
    models, architectures = [], []
    seed = seed_offset
    for _ in range(ENSEMBLE_REPEATS):
        for cfg in ENSEMBLE_CONFIGS:
            seed += 1
            cfg = dict(cfg)
            n_epochs = cfg.pop("n_epochs")
            arch = {k: cfg[k] for k in ("gmf_dim", "mlp_dim", "hidden_dims")}
            model, _ = fit_neural_mf(
                n_users, n_items, user_ids, item_ids, ratings, n_epochs=n_epochs,
                batch_size=4096, random_state=seed,
                user_features=user_features, item_features=item_features, **cfg,
            )
            models.append(model)
            architectures.append(arch)
    return models, architectures


def ensemble_predict(models, user_ids, item_ids):
    preds = np.stack([predict(m, user_ids, item_ids, clip=None) for m in models], axis=0)
    return preds.mean(axis=0)


def ensemble_full_matrix(models, n_users, n_items, clip=(1.0, 5.0)):
    mats = [full_prediction_matrix(m, n_users, n_items, clip=None) for m in models]
    out = np.stack(mats, axis=0).mean(axis=0)
    if clip is not None:
        out = np.clip(out, clip[0], clip[1])
    return out.astype("float32")


def save_ensemble(models, architectures, out_dir):
    import json

    os.makedirs(out_dir, exist_ok=True)
    for idx, (model, arch) in enumerate(zip(models, architectures)):
        member_dir = os.path.join(out_dir, f"member_{idx}")
        save_model(model, member_dir)
        with open(os.path.join(member_dir, "architecture.json"), "w") as f:
            json.dump(arch, f)


def load_ensemble(n_users, n_items, out_dir, user_features=None, item_features=None):
    import json

    def is_member_dir(d):
        return d.startswith("member_") and os.path.isdir(os.path.join(out_dir, d))

    member_dirs = sorted(d for d in os.listdir(out_dir) if is_member_dir(d))
    models = []
    for d in member_dirs:
        member_dir = os.path.join(out_dir, d)
        with open(os.path.join(member_dir, "architecture.json")) as f:
            arch = json.load(f)
        arch["hidden_dims"] = tuple(arch["hidden_dims"])
        models.append(
            load_model(
                n_users, n_items, member_dir,
                user_features=user_features, item_features=item_features, **arch,
            )
        )
    return models


def fit_blend_weights(component_predictions, actual_ratings):
    """component_predictions: dict[name -> 1D array]. Ordinary least squares
    over the components plus an intercept. Returns {"names": [...], "weights": [...]}."""
    names = list(component_predictions.keys())
    X = np.column_stack([component_predictions[n] for n in names] + [np.ones_like(actual_ratings)])
    coef, *_ = np.linalg.lstsq(X, actual_ratings, rcond=None)
    return {"names": names, "weights": coef.tolist()}


def apply_blend(blend, component_predictions):
    """component_predictions: dict[name -> array] (1D ratings vector OR 2D dense
    matrix, as long as shapes match across components)."""
    names, weights = blend["names"], blend["weights"]
    out = weights[-1]
    for name, w in zip(names, weights[:-1]):
        out = out + w * component_predictions[name]
    return out


def save_blend(blend, path):
    import json

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(blend, f, indent=2)


def load_blend(path):
    import json

    with open(path) as f:
        return json.load(f)
