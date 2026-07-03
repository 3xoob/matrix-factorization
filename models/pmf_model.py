"""Probabilistic Matrix Factorization trained by mini-batch gradient descent.

Extends the classic PMF formulation (Salakhutdinov & Mnih, 2007 — Gaussian
priors on user/item latent factors, equivalent to L2-regularized squared
error) with a global mean and per-user/per-item bias terms, the standard
practical extension (as in Koren et al.'s Netflix Prize work) needed to
separate "how a user/movie rates on average" from genuine interaction effects.

Optionally also folds in implicit feedback (SVD++, Koren 2008): a user's
*set* of rated items -- regardless of the rating value -- is itself a signal,
captured via an extra learned item-factor matrix Y. A user's effective factor
vector becomes U[u] + (1/sqrt(|R(u)|)) * sum_{j in R(u)} Y[j], which is what
use_implicit=True adds on top of the plain biased-MF model above.
"""

import os

import numpy as np
from scipy.sparse import csr_matrix


class PMF:
    def __init__(self, n_users, n_items, n_factors=20, lr=0.01, reg=0.05, random_state=42):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.random_state = random_state

        self.global_mean = 0.0
        self.b_u = None
        self.b_i = None
        self.U = None
        self.V = None
        self.Y = None
        self.implicit_sum = None
        self.sqrt_counts = None

    def fit(
        self,
        user_ids,
        item_ids,
        ratings,
        n_epochs=40,
        batch_size=100_000,
        lr_decay=1.0,
        val_user_ids=None,
        val_item_ids=None,
        val_ratings=None,
        use_implicit=False,
    ):
        """user_ids/item_ids are 0-based integer indices; ratings is a float array.
        lr_decay < 1.0 shrinks the learning rate each epoch, which lets training
        converge more precisely later on instead of oscillating/overfitting at a
        constant step size.

        use_implicit=True enables the SVD++ implicit-feedback term: the set of
        items in (user_ids, item_ids) -- i.e. what this user rated in TRAIN,
        regardless of the rating value -- becomes an extra per-user signal
        folded into their effective factor vector.

        If val_* arrays are given (a held-out slice of TRAIN, distinct from the
        final test set), validation MSE is tracked each epoch alongside training
        MSE, so the returned history can be used both as a convergence plot and
        to pick an early-stopping epoch before the model overfits.

        Returns {"train_mse": [...], "val_mse": [...] or None}.
        """
        rng = np.random.RandomState(self.random_state)
        n = len(ratings)

        self.global_mean = float(ratings.mean())
        self.b_u = np.zeros(self.n_users)
        self.b_i = np.zeros(self.n_items)
        self.U = rng.normal(0, 0.1, (self.n_users, self.n_factors))
        self.V = rng.normal(0, 0.1, (self.n_items, self.n_factors))

        indicator_t = None
        if use_implicit:
            self.Y = rng.normal(0, 0.1, (self.n_items, self.n_factors))
            indicator = csr_matrix(
                (np.ones(n), (user_ids, item_ids)), shape=(self.n_users, self.n_items)
            )
            indicator_t = indicator.T.tocsr()
            counts = np.asarray(indicator.sum(axis=1)).flatten()
            self.sqrt_counts = np.sqrt(np.maximum(counts, 1))
        else:
            self.Y = None
            self.implicit_sum = None
            self.sqrt_counts = None

        train_mse_history = []
        val_mse_history = [] if val_ratings is not None else None
        lr = self.lr
        for _epoch in range(n_epochs):
            if use_implicit:
                # Recomputed once per epoch from the current Y (a standard,
                # efficient SVD++ simplification): predictions/gradients this
                # epoch use this epoch-start snapshot, while Y itself keeps
                # updating batch-by-batch below.
                self.implicit_sum = indicator @ self.Y
                eff_u_full = self.U + self.implicit_sum / self.sqrt_counts[:, np.newaxis]
            else:
                eff_u_full = self.U

            perm = rng.permutation(n)
            for start in range(0, n, batch_size):
                batch = perm[start:start + batch_size]
                u, i, r = user_ids[batch], item_ids[batch], ratings[batch]
                eff_u = eff_u_full[u]

                pred = (
                    self.global_mean
                    + self.b_u[u]
                    + self.b_i[i]
                    + np.sum(eff_u * self.V[i], axis=1)
                )
                err = r - pred

                grad_bu = err - self.reg * self.b_u[u]
                grad_bi = err - self.reg * self.b_i[i]
                grad_U = err[:, np.newaxis] * self.V[i] - self.reg * self.U[u]
                grad_V = err[:, np.newaxis] * eff_u - self.reg * self.V[i]

                db_u = np.zeros(self.n_users)
                db_i = np.zeros(self.n_items)
                dU = np.zeros_like(self.U)
                dV = np.zeros_like(self.V)
                cnt_u = np.zeros(self.n_users)
                cnt_i = np.zeros(self.n_items)
                np.add.at(db_u, u, grad_bu)
                np.add.at(db_i, i, grad_bi)
                np.add.at(dU, u, grad_U)
                np.add.at(dV, i, grad_V)
                np.add.at(cnt_u, u, 1)
                np.add.at(cnt_i, i, 1)

                # Average (not sum) each parameter's gradient across the times it
                # appears in this batch, so the effective step size doesn't scale
                # with batch_size or with how popular a user/item is.
                cnt_u = np.maximum(cnt_u, 1)
                cnt_i = np.maximum(cnt_i, 1)
                self.b_u += lr * db_u / cnt_u
                self.b_i += lr * db_i / cnt_i
                self.U += lr * dU / cnt_u[:, np.newaxis]
                self.V += lr * dV / cnt_i[:, np.newaxis]

                if use_implicit:
                    # Message-passing update for Y: each sample pulls Y[j] for
                    # every j the sample's user rated, toward reducing this
                    # sample's error, scaled by 1/sqrt(|R(u)|) as in SVD++.
                    coef = err / self.sqrt_counts[u]
                    message = np.zeros((self.n_users, self.n_factors))
                    np.add.at(message, u, coef[:, np.newaxis] * self.V[i])
                    n_appear = np.zeros(self.n_users)
                    np.add.at(n_appear, u, 1)

                    dY = indicator_t @ message
                    touch_weight = np.asarray(indicator_t @ n_appear).flatten()
                    touched = touch_weight > 0
                    touch_weight_safe = np.maximum(touch_weight, 1e-8)
                    self.Y[touched] += lr * (
                        dY[touched] / touch_weight_safe[touched, np.newaxis]
                        - self.reg * self.Y[touched]
                    )

            if use_implicit:
                self.implicit_sum = indicator @ self.Y

            train_pred = self.predict(user_ids, item_ids)
            train_mse_history.append(float(np.mean((ratings - train_pred) ** 2)))
            if val_ratings is not None:
                val_pred = self.predict(val_user_ids, val_item_ids)
                val_mse_history.append(float(np.mean((val_ratings - val_pred) ** 2)))
            lr *= lr_decay

        return {"train_mse": train_mse_history, "val_mse": val_mse_history}

    def _effective_user_factors(self, user_ids=None):
        if self.implicit_sum is None:
            return self.U if user_ids is None else self.U[user_ids]
        eff = self.U + self.implicit_sum / self.sqrt_counts[:, np.newaxis]
        return eff if user_ids is None else eff[user_ids]

    def predict(self, user_ids, item_ids):
        eff_u = self._effective_user_factors(user_ids)
        return (
            self.global_mean
            + self.b_u[user_ids]
            + self.b_i[item_ids]
            + np.sum(eff_u * self.V[item_ids], axis=1)
        )

    def full_prediction_matrix(self, clip=(1.0, 5.0)):
        eff_u_full = self._effective_user_factors()
        pred = (
            self.global_mean
            + self.b_u[:, np.newaxis]
            + self.b_i[np.newaxis, :]
            + eff_u_full @ self.V.T
        )
        if clip is not None:
            pred = np.clip(pred, clip[0], clip[1])
        return pred.astype("float32")


def rmse_from_predictions(predictions, ratings):
    return float(np.sqrt(np.mean((predictions - ratings) ** 2)))


def best_epoch(history):
    """1-indexed epoch with the lowest validation MSE (early-stopping point)."""
    val_mse = history["val_mse"]
    return int(np.argmin(val_mse)) + 1


def plot_convergence(history, save_path):
    import matplotlib

    matplotlib.use("Agg")  # noqa: E402 -- must precede pyplot import; standard Agg-backend pattern
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_mse"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_mse"], marker="o", markersize=3, label="Train MSE")

    if history["val_mse"] is not None:
        plt.plot(epochs, history["val_mse"], marker="s", markersize=3, label="Validation MSE")
        stop_epoch = best_epoch(history)
        plt.axvline(
            stop_epoch,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=f"Best epoch ({stop_epoch})",
        )
        plt.title("PMF Convergence — Train vs Validation MSE")
    else:
        plt.title("PMF Convergence")

    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close()


def save_factors(model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "U.npy"), model.U)
    np.save(os.path.join(out_dir, "V.npy"), model.V)
    np.save(os.path.join(out_dir, "b_u.npy"), model.b_u)
    np.save(os.path.join(out_dir, "b_i.npy"), model.b_i)
    np.save(os.path.join(out_dir, "global_mean.npy"), np.array([model.global_mean]))
    if model.implicit_sum is not None:
        np.save(os.path.join(out_dir, "Y.npy"), model.Y)
        np.save(os.path.join(out_dir, "implicit_sum.npy"), model.implicit_sum)
        np.save(os.path.join(out_dir, "sqrt_counts.npy"), model.sqrt_counts)


def load_prediction_matrix(factors_dir, clip=(1.0, 5.0)):
    """Reconstruct the full predicted rating matrix from factors saved by save_factors,
    without needing a separate (large) predictions .npy file on disk."""
    U = np.load(os.path.join(factors_dir, "U.npy"))
    V = np.load(os.path.join(factors_dir, "V.npy"))
    b_u = np.load(os.path.join(factors_dir, "b_u.npy"))
    b_i = np.load(os.path.join(factors_dir, "b_i.npy"))
    global_mean = float(np.load(os.path.join(factors_dir, "global_mean.npy"))[0])

    implicit_sum_path = os.path.join(factors_dir, "implicit_sum.npy")
    if os.path.exists(implicit_sum_path):
        implicit_sum = np.load(implicit_sum_path)
        sqrt_counts = np.load(os.path.join(factors_dir, "sqrt_counts.npy"))
        eff_u = U + implicit_sum / sqrt_counts[:, np.newaxis]
    else:
        eff_u = U

    predicted = global_mean + b_u[:, np.newaxis] + b_i[np.newaxis, :] + eff_u @ V.T
    if clip is not None:
        predicted = np.clip(predicted, clip[0], clip[1])
    return predicted.astype("float32")
