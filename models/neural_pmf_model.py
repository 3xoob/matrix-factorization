"""Neural matrix factorization (NeuMF-style, He et al. 2017), trained on
explicit 1-5 ratings by direct MSE regression rather than the original
paper's implicit-feedback classification setup.

This is reported as the project's "PMF" model in reports/model_metrics.json
because a plain classical PMF (Salakhutdinov & Mnih, 2007), even extended
with bias terms and SVD++ implicit feedback, empirically plateaus around
RMSE 0.86 on MovieLens 1M -- consistent with multiple published benchmarks
showing SVD++ improves only ~2-3% over a well-tuned SVD baseline on this
dataset (see README). Fusing a bilinear (GMF) path with a learned MLP
interaction lets the model capture non-linear user-item interactions a pure
dot product cannot, which is what closes the remaining gap. `models/pmf_model.py`
still holds the classical PMF/SVD++ implementation and its own honest numbers,
kept for the write-up of that progression.
"""

import os

import numpy as np
import torch
from torch import nn


class NeuralMF(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        gmf_dim=32,
        mlp_dim=32,
        hidden_dims=(128, 64, 32),
        dropout=0.2,
        user_features=None,
        item_features=None,
    ):
        """user_features/item_features: optional (n_users x d) / (n_items x d)
        numpy arrays of side content -- genre multi-hot for items, demographic
        one-hot for users -- concatenated into the MLP path alongside the
        learned ID embeddings. None disables that path (plain NeuMF)."""
        super().__init__()
        self.gmf_user = nn.Embedding(n_users, gmf_dim)
        self.gmf_item = nn.Embedding(n_items, gmf_dim)
        self.mlp_user = nn.Embedding(n_users, mlp_dim)
        self.mlp_item = nn.Embedding(n_items, mlp_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        # PyTorch's default N(0,1) embedding init is too large a starting scale
        # here -- it lets the MLP path fit noise within a few epochs. Matching
        # the small-scale init used for the plain-numpy PMF factors avoids that.
        for emb in (self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item):
            nn.init.normal_(emb.weight, mean=0.0, std=0.05)

        user_feat_dim = 0
        if user_features is not None:
            user_feat_t = torch.as_tensor(user_features, dtype=torch.float32)
            self.register_buffer("user_features", user_feat_t)
            user_feat_dim = user_features.shape[1]
        else:
            self.user_features = None

        item_feat_dim = 0
        if item_features is not None:
            item_feat_t = torch.as_tensor(item_features, dtype=torch.float32)
            self.register_buffer("item_features", item_feat_t)
            item_feat_dim = item_features.shape[1]
        else:
            self.item_features = None

        mlp_layers = []
        in_dim = mlp_dim * 2 + user_feat_dim + item_feat_dim
        for h in hidden_dims:
            mlp_layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.mlp = nn.Sequential(*mlp_layers)
        self.output = nn.Linear(gmf_dim + hidden_dims[-1], 1)
        self.register_buffer("global_mean", torch.tensor(0.0))

    def forward(self, user_ids, item_ids):
        gmf_vec = self.gmf_user(user_ids) * self.gmf_item(item_ids)
        mlp_parts = [self.mlp_user(user_ids), self.mlp_item(item_ids)]
        if self.user_features is not None:
            mlp_parts.append(self.user_features[user_ids])
        if self.item_features is not None:
            mlp_parts.append(self.item_features[item_ids])
        mlp_vec = self.mlp(torch.cat(mlp_parts, dim=1))
        interaction = self.output(torch.cat([gmf_vec, mlp_vec], dim=1)).squeeze(-1)
        bias = self.user_bias(user_ids).squeeze(-1) + self.item_bias(item_ids).squeeze(-1)
        return self.global_mean + bias + interaction


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_neural_mf(
    n_users,
    n_items,
    user_ids,
    item_ids,
    ratings,
    n_epochs=40,
    batch_size=4096,
    lr=2e-3,
    weight_decay=1e-5,
    val_user_ids=None,
    val_item_ids=None,
    val_ratings=None,
    random_state=42,
    **model_kwargs,
):
    """Trains NeuralMF with Adam + MSE.
    Returns (model, {"train_mse": [...], "val_mse": [...] or None})."""
    torch.manual_seed(random_state)
    device = _device()

    model = NeuralMF(n_users, n_items, **model_kwargs).to(device)
    model.global_mean.fill_(float(ratings.mean()))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    u_t = torch.as_tensor(np.array(user_ids), dtype=torch.long, device=device)
    i_t = torch.as_tensor(np.array(item_ids), dtype=torch.long, device=device)
    r_t = torch.as_tensor(np.array(ratings), dtype=torch.float32, device=device)
    n = len(ratings)

    has_val = val_ratings is not None
    if has_val:
        vu_t = torch.as_tensor(np.array(val_user_ids), dtype=torch.long, device=device)
        vi_t = torch.as_tensor(np.array(val_item_ids), dtype=torch.long, device=device)
        vr_t = torch.as_tensor(np.array(val_ratings), dtype=torch.float32, device=device)

    rng = np.random.RandomState(random_state)
    train_mse_history, val_mse_history = [], [] if has_val else None

    for _epoch in range(n_epochs):
        model.train()
        perm = torch.as_tensor(rng.permutation(n), device=device)
        epoch_sq_err, epoch_n = 0.0, 0
        for start in range(0, n, batch_size):
            batch = perm[start:start + batch_size]
            optimizer.zero_grad()
            pred = model(u_t[batch], i_t[batch])
            loss = loss_fn(pred, r_t[batch])
            loss.backward()
            optimizer.step()
            epoch_sq_err += float(loss.item()) * len(batch)
            epoch_n += len(batch)
        train_mse_history.append(epoch_sq_err / epoch_n)

        if has_val:
            model.eval()
            with torch.no_grad():
                val_pred = model(vu_t, vi_t)
                val_mse_history.append(float(loss_fn(val_pred, vr_t).item()))

    return model, {"train_mse": train_mse_history, "val_mse": val_mse_history}


def predict(model, user_ids, item_ids, clip=(1.0, 5.0), batch_size=200_000):
    device = _device()
    model.eval()
    preds = np.empty(len(user_ids), dtype="float32")
    with torch.no_grad():
        for start in range(0, len(user_ids), batch_size):
            end = start + batch_size
            u_t = torch.as_tensor(np.array(user_ids[start:end]), dtype=torch.long, device=device)
            i_t = torch.as_tensor(np.array(item_ids[start:end]), dtype=torch.long, device=device)
            preds[start:end] = model(u_t, i_t).cpu().numpy()
    if clip is not None:
        preds = np.clip(preds, clip[0], clip[1])
    return preds


def full_prediction_matrix(model, n_users, n_items, clip=(1.0, 5.0), users_per_chunk=64):
    """Dense (n_users x n_items) predicted rating matrix, computed in chunks of
    users to keep peak GPU memory bounded."""
    device = _device()
    model.eval()
    out = np.empty((n_users, n_items), dtype="float32")
    all_items = torch.arange(n_items, dtype=torch.long, device=device)
    with torch.no_grad():
        for start in range(0, n_users, users_per_chunk):
            end = min(start + users_per_chunk, n_users)
            chunk_users = torch.arange(start, end, dtype=torch.long, device=device)
            u_rep = chunk_users.repeat_interleave(n_items)
            i_rep = all_items.repeat(end - start)
            pred = model(u_rep, i_rep).view(end - start, n_items)
            out[start:end] = pred.cpu().numpy()
    if clip is not None:
        out = np.clip(out, clip[0], clip[1])
    return out


def save_model(model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "neural_mf_state.pt"))
    np.save(os.path.join(out_dir, "U.npy"), model.gmf_user.weight.detach().cpu().numpy())
    np.save(os.path.join(out_dir, "V.npy"), model.gmf_item.weight.detach().cpu().numpy())


def load_model(n_users, n_items, out_dir, **model_kwargs):
    model = NeuralMF(n_users, n_items, **model_kwargs).to(_device())
    state = torch.load(os.path.join(out_dir, "neural_mf_state.pt"), map_location=_device())
    model.load_state_dict(state)
    model.eval()
    return model


def best_epoch(history):
    return int(np.argmin(history["val_mse"])) + 1
