"""Training and evaluation loop for the neural forecasters.

One protocol is shared by every model: Adam at 1e-3, minibatch 64, validation
based early stopping with patience 10, a cap of 100 epochs. Inputs and target
are standardised on training statistics only; metrics are reported in original
AQI units.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import models
from .utils import scores, set_seed


def make_windows(X, y, window: int = 24):
    """Stack `window` consecutive feature rows for the sequence models."""
    X, y = np.asarray(X, np.float32), np.asarray(y, np.float32)
    n = len(X) - window + 1
    idx = np.arange(window)[None, :] + np.arange(n)[:, None]
    return X[idx], y[window - 1 :]


def train_neural(
    name: str,
    X_tr, y_tr, X_va, y_va, X_te, y_te,
    seed: int = 0,
    epochs: int = 100,
    patience: int = 10,
    lr: float = 1e-3,
    batch_size: int = 64,
    window: int = 24,
    device: str = "cpu",
):
    """Train one model from one seed. Returns (metrics, predictions, state_dict)."""
    set_seed(seed)
    sequence = name.lower() in models.SEQUENCE_MODELS

    x_mu, x_sd = X_tr.mean(0), X_tr.std(0) + 1e-8
    y_mu, y_sd = y_tr.mean(), y_tr.std() + 1e-8
    nz = lambda A: ((np.asarray(A, np.float32) - x_mu) / x_sd).astype(np.float32)
    tz = lambda a: ((np.asarray(a, np.float32) - y_mu) / y_sd).astype(np.float32)

    if sequence:
        Xtr, ytr = make_windows(nz(X_tr), tz(y_tr), window)
        Xva, yva = make_windows(nz(X_va), tz(y_va), window)
        Xte, yte_raw = make_windows(nz(X_te), np.asarray(y_te, np.float32), window)
    else:
        Xtr, ytr = nz(X_tr), tz(y_tr)
        Xva, yva = nz(X_va), tz(y_va)
        Xte, yte_raw = nz(X_te), np.asarray(y_te, np.float32)

    model = models.build(name, X_tr.shape[1], window=window).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
        batch_size=batch_size, shuffle=True,
    )
    Xva_t = torch.tensor(Xva, device=device)
    yva_t = torch.tensor(yva, device=device)

    best_loss, best_state, bad, history = np.inf, None, 0, []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(Xva_t), yva_t))
        history.append(val)
        if val < best_loss - 1e-6:
            best_loss, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    train_seconds = time.perf_counter() - t0
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xte, device=device)).cpu().numpy()
    pred = pred * y_sd + y_mu

    out = scores(yte_raw, pred)
    out.update(
        epochs_run=epoch + 1,
        best_val_loss=best_loss,
        train_seconds=round(train_seconds, 2),
        n_params=sum(p.numel() for p in model.parameters()),
    )
    return out, pred, best_state, history
