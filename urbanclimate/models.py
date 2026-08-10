"""Neural forecasters: C-Mod (Neural ODE), MLP, LSTM, Transformer encoder.

The ODE solver is a fixed-step RK4 implemented here so that the package has no
hard dependency on torchdiffeq. Set `use_torchdiffeq=True` on CMod to use the
adjoint solver instead when the package is available; both reproduce the same
forward pass to solver tolerance.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ODEFunc(nn.Module):
    """Learned latent dynamics f_theta with bounded (tanh) output."""

    def __init__(self, latent: int = 32, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, hidden), nn.Tanh(), nn.Linear(hidden, latent), nn.Tanh()
        )

    def forward(self, t, z):  # signature matches torchdiffeq
        return self.net(z)


def rk4(func, z0, t0: float, t1: float, steps: int):
    """Fixed-step classical Runge-Kutta integration of dz/dt = func(t, z)."""
    h = (t1 - t0) / steps
    z, t = z0, t0
    for _ in range(steps):
        k1 = func(t, z)
        k2 = func(t + h / 2, z + h / 2 * k1)
        k3 = func(t + h / 2, z + h / 2 * k2)
        k4 = func(t + h, z + h * k3)
        z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + h
    return z


class CMod(nn.Module):
    """Continuous-time forecaster: encode, integrate over [0, 1], decode."""

    def __init__(
        self,
        n_features: int,
        latent: int = 32,
        hidden: int = 64,
        solver_steps: int = 4,
        use_torchdiffeq: bool = False,
    ):
        super().__init__()
        self.encoder = nn.Linear(n_features, latent)
        self.odefunc = ODEFunc(latent, hidden)
        self.decoder = nn.Linear(latent, 1)
        self.solver_steps = solver_steps
        self.use_torchdiffeq = use_torchdiffeq

    def forward(self, x):
        z0 = self.encoder(x)
        if self.use_torchdiffeq:
            from torchdiffeq import odeint

            t = torch.tensor([0.0, 1.0], device=x.device, dtype=x.dtype)
            zt = odeint(self.odefunc, z0, t, method="rk4",
                        options={"step_size": 1.0 / self.solver_steps})[-1]
        else:
            zt = rk4(self.odefunc, z0, 0.0, 1.0, self.solver_steps)
        return self.decoder(zt).squeeze(-1)


class MLP(nn.Module):
    """Feedforward baseline over the same tabular features as C-Mod."""

    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Linear(n_features, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LSTMForecaster(nn.Module):
    """Single-layer LSTM over a 24-hour input window."""

    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # x: (batch, window, features)
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class TransformerForecaster(nn.Module):
    """Two-layer Transformer encoder over a 24-hour input window."""

    def __init__(self, n_features: int, d_model: int = 64, nhead: int = 4,
                 layers: int = 2, window: int = 24):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model,
            batch_first=True, dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.proj(x) + self.pos[:, : x.shape[1]]
        h = self.encoder(h)
        return self.head(h[:, -1]).squeeze(-1)


SEQUENCE_MODELS = {"lstm", "transformer"}


def build(name: str, n_features: int, window: int = 24) -> nn.Module:
    """Model factory keyed by the names used in the result tables."""
    name = name.lower()
    if name == "cmod":
        return CMod(n_features)
    if name == "mlp64":
        return MLP(n_features, hidden=64, dropout=0.0)
    if name == "mlp128":
        return MLP(n_features, hidden=128, dropout=0.2)
    if name == "lstm":
        return LSTMForecaster(n_features, hidden=64)
    if name == "transformer":
        return TransformerForecaster(n_features, window=window)
    raise ValueError(f"unknown model {name!r}")
