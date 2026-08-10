"""Seeding, timing, checkpoint I/O and metrics."""
from __future__ import annotations

import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (if available) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@contextmanager
def timer(label: str, sink: dict | None = None):
    """Wall-clock timer. Records seconds into sink[label] if given."""
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    if sink is not None:
        sink[label] = round(dt, 3)
    print(f"[time] {label}: {dt:.2f}s", flush=True)


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def r2(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def scores(y, yhat) -> dict:
    return {"rmse": rmse(y, yhat), "mae": mae(y, yhat), "r2": r2(y, yhat)}


def paired_bootstrap(y, pred_a, pred_b, n_boot: int = 10_000, seed: int = 0):
    """Bootstrap the RMSE gap (B minus A) on identical targets.

    Resamples squared-error pairs, which is the correct unit because both
    models are scored on the same rows. Returns (point, lo95, hi95).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    ea = (y - np.asarray(pred_a, float)) ** 2
    eb = (y - np.asarray(pred_b, float)) ** 2
    n = len(y)
    gaps = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        gaps[i] = np.sqrt(eb[idx].mean()) - np.sqrt(ea[idx].mean())
    point = np.sqrt(eb.mean()) - np.sqrt(ea.mean())
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def load_json(path):
    return json.loads(Path(path).read_text())


def checkpoint_path(root, city: str, model: str, horizon: int, seed: int) -> Path:
    p = Path(root) / city.lower() / f"h{horizon}"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{model}_seed{seed}.pt"
