#!/usr/bin/env python3

Part 1 is analytic. Writing omega = 2 pi / 24, the three templates are
    Temp = 28  - 8 cos(omega t)  + eps_T
    RH   = 60  + 15 sin(omega t) + eps_H
    Wind = 3.5 + 1.5 sin(omega t) + eps_W
so the deterministic parts of RH and wind are exactly proportional and the
three of them span only the two-dimensional space {sin, cos} at the diurnal
frequency. This script measures the resulting correlations and the numerical
rank on the actual feature matrix.

Part 2 tests the interpretation. If the templates carry nothing but the
hour-of-day phasor, replacing all three with (cos omega t, sin omega t) should
not change forecast skill. The comparison is run through the unmodified
training protocol.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urbanclimate import baselines, data
from urbanclimate.forecasting import train_neural
from urbanclimate.utils import scores

torch.set_num_threads(1)
WINDOW = 24
OMEGA = 2 * np.pi / 24


def analytic(index):
    exo = data.synthetic_exogenous(index, seed=0)
    t = index.hour.to_numpy().astype(float)
    det = np.c_[28 - 8 * np.cos(OMEGA * t),
                60 + 15 * np.sin(OMEGA * t),
                3.5 + 1.5 * np.sin(OMEGA * t)]
    # confirm the closed forms match the implementation exactly
    det_impl = np.c_[28 + 8 * np.sin(2 * np.pi * (t - 6) / 24),
                     60 - 15 * np.sin(2 * np.pi * (t - 12) / 24),
                     3.5 + 1.5 * np.sin(2 * np.pi * t / 24)]
    max_dev = float(np.abs(det - det_impl).max())

    C = np.corrcoef(exo[["temp", "rh", "wind"]].to_numpy().T)
    Cd = np.corrcoef((det - det.mean(0)).T)
    sv = np.linalg.svd(det - det.mean(0), compute_uv=False)
    rank = int(np.linalg.matrix_rank(det - det.mean(0), tol=1e-8))

    # traffic template against the met templates and against hour-of-day
    traf = exo["traffic"].to_numpy()
    phasor = np.c_[np.cos(OMEGA * t), np.sin(OMEGA * t)]
    r2_traffic = float(np.corrcoef(
        traf, np.linalg.lstsq(np.c_[np.ones(len(t)), phasor], traf,
                              rcond=None)[0] @ np.c_[np.ones(len(t)), phasor].T
    )[0, 1] ** 2)
    return {
        "closed_form_max_deviation": max_dev,
        "corr_with_noise": {"temp_rh": round(float(C[0, 1]), 4),
                            "temp_wind": round(float(C[0, 2]), 4),
                            "rh_wind": round(float(C[1, 2]), 4)},
        "corr_noiseless": {"temp_rh": round(float(Cd[0, 1]), 6),
                           "temp_wind": round(float(Cd[0, 2]), 6),
                           "rh_wind": round(float(Cd[1, 2]), 6)},
        "singular_values_noiseless": [round(float(s), 4) for s in sv],
        "numerical_rank_noiseless": rank,
        "traffic_r2_on_diurnal_phasor": round(r2_traffic, 4),
    }


def build_features(series, horizon, mode):
    """mode: 'templates' (as submitted) | 'fourier' | 'lags_only'."""
    exo = data.synthetic_exogenous(series.index, seed=0)
    t = series.index.hour.to_numpy().astype(float)
    cols = {}
    if mode == "templates":
        for c in ("temp", "rh", "wind"):
            cols[c] = exo[c]
        cols["traffic"] = exo["traffic"]
    elif mode == "fourier":
        cols["cos24"] = pd.Series(np.cos(OMEGA * t), index=series.index)
        cols["sin24"] = pd.Series(np.sin(OMEGA * t), index=series.index)
    elif mode == "lags_only":
        pass
    else:
        raise ValueError(mode)
    for lag in (1, 2, 3, 24):
        cols[f"aqi_lag{lag}"] = series.shift(lag - data.LAG_OFFSET)
    X = pd.DataFrame(cols, index=series.index)
    y = series.shift(-horizon)
    keep = X.notna().all(axis=1) & y.notna()
    return X[keep], y[keep]


def run_mode(series, horizon, mode, seeds):
    X, y = build_features(series, horizon, mode)
    tr, va, te = data.chrono_split(len(X))
    X_np, y_np = X.to_numpy(np.float32), y.to_numpy(np.float32)
    trim = WINDOW - 1
    y_te_full = y_np[te]
    y_te = y_te_full[trim:]
    out = {"n_features": int(X.shape[1]), "n_scored": int(len(y_te)),
           "columns": list(X.columns)}
    pr, _ = baselines.ridge_forecast(X_np[tr], y_np[tr], X_np[te])
    out["ridge"] = scores(y_te, np.asarray(pr, np.float32)[trim:])
    per = []
    for s in seeds:
        m, pred, *_ = train_neural("cmod", X_np[tr], y_np[tr], X_np[va],
                                   y_np[va], X_np[te], y_te_full, seed=s,
                                   epochs=100, window=WINDOW)
        p = np.asarray(pred, np.float32)
        p = p if len(p) == len(y_te) else p[trim:]
        per.append(scores(y_te, p))
        print(f"    cmod s{s} ({mode}): rmse {per[-1]['rmse']:.3f}", flush=True)
    r = np.array([q["rmse"] for q in per])
    out["cmod"] = {"rmse_mean": float(r.mean()), "rmse_std": float(r.std()),
                   "per_seed_rmse": [float(x) for x in r]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="Delhi")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 24])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results/r2_templates.json")
    args = ap.parse_args()

    series = pd.read_csv(f"data/processed/{args.city.lower()}_aqi.csv",
                         index_col=0, parse_dates=True)["aqi"]
    res = {"city": args.city, "analytic": analytic(series.index)}
    a = res["analytic"]
    print(f"closed forms match implementation to {a['closed_form_max_deviation']:.2e}")
    print(f"corr(RH, wind) = {a['corr_with_noise']['rh_wind']} with noise, "
          f"{a['corr_noiseless']['rh_wind']} noiseless")
    print(f"noiseless singular values {a['singular_values_noiseless']}, "
          f"rank {a['numerical_rank_noiseless']}")
    print(f"traffic template R^2 on the diurnal phasor: "
          f"{a['traffic_r2_on_diurnal_phasor']}")

    for h in args.horizons:
        res[f"h{h}"] = {}
        for mode in ("templates", "fourier", "lags_only"):
            print(f"\n  h={h} mode={mode}", flush=True)
            res[f"h{h}"][mode] = run_mode(series, h, mode, args.seeds)
        base = res[f"h{h}"]["templates"]["cmod"]["rmse_mean"]
        for mode in ("fourier", "lags_only"):
            m = res[f"h{h}"][mode]["cmod"]
            res[f"h{h}"][mode]["delta_vs_templates_pct"] = round(
                100 * (m["rmse_mean"] - base) / base, 2)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
