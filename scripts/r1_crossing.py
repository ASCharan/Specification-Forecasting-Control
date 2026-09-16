#!/usr/bin/env python3
"""Full crossing of architecture x lag convention x seed, all six cells.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urbanclimate import baselines, data
from urbanclimate.forecasting import train_neural
from urbanclimate.utils import scores

torch.set_num_threads(1)

NEURAL = ["cmod", "mlp64", "mlp128", "lstm", "transformer"]
WINDOW = 24
CITIES = ["Delhi", "Mumbai", "Kolkata"]
HORIZONS = [1, 24]


def load_series(city, variant):
    if variant == "balanced":
        p = Path(f"data/processed/{city.lower()}_aqi_balanced.csv")
    else:
        p = Path(f"data/processed/{city.lower()}_aqi.csv")
    return pd.read_csv(p, index_col=0, parse_dates=True)["aqi"]


def interp_free_mask(city, index):
    """True where the row uses no interpolated observation."""
    p = Path(f"data/processed/{city.lower()}_interp_mask.csv")
    m = pd.read_csv(p, index_col=0, parse_dates=True)["interpolated"]
    return m.reindex(index).fillna(False).to_numpy(bool)


def run_cell(city, horizon, lag_offset, seeds, variant, pred_dir, tfm_seeds=None, models=None):
    series = load_series(city, variant)
    X, y = data.make_features(series, horizon=horizon, lag_offset=lag_offset)
    tr, va, te = data.chrono_split(len(X))
    X_np, y_np = X.to_numpy(np.float32), y.to_numpy(np.float32)
    trim = WINDOW - 1
    y_te_full = y_np[te]
    y_te = y_te_full[trim:]
    idx_te = X.index[te][trim:]

    keep = np.ones(len(y_te), bool)
    if variant == "interpfree":
        bad = interp_free_mask(city, series.index)
        badser = pd.Series(bad, index=series.index)
        # a scoring row is dropped if the target, any lag, or (for the sequence
        # models) any row of the 24-hour input window touches an interpolated
        # value
        touched = badser.rolling(WINDOW + 24, min_periods=1).max().astype(bool)
        fwd = badser.shift(-horizon).fillna(True).astype(bool)
        flag = (touched.reindex(idx_te).fillna(True).to_numpy(bool)
                | fwd.reindex(idx_te).fillna(True).to_numpy(bool))
        keep = ~flag

    out = {"n_scored": int(keep.sum()), "n_scored_full": int(len(y_te)),
           "hours": int(len(series))}
    yk = y_te[keep]

    lag1 = X["aqi_lag1"].to_numpy(np.float32)[te][trim:]
    out["persistence"] = scores(yk, lag1[keep])
    pred_ridge, _ = baselines.ridge_forecast(X_np[tr], y_np[tr], X_np[te])
    out["ridge"] = scores(yk, np.asarray(pred_ridge, np.float32)[trim:][keep])

    store = {"y_true": yk, "persistence": lag1[keep],
             "ridge": np.asarray(pred_ridge, np.float32)[trim:][keep]}

    for name in (models or NEURAL):
        use = list(tfm_seeds) if (name == "transformer" and tfm_seeds) else list(seeds)
        per_seed = []
        for s in use:
            t0 = time.perf_counter()
            m, pred, state, hist = train_neural(
                name, X_np[tr], y_np[tr], X_np[va], y_np[va],
                X_np[te], y_te_full, seed=s, epochs=100, window=WINDOW)
            p = np.asarray(pred, np.float32)
            p = p if len(p) == len(y_te) else p[trim:]
            sc = scores(yk, p[keep])
            sc["seconds"] = round(time.perf_counter() - t0, 1)
            per_seed.append(sc)
            store[f"{name}_s{s}"] = p[keep]
            print(f"      {name} s{s}: rmse {sc['rmse']:.3f} "
                  f"[{sc['seconds']}s]", flush=True)
        r = np.array([q["rmse"] for q in per_seed])
        out[name] = {"rmse_mean": float(r.mean()), "rmse_std": float(r.std()),
                     "mae_mean": float(np.mean([q["mae"] for q in per_seed])),
                     "r2_mean": float(np.mean([q["r2"] for q in per_seed])),
                     "per_seed_rmse": [float(x) for x in r],
                     "seeds": use}

    pred_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{city.lower()}_h{horizon}_off{lag_offset}_{variant}"
    np.savez_compressed(pred_dir / f"{tag}.npz", **store)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=CITIES)
    ap.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    ap.add_argument("--lag-offsets", type=int, nargs="+", default=[1, 0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tfm-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--models", nargs="+", default=NEURAL)
    ap.add_argument("--variant", default="main",
                    choices=["main", "interpfree", "balanced"])
    ap.add_argument("--out", default="results/r1_crossing.json")
    ap.add_argument("--pred-dir", default="results/predictions_crossing")
    args = ap.parse_args()

    outp = Path(args.out)
    res = json.loads(outp.read_text()) if outp.exists() else {}
    res.setdefault("_config", vars(args))

    for city in args.cities:
        for h in args.horizons:
            for off in args.lag_offsets:
                key = f"{city.lower()}_h{h}_off{off}_{args.variant}"
                if key in res:
                    print(f"skip {key} (done)", flush=True)
                    continue
                print(f"\n>>> {key}", flush=True)
                t0 = time.perf_counter()
                res[key] = run_cell(city, h, off, args.seeds, args.variant,
                                    Path(args.pred_dir), args.tfm_seeds,
                                    args.models)
                res[key]["cell_seconds"] = round(time.perf_counter() - t0, 1)
                outp.parent.mkdir(parents=True, exist_ok=True)
                outp.write_text(json.dumps(res, indent=1))
                print(f"<<< {key} in {res[key]['cell_seconds']}s", flush=True)
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
