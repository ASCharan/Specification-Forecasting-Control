#!/usr/bin/env python3
"""Run the full forecasting comparison for one city and horizon.

Every model in one invocation is built from ONE feature matrix, scored under
ONE lag convention, and evaluated on ONE set of test rows. That is what makes
the resulting table a like-for-like comparison; see the paper's Section
"A one-position change in lag indexing manufactures a deep-learning advantage"
for what happens when it is not.

Usage:
  python scripts/02_run_forecasting.py --city Delhi --horizon 1  --seeds 0 1 2
  python scripts/02_run_forecasting.py --city Delhi --horizon 24 --seeds 0 1 2 3 4
  python scripts/02_run_forecasting.py --city Delhi --horizon 1  --smoke

Diagnostic only (reproduces the invalid mixed-convention table):
  python scripts/02_run_forecasting.py --city Delhi --horizon 1 --lag-offset 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urbanclimate import baselines, data
from urbanclimate.forecasting import train_neural
from urbanclimate.utils import checkpoint_path, save_json, scores, timer

NEURAL = ["cmod", "mlp64", "mlp128", "lstm", "transformer"]
WINDOW = 24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="Delhi")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--models", nargs="+", default=NEURAL)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--skip-sarima", action="store_true")
    ap.add_argument("--lag-offset", type=int, default=data.LAG_OFFSET,
                    choices=[0, 1],
                    help="1 = lag k is series[t] (default, reported results). "
                         "0 = lag k is series[t-1], withholds the most recent "
                         "observation. DIAGNOSTIC ONLY. Never mix the two "
                         "within a table.")
    ap.add_argument("--align-test-rows", dest="align", action="store_true",
                    default=True,
                    help="Score every model on the rows the sequence models "
                         "can reach (default: on).")
    ap.add_argument("--no-align-test-rows", dest="align", action="store_false")
    ap.add_argument("--no-met", action="store_true",
                    help="Drop temperature, RH and wind (ablation).")
    ap.add_argument("--no-traffic", action="store_true",
                    help="Drop the traffic proxy (ablation).")
    ap.add_argument("--tag", default="", help="Suffix for the output filename.")
    ap.add_argument("--smoke", action="store_true",
                    help="2000 rows, 3 epochs, 1 seed, no SARIMA")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    args = ap.parse_args()

    if args.smoke:
        args.epochs, args.seeds, args.skip_sarima = 3, [0], True

    if args.lag_offset != data.LAG_OFFSET:
        print(f"!! DIAGNOSTIC RUN: lag_offset={args.lag_offset} differs from "
              f"the reported convention ({data.LAG_OFFSET}). Results from this "
              f"run must not be placed in the same table as default-convention "
              f"results.", flush=True)

    proc = Path("data/processed") / f"{args.city.lower()}_aqi.csv"
    if not proc.exists():
        raise SystemExit(f"missing {proc} -- run scripts/01_build_dataset.py first")
    series = pd.read_csv(proc, index_col=0, parse_dates=True)["aqi"]
    if args.smoke:
        series = series.iloc[:2000]

    X, y = data.make_features(
        series, horizon=args.horizon, lag_offset=args.lag_offset,
        use_met=not args.no_met, use_traffic=not args.no_traffic,
    )
    tr, va, te = data.chrono_split(len(X))
    X_np, y_np = X.to_numpy(np.float32), y.to_numpy(np.float32)

    # Sequence models consume a 24-row window and therefore cannot score the
    # first WINDOW-1 rows of the test split. Align every model onto that
    # common suffix so the table compares like with like.
    trim = WINDOW - 1 if args.align else 0
    y_te_full = y_np[te]
    y_te = y_te_full[trim:]
    print(f"{args.city} h={args.horizon} lag_offset={args.lag_offset}: "
          f"{len(series)} hours, features {X.shape[1]}, "
          f"train/val/test = {tr.stop}/{va.stop - va.start}/{len(y_te_full)}, "
          f"scored on {len(y_te)} rows (align={args.align})", flush=True)

    results, preds, times = {}, {}, {}

    def add(name, pred_full):
        """Score a full-length test prediction on the common rows."""
        p = np.asarray(pred_full, np.float32)[trim:]
        results[name] = scores(y_te, p)
        preds[name] = p

    # --- naive and linear baselines ------------------------------------
    add("climatology", baselines.climatology(y_np[tr], len(y_te_full)))
    lag1 = X["aqi_lag1"].to_numpy(np.float32)[te]
    add("persistence", lag1)
    # At h=24 the seasonal-naive rule (series[t+h-24]) coincides with
    # persistence (series[t]); at h=1 it is the lag-24 column.
    add("seasonal_naive",
        lag1 if args.horizon == 24 else X["aqi_lag24"].to_numpy(np.float32)[te])

    with timer("ridge", times):
        pred_ridge, _ = baselines.ridge_forecast(X_np[tr], y_np[tr], X_np[te])
    add("ridge", pred_ridge)

    # --- seasonal ARIMA / SARIMAX --------------------------------------
    # NOTE: this path fits on the raw series rather than the feature matrix,
    # so its rows are offset from the neural models'. We re-align by matching
    # the tail, which is the only exactly comparable segment.
    if not args.skip_sarima:
        vals = series.to_numpy(float)
        cut = int(len(vals) * 0.8)
        try:
            with timer("sarima", times):
                p = baselines.sarima_forecast(vals[:cut], vals[cut:], args.horizon)
            n = min(len(p), len(y_te))
            results["sarima"] = scores(y_te[-n:], np.asarray(p)[-n:])
            results["sarima"]["n_scored"] = int(n)
        except Exception as exc:                      # pragma: no cover
            print(f"  sarima failed: {exc}", flush=True)

    # --- neural models --------------------------------------------------
    for name in args.models:
        per_seed, first_pred = [], None
        for seed in args.seeds:
            with timer(f"{name} seed{seed}", times):
                m, pred, state, hist = train_neural(
                    name, X_np[tr], y_np[tr], X_np[va], y_np[va],
                    X_np[te], y_te_full, seed=seed, epochs=args.epochs,
                    window=WINDOW,
                )
            # Sequence models already return WINDOW-1 fewer rows.
            p = np.asarray(pred, np.float32)
            p = p if len(p) == len(y_te) else p[trim:]
            m_aligned = scores(y_te, p)
            m_aligned.update(epochs_run=m["epochs_run"],
                             train_seconds=m["train_seconds"],
                             n_params=m["n_params"])
            per_seed.append(m_aligned)
            if first_pred is None:
                first_pred = p
            ck = checkpoint_path(args.ckpt_dir, args.city, name, args.horizon, seed)
            torch.save(
                {"model": name, "city": args.city, "horizon": args.horizon,
                 "seed": seed, "state_dict": state, "metrics": m_aligned,
                 "val_history": hist, "n_features": X_np.shape[1],
                 "lag_offset": args.lag_offset}, ck)
            print(f"  saved {ck}", flush=True)
        rm = np.array([s["rmse"] for s in per_seed])
        results[name] = {
            "rmse_mean": float(rm.mean()), "rmse_std": float(rm.std()),
            "rmse_min": float(rm.min()), "rmse_max": float(rm.max()),
            "mae_mean": float(np.mean([s["mae"] for s in per_seed])),
            "r2_mean": float(np.mean([s["r2"] for s in per_seed])),
            "train_seconds_mean": float(
                np.mean([s["train_seconds"] for s in per_seed])),
            "epochs_run_mean": float(np.mean([s["epochs_run"] for s in per_seed])),
            "n_params": per_seed[0]["n_params"], "per_seed": per_seed,
        }
        preds[name] = first_pred

    # Predictions are saved so that scripts/07_paired_bootstrap.py can run
    # without retraining.
    pdir = Path("results/predictions")
    pdir.mkdir(parents=True, exist_ok=True)
    tag = (f"{args.city.lower()}_h{args.horizon}_off{args.lag_offset}"
           + ("_smoke" if args.smoke else "") + (f"_{args.tag}" if args.tag else ""))
    np.savez_compressed(pdir / f"{tag}.npz", y_true=y_te,
                        **{k: v for k, v in preds.items() if v is not None})

    save_json({"config": vars(args), "lag_offset": args.lag_offset,
               "aligned": args.align, "n_scored": int(len(y_te)),
               "results": results, "seconds": times},
              f"results/forecast_{tag}.json")

    print("\n" + "=" * 64)
    print(f"{'model':<16}{'RMSE':>10}{'MAE':>10}{'R2':>9}{'train s':>10}")
    print("-" * 64)
    for k, v in sorted(results.items(),
                       key=lambda kv: kv[1].get("rmse", kv[1].get("rmse_mean", 1e9)),
                       reverse=True):
        r = v.get("rmse", v.get("rmse_mean"))
        m = v.get("mae", v.get("mae_mean"))
        q = v.get("r2", v.get("r2_mean"))
        t = v.get("train_seconds_mean", 0.0)
        print(f"{k:<16}{r:>10.2f}{m:>10.2f}{q:>9.3f}{t:>10.1f}")
    print("=" * 64)
    print(f"wrote results/forecast_{tag}.json")


if __name__ == "__main__":
    main()
