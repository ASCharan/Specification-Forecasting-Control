#!/usr/bin/env python3
"""C-Mod input ablation under the uniform lag convention.

Removes input groups and reports the RMSE degradation relative to the full
model. The traffic and meteorological inputs are synthetic diurnal templates
(Methods), so what this quantifies is the value of an explicit, noisy
diurnal encoding, not the value of measured meteorology or measured traffic.

Usage:  python scripts/06_ablation.py --city Delhi --horizon 1 --seeds 0 1 2
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urbanclimate import data
from urbanclimate.forecasting import train_neural
from urbanclimate.utils import save_json, scores

WINDOW = 24
CONFIGS = [
    ("full",          dict(use_met=True,  use_traffic=True)),
    ("minus_traffic", dict(use_met=True,  use_traffic=False)),
    ("minus_met",     dict(use_met=False, use_traffic=True)),
    ("lags_only",     dict(use_met=False, use_traffic=False)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="Delhi")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--model", default="cmod")
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    series = pd.read_csv(
        Path("data/processed") / f"{args.city.lower()}_aqi.csv",
        index_col=0, parse_dates=True)["aqi"]

    out = {}
    for label, kw in CONFIGS:
        X, y = data.make_features(series, horizon=args.horizon,
                                  lag_offset=data.LAG_OFFSET, **kw)
        tr, va, te = data.chrono_split(len(X))
        Xn, yn = X.to_numpy(np.float32), y.to_numpy(np.float32)
        y_te_full = yn[te]
        y_te = y_te_full[WINDOW - 1:]
        per = []
        for seed in args.seeds:
            m, pred, _, _ = train_neural(
                args.model, Xn[tr], yn[tr], Xn[va], yn[va],
                Xn[te], y_te_full, seed=seed, epochs=args.epochs, window=WINDOW)
            p = np.asarray(pred, np.float32)
            p = p if len(p) == len(y_te) else p[WINDOW - 1:]
            per.append(scores(y_te, p))
        rm = np.array([s["rmse"] for s in per])
        out[label] = {"features": int(X.shape[1]),
                      "rmse_mean": float(rm.mean()),
                      "rmse_std": float(rm.std()),
                      "r2_mean": float(np.mean([s["r2"] for s in per])),
                      "n_seeds": len(per), "n_scored": int(len(y_te))}
        print(f"{label:<16}{X.shape[1]:>3} feat  RMSE "
              f"{rm.mean():7.2f} +- {rm.std():4.2f}  "
              f"R2 {out[label]['r2_mean']:.3f}", flush=True)

    base = out["full"]["rmse_mean"]
    for label in out:
        out[label]["delta_rmse_pct"] = round(
            100.0 * (out[label]["rmse_mean"] - base) / base, 2)

    tag = f"{args.city.lower()}_h{args.horizon}"
    save_json({"config": vars(args), "lag_offset": data.LAG_OFFSET,
               "results": out}, f"results/ablation_{tag}.json")

    print("\n" + "=" * 56)
    print(f"{'configuration':<18}{'RMSE':>10}{'R2':>9}{'dRMSE %':>10}")
    print("-" * 56)
    for label, v in out.items():
        print(f"{label:<18}{v['rmse_mean']:>10.2f}{v['r2_mean']:>9.3f}"
              f"{v['delta_rmse_pct']:>+10.2f}")
    print("=" * 56)
    print(f"wrote results/ablation_{tag}.json")


if __name__ == "__main__":
    main()
