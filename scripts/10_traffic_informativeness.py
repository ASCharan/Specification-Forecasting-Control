#!/usr/bin/env python3
"""How informative would a measured traffic series have to be to matter?

The synthetic traffic proxy contributes nothing (input ablation). The natural
objection is that a REAL vehicle-density series, from a camera pipeline, would
behave differently. This script answers that quantitatively without needing the
camera, by constructing traffic covariates of controlled explanatory power and
measuring the forecast response.

Design. Fit a lag-only ridge model on the training split and compute its
residual r(t) = y(t+h) - yhat_lags(t) on every row. Standardise r using
training statistics only. Then build

    T_rho(t) = rho * z(t) + sqrt(1 - rho^2) * eps(t),   eps ~ N(0,1)

so that rho is exactly the correlation between the synthetic covariate and the
part of the target that the AQI lags cannot explain. rho = 0 reproduces an
uninformative proxy; rho = 1 would be an oracle. Sweeping rho and refitting
gives the response curve of forecast skill to traffic information quality, and
therefore the threshold a real sensor would have to clear.

This is a sensitivity analysis, not a claim about any particular city's
traffic. It bounds what the camera pipeline could buy.

Usage:
  python scripts/10_traffic_informativeness.py --city Delhi --horizon 24
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urbanclimate import baselines, data
from urbanclimate.forecasting import train_neural
from urbanclimate.utils import save_json, scores

WINDOW = 24
LAGCOLS = ["aqi_lag1", "aqi_lag2", "aqi_lag3", "aqi_lag24"]


def build(city, horizon):
    s = pd.read_csv(Path("data/processed") / f"{city.lower()}_aqi.csv",
                    index_col=0, parse_dates=True)["aqi"]
    X, y = data.make_features(s, horizon=horizon, lag_offset=data.LAG_OFFSET)
    tr, va, te = data.chrono_split(len(X))
    return X, y.to_numpy(np.float32), tr, va, te


def residual_signal(X, y, tr):
    """Part of the target the AQI lags cannot explain, standardised on train."""
    L = X[LAGCOLS].to_numpy(np.float32)
    pred_tr, model = baselines.ridge_forecast(L[tr], y[tr], L[tr])
    pred_all, _ = baselines.ridge_forecast(L[tr], y[tr], L)
    r = y - pred_all
    mu, sd = float(r[tr].mean()), float(r[tr].std())
    return ((r - mu) / sd).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="Delhi")
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--rhos", type=float, nargs="+",
                    default=[0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--neural-rhos", type=float, nargs="+",
                    default=[0.0, 0.10, 0.30, 0.50])
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    X, y, tr, va, te = build(args.city, args.horizon)
    z = residual_signal(X, y, tr)
    trim = WINDOW - 1
    y_te_full = y[te]
    y_te = y_te_full[trim:]
    rng = np.random.default_rng(0)
    eps = rng.standard_normal(len(z)).astype(np.float32)

    base_cols = [c for c in X.columns if c != "traffic"]
    out = {"city": args.city, "horizon": args.horizon,
           "n_scored": int(len(y_te)), "ridge": {}, "cmod": {}}

    print(f"{args.city} h={args.horizon}: sweeping traffic informativeness\n")
    print(f"{'rho':>6}{'ridge RMSE':>13}{'gain %':>9}")
    print("-" * 30)
    for rho in args.rhos:
        T = (rho * z + np.sqrt(max(0.0, 1 - rho ** 2)) * eps).astype(np.float32)
        Xr = X.copy()
        Xr["traffic"] = T
        Xn = Xr[base_cols + ["traffic"]].to_numpy(np.float32)
        p, _ = baselines.ridge_forecast(Xn[tr], y[tr], Xn[te])
        m = scores(y_te, p[trim:])
        out["ridge"][f"{rho:.2f}"] = m
        base = out["ridge"]["0.00"]["rmse"]
        print(f"{rho:>6.2f}{m['rmse']:>13.2f}{100*(base-m['rmse'])/base:>9.2f}")

    print(f"\n{'rho':>6}{'C-Mod RMSE':>13}{'sd':>7}{'gain %':>9}")
    print("-" * 37)
    for rho in args.neural_rhos:
        T = (rho * z + np.sqrt(max(0.0, 1 - rho ** 2)) * eps).astype(np.float32)
        Xr = X.copy()
        Xr["traffic"] = T
        Xn = Xr[base_cols + ["traffic"]].to_numpy(np.float32)
        per = []
        for sd in args.seeds:
            mm, pred, _, _ = train_neural(
                "cmod", Xn[tr], y[tr], Xn[va], y[va], Xn[te], y_te_full,
                seed=sd, epochs=args.epochs, window=WINDOW)
            p = np.asarray(pred, np.float32)
            per.append(scores(y_te, p if len(p) == len(y_te) else p[trim:]))
        rm = np.array([q["rmse"] for q in per])
        out["cmod"][f"{rho:.2f}"] = {
            "rmse_mean": float(rm.mean()), "rmse_std": float(rm.std()),
            "r2_mean": float(np.mean([q["r2"] for q in per])),
            "n_seeds": len(per)}
        b = out["cmod"]["0.00"]["rmse_mean"]
        print(f"{rho:>6.2f}{rm.mean():>13.2f}{rm.std():>7.2f}"
              f"{100*(b-rm.mean())/b:>9.2f}")

    # Threshold: rho at which the ridge gain first exceeds the C-Mod seed
    # standard deviation at rho = 0, i.e. the point at which a traffic series
    # would move the result by more than run-to-run noise.
    noise = out["cmod"]["0.00"]["rmse_std"]
    b = out["ridge"]["0.00"]["rmse"]
    rhos = sorted(float(k) for k in out["ridge"])
    gains = [b - out["ridge"][f"{r:.2f}"]["rmse"] for r in rhos]

    thr_grid, thr_interp = None, None
    for i, (r, g) in enumerate(zip(rhos, gains)):
        if g > noise:
            thr_grid = r
            # The swept grid is coarse, so report the crossing point itself:
            # linear interpolation between the last rho below the noise floor
            # and the first rho above it. This interpolated value is the one
            # quoted in the manuscript.
            if i == 0:
                thr_interp = r
            else:
                r0, g0 = rhos[i - 1], gains[i - 1]
                thr_interp = r0 + (r - r0) * (noise - g0) / (g - g0)
            break

    # Validity precondition. The covariate is built from the residual standardised
    # on the training split, so the sweep is only interpretable if the residual
    # variance is roughly stationary across the split. Where it is not, the ridge
    # coefficient fitted on train is mis-scaled on test and the measured gain can
    # be negative -- see Mumbai, whose training window straddles the 2020-21
    # period and whose residual variance halves by the test window.
    L = X[LAGCOLS].to_numpy(np.float32)
    pred_all, _ = baselines.ridge_forecast(L[tr], y[tr], L)
    resid = y - pred_all
    sd_ratio = float(resid[te].std() / resid[tr].std())
    out["residual_sd_ratio_test_over_train"] = sd_ratio
    out["sweep_valid"] = bool(0.8 <= sd_ratio <= 1.25)
    if not out["sweep_valid"]:
        print(f"\n!! residual sd ratio test/train = {sd_ratio:.2f}, outside [0.80, 1.25]. "
              f"The residual distribution shifts across the split, so the constructed "
              f"covariate does not carry its nominal rho on the test rows and this "
              f"sweep should not be read as an informativeness bound.")

    out["seed_noise_rmse"] = noise
    out["rho_to_exceed_seed_noise"] = thr_grid
    out["rho_to_exceed_seed_noise_interp"] = (
        None if thr_interp is None else round(thr_interp, 4))
    print(f"\nseed noise (C-Mod sd at rho=0 covariate): {noise:.2f} RMSE")
    print(f"smallest swept rho whose gain exceeds that noise: {thr_grid}")
    print(f"interpolated crossing point (reported in the paper): "
          f"{thr_interp:.3f}" if thr_interp is not None else "no crossing")

    tag = f"{args.city.lower()}_h{args.horizon}"
    save_json(out, f"results/traffic_informativeness_{tag}.json")
    print(f"\nwrote results/traffic_informativeness_{tag}.json")


if __name__ == "__main__":
    main()
