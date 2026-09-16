#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CITIES = ["delhi", "mumbai", "kolkata"]
HORIZONS = [1, 24]
WINDOW = 24
LAGS = (1, 2, 3, 24)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def scoring_index(city, horizon, lag_offset=1):
    """Rebuild the timestamps of the aligned test rows, and the drop mask."""
    s = pd.read_csv(f"data/processed/{city}_aqi.csv", index_col=0,
                    parse_dates=True)["aqi"]
    m = pd.read_csv(f"data/processed/{city}_interp_mask.csv", index_col=0,
                    parse_dates=True)["interpolated"].reindex(s.index).fillna(False)

    cols = {f"aqi_lag{l}": s.shift(l - lag_offset) for l in LAGS}
    X = pd.DataFrame(cols, index=s.index)
    X["_exo"] = 0.0
    y = s.shift(-horizon)
    keep = X.notna().all(axis=1) & y.notna()
    idx = s.index[keep]
    n = len(idx)
    a = int(n * 0.6)
    b = a + int(n * 0.2)
    te_idx = idx[b:][WINDOW - 1:]

    bad = m.to_numpy(bool)
    pos = {t: i for i, t in enumerate(s.index)}
    drop = np.zeros(len(te_idx), bool)
    for j, t in enumerate(te_idx):
        i = pos[t]
        # target, the four lags, and the sequence window that ends at this row
        need = [i + horizon] + [i - (l - lag_offset) for l in LAGS]
        need += list(range(max(0, i - (WINDOW - 1) - max(LAGS)), i + 1))
        need = [k for k in need if 0 <= k < len(bad)]
        drop[j] = bool(bad[need].any())
    return te_idx, ~drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="results/predictions_crossing")
    ap.add_argument("--out", default="results/r1_interpfree.json")
    args = ap.parse_args()

    res = {}
    for city in CITIES:
        for h in HORIZONS:
            f = Path(args.pred_dir) / f"{city}_h{h}_off1_main.npz"
            if not f.exists():
                print(f"missing {f}", flush=True)
                continue
            z = np.load(f)
            y = z["y_true"].astype(float)
            te_idx, keep = scoring_index(city, h)
            if len(keep) != len(y):
                print(f"  !! row count mismatch {city} h{h}: "
                      f"{len(keep)} rebuilt vs {len(y)} stored", flush=True)
                k = min(len(keep), len(y))
                keep, y = keep[-k:], y[-k:]
            entry = {"n_full": int(len(y)), "n_interp_free": int(keep.sum()),
                     "n_dropped": int((~keep).sum()),
                     "pct_dropped": round(100 * float((~keep).mean()), 2),
                     "models": {}}

            # persistence lag-1 autocorrelation implied by the two row sets
            for name in z.files:
                if name == "y_true":
                    continue
                p = z[name].astype(float)[-len(y):]
                base, red = rmse(y, p), rmse(y[keep], p[keep])
                entry["models"][name] = {
                    "rmse_full": round(base, 3),
                    "rmse_interp_free": round(red, 3),
                    "delta_pct": round(100 * (red - base) / base, 3)}

            # aggregate per architecture over seeds
            agg = {}
            for name, v in entry["models"].items():
                base = name.split("_s")[0]
                agg.setdefault(base, []).append(v)
            entry["by_model"] = {
                k: {"rmse_full": round(float(np.mean([q["rmse_full"] for q in v])), 3),
                    "rmse_interp_free": round(
                        float(np.mean([q["rmse_interp_free"] for q in v])), 3),
                    "delta_pct": round(
                        float(np.mean([q["delta_pct"] for q in v])), 3),
                    "n": len(v)}
                for k, v in agg.items()}

            order_full = sorted(entry["by_model"], key=lambda k: entry["by_model"][k]["rmse_full"])
            order_red = sorted(entry["by_model"], key=lambda k: entry["by_model"][k]["rmse_interp_free"])
            entry["ranking_full"] = order_full
            entry["ranking_interp_free"] = order_red
            entry["ranking_unchanged"] = order_full == order_red
            res[f"{city}_h{h}"] = entry

            print(f"{city} h{h}: {entry['n_interp_free']}/{entry['n_full']} rows "
                  f"kept ({entry['pct_dropped']}% dropped), "
                  f"max |dRMSE| {max(abs(v['delta_pct']) for v in entry['by_model'].values()):.3f}%, "
                  f"ranking {'unchanged' if entry['ranking_unchanged'] else 'CHANGED'}",
                  flush=True)
            print(f"   full: {order_full}")
            if not entry["ranking_unchanged"]:
                print(f"   free: {order_red}")

    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
