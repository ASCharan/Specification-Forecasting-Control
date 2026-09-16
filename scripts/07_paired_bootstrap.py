#!/usr/bin/env python3
"""Paired-bootstrap comparison between two forecasters on identical rows.

Reads the predictions saved by scripts/02_run_forecasting.py, so it never
retrains. Resamples squared-error pairs, which is the correct unit because
both models are scored on the same targets.

Usage:
  python scripts/07_paired_bootstrap.py --city Delhi --horizon 24 \
      --a cmod --b lstm
  python scripts/07_paired_bootstrap.py --city Delhi --horizon 1 --all
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urbanclimate import data
from urbanclimate.utils import paired_bootstrap, rmse, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="Delhi")
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--lag-offset", type=int, default=data.LAG_OFFSET)
    ap.add_argument("--a", default="cmod")
    ap.add_argument("--b", default="lstm")
    ap.add_argument("--all", action="store_true",
                    help="Every pair present in the prediction file.")
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args()

    f = (Path("results/predictions") /
         f"{args.city.lower()}_h{args.horizon}_off{args.lag_offset}.npz")
    if not f.exists():
        raise SystemExit(f"missing {f} -- run scripts/02_run_forecasting.py first")
    z = np.load(f)
    y = z["y_true"]
    models = [k for k in z.files if k != "y_true"]
    pairs = list(itertools.combinations(models, 2)) if args.all \
        else [(args.a, args.b)]

    out = {}
    print(f"{args.city} h={args.horizon}, n={len(y)}, {args.n_boot} resamples\n")
    print(f"{'A':<14}{'B':<14}{'RMSE A':>9}{'RMSE B':>9}"
          f"{'B - A':>9}{'95% CI':>20}")
    print("-" * 75)
    for a, b in pairs:
        if a not in z.files or b not in z.files:
            print(f"  skipping {a} vs {b}: not in prediction file")
            continue
        pt, lo, hi = paired_bootstrap(y, z[a], z[b], n_boot=args.n_boot)
        verdict = ("B worse" if lo > 0 else
                   "A worse" if hi < 0 else "indistinguishable")
        out[f"{a}_vs_{b}"] = {"rmse_a": rmse(y, z[a]), "rmse_b": rmse(y, z[b]),
                              "gap_b_minus_a": pt, "lo95": lo, "hi95": hi,
                              "verdict": verdict, "n": int(len(y))}
        print(f"{a:<14}{b:<14}{rmse(y, z[a]):>9.2f}{rmse(y, z[b]):>9.2f}"
              f"{pt:>+9.2f}   [{lo:+6.2f}, {hi:+6.2f}]  {verdict}")

    tag = f"{args.city.lower()}_h{args.horizon}_off{args.lag_offset}"
    save_json({"config": vars(args), "results": out},
              f"results/bootstrap_{tag}.json")
    print(f"\nwrote results/bootstrap_{tag}.json")


if __name__ == "__main__":
    main()
