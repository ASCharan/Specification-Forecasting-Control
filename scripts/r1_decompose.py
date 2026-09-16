#!/usr/bin/env python3
"""Put specification and architecture effects on one scale.

Two outputs.

1. The lag-convention penalty as a PAIRED difference. Convention A and B share
   the seed and the perturbation is deterministic, so the correct noise scale
   for the penalty is the across-seed spread of the per-seed difference, not
   the marginal spread of either arm. The submitted Table 4 reported
   single-seed penalties and its caption claimed the effect sizes were far
   larger than the seed spreads. That holds at one hour and needs correcting at
   24 hours.

2. A variance decomposition. RMSE is normalised by the persistence RMSE of the
   same cell so cities and horizons are commensurable, then the variation is
   split into a convention component, an architecture component, their
   interaction, and a seed (residual) component. Reported as standard
   deviations in percent of the persistence RMSE, which is the scale on which
   the two effects can be compared directly.
"""
import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

MODELS = ["cmod", "mlp64", "mlp128", "lstm", "transformer"]
CITIES = ["delhi", "mumbai", "kolkata"]
HORIZONS = [1, 24]


def cell(res, city, h, off, variant="main"):
    return res.get(f"{city}_h{h}_off{off}_{variant}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crossing", default="results/r1_crossing.json")
    ap.add_argument("--variant", default="main")
    ap.add_argument("--out", default="results/r1_decomposition.json")
    args = ap.parse_args()

    res = json.loads(Path(args.crossing).read_text())
    out = {"paired_convention": [], "decomposition": [], "architecture_range": []}

    for city, h in product(CITIES, HORIZONS):
        A = cell(res, city, h, 1, args.variant)
        B = cell(res, city, h, 0, args.variant)
        if A is None:
            continue
        persA = A["persistence"]["rmse"]

        # ---- paired convention penalties -------------------------------
        if B is not None:
            for m in ["persistence", "ridge"] + MODELS:
                if m not in A or m not in B:
                    continue
                if m in ("persistence", "ridge"):
                    pen = 100 * (B[m]["rmse"] - A[m]["rmse"]) / A[m]["rmse"]
                    out["paired_convention"].append({
                        "city": city, "horizon": h, "model": m,
                        "conv_A": round(A[m]["rmse"], 3),
                        "conv_B": round(B[m]["rmse"], 3),
                        "penalty_pct": round(pen, 2),
                        "penalty_sd_pct": 0.0, "n_seeds": 0,
                        "ratio": float("inf")})
                    continue
                sa, sb = A[m]["seeds"], B[m]["seeds"]
                common = [s for s in sa if s in sb]
                if len(common) < 2:
                    continue
                ra = np.array([A[m]["per_seed_rmse"][sa.index(s)] for s in common])
                rb = np.array([B[m]["per_seed_rmse"][sb.index(s)] for s in common])
                pen = 100 * (rb - ra) / ra
                out["paired_convention"].append({
                    "city": city, "horizon": h, "model": m,
                    "n_seeds": len(common),
                    "conv_A_mean": round(float(ra.mean()), 3),
                    "conv_B_mean": round(float(rb.mean()), 3),
                    "penalty_pct": round(float(pen.mean()), 2),
                    "penalty_sd_pct": round(float(pen.std(ddof=1)), 3),
                    "marginal_sd_pct": round(
                        float(100 * ra.std(ddof=1) / ra.mean()), 3),
                    "ratio_paired": round(
                        float(abs(pen.mean()) / max(pen.std(ddof=1), 1e-9)), 1),
                    "ratio_marginal": round(
                        float(abs(pen.mean()) /
                              max(100 * ra.std(ddof=1) / ra.mean(), 1e-9)), 2),
                })

        # ---- architecture range ----------------------------------------
        means = {m: A[m]["rmse_mean"] for m in MODELS if m in A}
        if len(means) >= 4:
            v = np.array(list(means.values()))
            no_tfm = {k: x for k, x in means.items() if k != "transformer"}
            no_seq = {k: x for k, x in means.items()
                      if k not in ("transformer", "lstm")}
            rng_ = lambda d: 100 * (max(d.values()) - min(d.values())) / min(d.values())
            q75, q25 = np.percentile(v, [75, 25])
            beats = [m for m, x in means.items() if x > persA]
            out["architecture_range"].append({
                "city": city, "horizon": h,
                "range_all_pct": round(rng_(means), 2),
                "range_no_transformer_pct": round(rng_(no_tfm), 2),
                "range_no_sequence_pct": round(rng_(no_seq), 2),
                "iqr_pct": round(float(100 * (q75 - q25) / v.min()), 2),
                "worse_than_persistence": beats,
                "persistence_rmse": round(persA, 3),
            })

        # ---- variance decomposition ------------------------------------
        if B is None:
            continue
        rows = []
        for m in MODELS:
            if m not in A or m not in B:
                continue
            sa, sb = A[m]["seeds"], B[m]["seeds"]
            common = [s for s in sa if s in sb]
            for s in common:
                rows.append((m, "A", s,
                             100 * A[m]["per_seed_rmse"][sa.index(s)] / persA))
                rows.append((m, "B", s,
                             100 * B[m]["per_seed_rmse"][sb.index(s)] / persA))
        if len(rows) < 8:
            continue
        mods = sorted({r[0] for r in rows})
        convs = ["A", "B"]
        seeds = sorted({r[2] for r in rows})
        cellmean, grand = {}, np.mean([r[3] for r in rows])
        for m in mods:
            for c in convs:
                vals = [r[3] for r in rows if r[0] == m and r[1] == c]
                if vals:
                    cellmean[(m, c)] = np.mean(vals)
        conv_eff = {c: np.mean([v for (m, cc), v in cellmean.items() if cc == c])
                    for c in convs}
        arch_eff = {m: np.mean([v for (mm, c), v in cellmean.items() if mm == m])
                    for m in mods}
        resid = [r[3] - cellmean[(r[0], r[1])] for r in rows]
        inter = [cellmean[k] - arch_eff[k[0]] - conv_eff[k[1]] + grand
                 for k in cellmean]
        sd = lambda x: float(np.std(list(x), ddof=0))
        out["decomposition"].append({
            "city": city, "horizon": h,
            "n_runs": len(rows), "n_models": len(mods), "n_seeds": len(seeds),
            "grand_mean_rel_rmse_pct": round(float(grand), 2),
            "sd_convention_pct": round(sd(conv_eff.values()), 3),
            "sd_architecture_pct": round(sd(arch_eff.values()), 3),
            "sd_interaction_pct": round(sd(inter), 3),
            "sd_seed_pct": round(sd(resid), 3),
        })

    Path(args.out).write_text(json.dumps(out, indent=1))

    print("PAIRED LAG-CONVENTION PENALTY (percent of convention-A RMSE)")
    print(f"{'city':<9}{'h':>4}{'model':<13}{'penalty':>9}{'paired sd':>11}"
          f"{'|t|':>8}{'marg sd':>9}{'pen/marg':>9}")
    for r in out["paired_convention"]:
        if r["n_seeds"] == 0:
            print(f"{r['city']:<9}{r['horizon']:>4}{r['model']:<13}"
                  f"{r['penalty_pct']:>8.2f}%{'  (exact)':>11}")
        else:
            print(f"{r['city']:<9}{r['horizon']:>4}{r['model']:<13}"
                  f"{r['penalty_pct']:>8.2f}%{r['penalty_sd_pct']:>11.3f}"
                  f"{r['ratio_paired']:>8.1f}{r['marginal_sd_pct']:>9.3f}"
                  f"{r['ratio_marginal']:>9.2f}")

    print("\nARCHITECTURE RANGE")
    print(f"{'city':<9}{'h':>4}{'all':>9}{'-tfm':>9}{'-seq':>9}{'IQR':>9}"
          f"   worse than persistence")
    for r in out["architecture_range"]:
        print(f"{r['city']:<9}{r['horizon']:>4}{r['range_all_pct']:>8.2f}%"
              f"{r['range_no_transformer_pct']:>8.2f}%"
              f"{r['range_no_sequence_pct']:>8.2f}%{r['iqr_pct']:>8.2f}%   "
              f"{r['worse_than_persistence'] or 'none'}")

    print("\nVARIANCE DECOMPOSITION (sd in percent of persistence RMSE)")
    print(f"{'city':<9}{'h':>4}{'convention':>12}{'architecture':>14}"
          f"{'interaction':>13}{'seed':>8}")
    for r in out["decomposition"]:
        print(f"{r['city']:<9}{r['horizon']:>4}{r['sd_convention_pct']:>12.3f}"
              f"{r['sd_architecture_pct']:>14.3f}{r['sd_interaction_pct']:>13.3f}"
              f"{r['sd_seed_pct']:>8.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
