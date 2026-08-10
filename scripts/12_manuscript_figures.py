#!/usr/bin/env python3
"""Regenerate the two manuscript comparison figures from aggregated results.

Produces, in the manuscript's own style:

  fig_model_comparison.png  skill relative to persistence, by model and city,
                            at both horizons (Figure 1)
  fig_seed_stability.png    across-seed s.d. of test RMSE at h=24 (Figure 3)

Both read results/aggregated_rerun.json, so they stay in step with whatever
02_run_forecasting.py last produced. Every neural model present at a horizon
is plotted; nothing is hard-coded per model, so adding or re-running a model
changes the figure without editing this script.

Usage:  python scripts/12_manuscript_figures.py [--outdir figures]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path("results")
CITIES = ["Delhi", "Mumbai", "Kolkata"]
PRETTY = {"sarima": "SARIMA", "ridge": "Ridge", "mlp64": "MLP-64",
          "mlp128": "MLP-128", "cmod": "C-Mod", "lstm": "LSTM",
          "transformer": "Transformer", "persistence": "Persistence"}
# Plot order: naive, classical, linear, feedforward, continuous-time, sequence.
ORDER = ["persistence", "sarima", "ridge", "mlp64", "mlp128", "cmod",
         "lstm", "transformer"]
SEQUENCE = {"lstm", "transformer"}


def rmse(entry):
    """RMSE of a result entry, whether single-run or seed-averaged."""
    if entry is None:
        return None
    return entry.get("rmse", entry.get("rmse_mean"))


def load():
    p = RES / "aggregated_rerun.json"
    if not p.exists():
        raise SystemExit(f"missing {p} -- run scripts/02_run_forecasting.py first")
    return json.loads(p.read_text())


def fig_model_comparison(agg, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for ax, h in zip(axes, (1, 24)):
        models = [m for m in ORDER
                  if any(rmse(agg.get(f"{c.lower()}_h{h}", {}).get(m)) is not None
                         for c in CITIES)]
        x = np.arange(len(models))
        width = 0.26
        for i, city in enumerate(CITIES):
            res = agg.get(f"{city.lower()}_h{h}", {})
            base = rmse(res.get("persistence"))
            gains, errs = [], []
            for m in models:
                e = res.get(m)
                r = rmse(e)
                if r is None or base is None:
                    gains.append(np.nan); errs.append(0.0); continue
                gains.append((base - r) / base * 100.0)
                # propagate seed spread onto the percentage scale
                sd = (e or {}).get("rmse_std", 0.0) or 0.0
                errs.append(sd / base * 100.0)
            ax.bar(x + (i - 1) * width, gains, width, yerr=errs,
                   capsize=3, label=city, error_kw=dict(lw=1.0))
        ax.axhline(0, color="black", lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([PRETTY.get(m, m) for m in models],
                           rotation=30, ha="right")
        ax.set_ylabel("RMSE improvement over persistence (%)")
        ax.set_title(f"$h = {h}$ hour" + ("s" if h > 1 else ""))
        ax.grid(alpha=0.3, axis="y")
        if h == 1:
            ax.legend(fontsize=9)
    fig.suptitle("Skill relative to the persistence baseline, by model and city")
    fig.tight_layout()
    out = outdir / "fig_model_comparison.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"  wrote {out}")


def fig_seed_stability(agg, outdir):
    models = [m for m in ORDER
              if any((agg.get(f"{c.lower()}_h24", {}).get(m) or {}).get("n_seeds")
                     for c in CITIES)]
    x = np.arange(len(models))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, city in enumerate(CITIES):
        res = agg.get(f"{city.lower()}_h24", {})
        sds = [(res.get(m) or {}).get("rmse_std", np.nan) for m in models]
        bars = ax.bar(x + (i - 1) * width, sds, width, label=city)
        # mark the sequence models, which are the point of the figure
        for m, b in zip(models, bars):
            if m in SEQUENCE:
                b.set_hatch("//")
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY.get(m, m) for m in models])
    ax.set_ylabel("Across-seed s.d. of RMSE")
    ax.set_title("Seed-to-seed stability at $h = 24$ (five seeds); "
                 "hatched bars are the sequence models")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    out = outdir / "fig_seed_stability.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    agg = load()
    fig_model_comparison(agg, outdir)
    fig_seed_stability(agg, outdir)


if __name__ == "__main__":
    main()
