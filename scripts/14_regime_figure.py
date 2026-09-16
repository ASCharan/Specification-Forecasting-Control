#!/usr/bin/env python3
"""Figure for the benefit-cost weighting sweep and the learner's detection threshold.

A  The value of intervention across weighting space: the gain in average
   per-step reward of the best constant action over doing nothing, as a function
   of the weight on pollutant level (w1) and on quadratic control cost (w3).
   Points mark weightings at which PPO was actually trained.

B  What PPO recovers of that value. Below a gain of roughly 0.1 reward units the
   agent collapses to the zero-action policy and captures almost none of the
   available benefit; above it, it captures nearly all.

Reads results/regime_diagram.json and results/ppo_vs_optimum.json.
Usage:  python scripts/14_regime_figure.py [--outdir figures]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    grid = json.loads(Path("results/regime_diagram.json").read_text())["grid"]
    ppo = json.loads(Path("results/ppo_vs_optimum.json").read_text())

    w1s = sorted({v["w1"] for v in grid.values()})
    w3s = sorted({v["w3"] for v in grid.values()})
    G = np.array([[grid[f"{a:.2f}|{b:.2f}"]["gain_over_inaction"] for b in w3s]
                  for a in w1s])
    U = np.array([[grid[f"{a:.2f}|{b:.2f}"]["u_star"] for b in w3s]
                  for a in w1s])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # --- A: value of intervention -------------------------------------------
    ax = axes[0]
    im = ax.imshow(G, origin="lower", aspect="auto", cmap="viridis",
                   extent=[-0.5, len(w3s) - 0.5, -0.5, len(w1s) - 0.5])
    ax.set_xticks(range(len(w3s))); ax.set_xticklabels([f"{b:g}" for b in w3s])
    ax.set_yticks(range(len(w1s))); ax.set_yticklabels([f"{a:g}" for a in w1s])
    ax.set_xlabel(r"$w_3$  (weight on control cost)")
    ax.set_ylabel(r"$w_1$  (weight on pollutant level)")
    ax.set_title("A  Value of intervention: reward gain of the\n"
                 "best constant action over inaction", loc="left", fontsize=10)
    # annotate the optimal action itself
    for i in range(len(w1s)):
        for j in range(len(w3s)):
            ax.text(j, i, f"{U[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if G[i, j] < G.max() * 0.55 else "black")
    fig.colorbar(im, ax=ax, label="gain (reward units per step)")

    # mark the weightings at which PPO was trained
    for k, v in ppo.items():
        j = w3s.index(v["w3"]); i = w1s.index(v["w1"])
        ok = v["fraction_of_gain_captured"] > 0.5
        ax.scatter([j], [i], s=150, marker="o" if ok else "X",
                   facecolors="none" if ok else "red",
                   edgecolors="red", linewidths=2.0, zorder=5)

    # --- B: what the learner recovers ---------------------------------------
    ax = axes[1]
    pts = sorted(ppo.values(), key=lambda v: v["gain_available"])
    x = [v["gain_available"] for v in pts]
    y = [100 * v["fraction_of_gain_captured"] for v in pts]
    ax.axhspan(0, 20, color="red", alpha=0.07)
    ax.axhline(100, color="grey", ls=":", lw=1)
    ax.plot(x, y, "o-", color="C3", ms=8, lw=1.6)
    for v in pts:
        ax.annotate(f"({v['w1']:g}, {v['w3']:g})",
                    (v["gain_available"], 100 * v["fraction_of_gain_captured"]),
                    textcoords="offset points", xytext=(8, -12), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("value of intervention available (reward units per step)")
    ax.set_ylabel("percentage of that value recovered by PPO")
    ax.set_ylim(-8, 122)
    ax.set_title("B  The learner has a detection threshold:\n"
                 "below ~0.1 it returns the zero-action policy",
                 loc="left", fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = outdir / "fig_regime_map.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
