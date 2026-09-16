#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from urbanclimate.envs import PollutantDynamicsEnv, fixed_policy
from urbanclimate.ppo import EVAL_SEED_BASE, torch_policy_fn, train_ppo

torch.set_num_threads(1)
CLOSURE = "saturating"
R = Path("results")
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 110})


# ------------------------------------------------------------------ 1
def fig_lag_convention(out):
    """Delhi h=1 under convention A, convention B, and a mixed table."""
    agg = json.loads((R / "aggregated_rerun.json").read_text())
    A = agg["delhi_h1"]
    B = agg["convention_off0_delhi_h1"]
    models = ["persistence", "ridge", "mlp64", "mlp128", "cmod", "lstm"]
    labels = ["Persistence", "Ridge", "MLP-64", "MLP-128", "C-Mod", "LSTM"]
    a = [A[m]["rmse"] if "rmse" in A[m] else A[m]["rmse_mean"] for m in models]
    b = [B[m] for m in models]
    # a mixed table takes whichever run happens to be at hand per row
    mixed = [a[0], b[1], b[2], b[3], b[4], a[5]]

    x = np.arange(len(models))
    w = 0.27
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.bar(x - w, a, w, label="Convention A applied uniformly", color="#3B6FA0")
    ax.bar(x, b, w, label="Convention B applied uniformly", color="#7BA7CC")
    ax.bar(x + w, mixed, w, label="Table assembled from both", color="#C0392B")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Test RMSE (AQI units)")
    ax.set_title("Delhi, one-hour horizon", loc="left", fontsize=10)
    ax.set_ylim(0, max(b) * 1.28)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3,
              columnspacing=1.4, handlelength=1.5,
              bbox_to_anchor=(0.0, 1.0))
    # mark the gap the mixed table appears to open between C-Mod and LSTM,
    # which is the point of the figure: bracket the two red bars it compares
    xc, xl = 4 + w, 5 + w
    top, bot = mixed[4], mixed[5]
    xb = (xc + xl) / 2
    ax.annotate("", xy=(xb, top), xytext=(xb, bot),
                arrowprops=dict(arrowstyle="<->", color="#C0392B", lw=1.1))
    ax.plot([xc, xb], [top, top], color="#C0392B", lw=0.8, ls=":")
    ax.plot([xb, xl], [bot, bot], color="#C0392B", lw=0.8, ls=":")
    ax.text(xb + 0.06, (top + bot) / 2, f"{top - bot:.2f}", fontsize=8,
            color="#C0392B", va="center", ha="left")
    fig.tight_layout(); fig.savefig(out / "fig_lag_convention.png", dpi=200)
    plt.close(fig)
    print(f"  fig_lag_convention.png   A={a[4]:.2f}/{a[5]:.2f} "
          f"B={b[4]:.2f}/{b[5]:.2f} mixed gap={mixed[4]-mixed[5]:.2f}")


# ------------------------------------------------------------------ 2
def fig_forecast_examples(out, hours=72):
    z = np.load(R / "predictions_crossing" / "delhi_h1_off1_main.npz")
    y = z["y_true"].astype(float)
    # a single trained model, not the seed ensemble: averaging predictions across
    # seeds gives a lower error than the seed-mean RMSE that Table 1 reports
    p = z["cmod_s0"].astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9))
    ax = axes[0]
    ax.plot(np.arange(hours), y[-hours:], lw=1.6, color="#222", label="Observed")
    ax.plot(np.arange(hours), p[-hours:], lw=1.4, color="#C0392B", ls="--",
            label="C-Mod (seed 0)")
    ax.set_xlabel("Hour of the final 72 h of the Delhi test split")
    ax.set_ylabel("AQI"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("A  Final 72 hours", loc="left", fontsize=10)

    ax = axes[1]
    ax.scatter(y, p, s=3, alpha=0.25, color="#3B6FA0", edgecolors="none")
    lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
    ax.plot([lo, hi], [lo, hi], color="#555", lw=1.0, ls=":")
    ax.set_xlabel("Observed AQI"); ax.set_ylabel("Predicted AQI")
    r = float(np.sqrt(np.mean((y - p) ** 2)))
    ax.set_title("B  Whole test split, seed 0", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "fig_forecast_examples.png", dpi=200)
    plt.close(fig)
    print(f"  fig_forecast_examples.png  n={len(y)}  RMSE={r:.2f}")


# ------------------------------------------------------------------ 3
def fig_traffic(out):
    """Ridge sweep over rho, against the across-seed noise floor."""
    allc = json.loads((R / "traffic_informativeness_all_cells.json").read_text())
    fig, ax = plt.subplots(figsize=(8.8, 3.6))
    colours = {1: "#3B6FA0", 24: "#C0392B"}
    for h in (1, 24):
        d = json.loads((R / f"traffic_informativeness_delhi_h{h}.json").read_text())
        rows = d["ridge"]
        rho = sorted(float(k) for k in rows)
        base = rows[f"{rho[0]:.2f}"]["rmse"]
        gain = [100 * (base - rows[f"{r:.2f}"]["rmse"]) / base for r in rho]
        ax.plot(rho, gain, "o-", ms=4, lw=1.5, color=colours[h], label=f"$h = {h}$ h")
        cell = next(c for c in allc
                    if c["city"] == "Delhi" and c["horizon"] == h)
        # noise floor in percent of the rho = 0 C-Mod error
        cbase = d["cmod"][f"{rho[0]:.2f}"]["rmse_mean"]
        band = 100 * cell["seed_noise_rmse"] / cbase
        ax.axhspan(-band, band, color=colours[h], alpha=0.10,
                   label=f"seed noise floor, $h = {h}$ h")
        thr = cell["threshold_rho"]
        if thr:
            ax.axvline(thr, color=colours[h], lw=0.9, ls=":")
            ax.annotate(rf"$\rho^\ast = {thr:.3f}$", xy=(thr, 17 if h == 1 else 11),
                        xytext=(6, 0), textcoords="offset points",
                        fontsize=8, color=colours[h])
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_xlabel(r"$\rho$: correlation between traffic covariate and target")
    ax.set_ylabel(r"RMSE improvement over $\rho = 0$ (%)")
    ax.set_title("Delhi", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2,
              columnspacing=1.2, handlelength=1.6)
    fig.tight_layout(); fig.savefig(out / "fig_traffic_informativeness.png", dpi=200)
    plt.close(fig)
    print("  fig_traffic_informativeness.png")


# ------------------------------------------------------------------ 4, 5
def fig_ppo_and_regimes(out, episodes=300, hours=168):
    env = PollutantDynamicsEnv(episode_hours=hours, seed=0, closure=CLOSURE)
    policy, returns, _ = train_ppo(env, episodes=episodes, seed=0, log_every=0)
    r = np.asarray(returns, float)
    ma = pd.Series(r).rolling(20, min_periods=1).mean().to_numpy()
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.plot(r, lw=0.7, alpha=0.4, color="#7BA7CC", label="episodic return")
    ax.plot(ma, lw=1.8, color="#3B6FA0", label="20-episode moving average")
    ax.set_xlabel("Training episode"); ax.set_ylabel("Return (168 h episode)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"PPO, {CLOSURE} closure, seed 0", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "ppo_learning_curve.png", dpi=200)
    plt.close(fig)
    print(f"  ppo_learning_curve.png   final MA {ma[-1]:.2f}")

    pols = [("No control", fixed_policy("none"), "#888"),
            ("Random", fixed_policy("random"), "#B7950B"),
            ("Fixed $u=0.5$", fixed_policy("half"), "#7BA7CC"),
            ("Fixed max", fixed_policy("max"), "#3B6FA0"),
            ("PPO", torch_policy_fn(policy), "#C0392B")]
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    for name, fn, col in pols:
        e = PollutantDynamicsEnv(episode_hours=hours, seed=EVAL_SEED_BASE,
                                 closure=CLOSURE)
        obs = e.reset(seed=EVAL_SEED_BASE)
        trace, done = [], False
        while not done:
            obs, _, done, info = e.step(np.asarray(fn(obs), float))
            trace.append(info["pm25"])
        ax.plot(trace, lw=1.3, color=col, label=name)
    ax.axhline(15.0, color="#2E7D32", lw=1.0, ls="--",
               label="WHO 24 h guideline")
    ax.set_xlabel("Simulated hour"); ax.set_ylabel(r"PM$_{2.5}$ ($\mu$g m$^{-3}$)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_title("One matched evaluation episode", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "control_regimes.png", dpi=200)
    plt.close(fig)
    print("  control_regimes.png")


# ------------------------------------------------------------------ 6
def fig_regime_map(out):
    d = json.loads((R / "r1_control_rerun.json").read_text())[CLOSURE]
    grid = d["weighting_map"]
    w1s = sorted({v["w1"] for v in grid.values()})
    w3s = sorted({v["w3"] for v in grid.values()})
    G = np.array([[grid[f"{a:.2f}|{b:.2f}"]["gain_over_inaction"] for b in w3s]
                  for a in w1s])
    U = np.array([[grid[f"{a:.2f}|{b:.2f}"]["u_star"] for b in w3s] for a in w1s])

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    ax = axes[0]
    im = ax.imshow(G, origin="lower", aspect="auto", cmap="viridis",
                   extent=[-0.5, len(w3s) - 0.5, -0.5, len(w1s) - 0.5])
    ax.set_xticks(range(len(w3s))); ax.set_xticklabels([f"{b:g}" for b in w3s])
    ax.set_yticks(range(len(w1s))); ax.set_yticklabels([f"{a:g}" for a in w1s])
    ax.set_xlabel(r"$w_3$ (weight on control cost)")
    ax.set_ylabel(r"$w_1$ (weight on pollutant level)")
    ax.set_title("A  Value of intervention: reward gain of the best\n"
                 "constant action over inaction", loc="left", fontsize=10)
    for i in range(len(w1s)):
        for j in range(len(w3s)):
            ax.text(j, i, f"{U[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if G[i, j] < G.max() * 0.55 else "black")
    fig.colorbar(im, ax=ax, label="gain (reward units per step)")

    ax = axes[1]
    rec = d["value_recovered"]
    av = np.array([r["value_available"] for r in rec])
    sh = np.array([r["recovered_pct"] for r in rec])
    order = np.argsort(av)
    ax.plot(av[order], sh[order], "o-", color="#3B6FA0", lw=1.6, ms=6)
    for r in rec:
        ax.annotate(f"({r['w1']}, {r['w3']})",
                    xy=(r["value_available"], r["recovered_pct"]),
                    xytext=(4, -10), textcoords="offset points", fontsize=7)
    ax.axhline(0, color="#555", lw=0.8)
    ax.axhline(100, color="#2E7D32", lw=0.9, ls="--",
               label="best constant action")
    ax.axvspan(0.03, 0.10, color="#C0392B", alpha=0.12,
               label="transition band, all three closures")
    ax.set_xscale("log")
    ax.set_xlabel("Value available (reward units per step, log scale)")
    ax.set_ylabel("Share of available value recovered by PPO (%)")
    ax.set_title("B  What the learner recovers", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "fig_regime_map.png", dpi=200)
    plt.close(fig)
    n_zero = sum(1 for v in grid.values() if v["u_star"] == 0.0)
    print(f"  fig_regime_map.png   u* min {U.min():.2f}, inaction-optimal cells "
          f"{n_zero}, gain {G.min():.4f}-{G.max():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    jobs = {"convention": fig_lag_convention, "examples": fig_forecast_examples,
            "traffic": fig_traffic, "control": fig_ppo_and_regimes,
            "regime": fig_regime_map}
    for name, fn in jobs.items():
        if args.only and name not in args.only:
            continue
        print(f"[{name}]", flush=True)
        fn(out)
    print("done")


if __name__ == "__main__":
    main()
