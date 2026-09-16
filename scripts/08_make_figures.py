#!/usr/bin/env python3
"""Regenerate every figure in the manuscript from the released result files.

Run AFTER 02_run_forecasting.py (Delhi h=1 and h=24) and 03_train_ppo.py.
Figures are written to figures/ as 200 dpi PNG.

Usage:  python scripts/08_make_figures.py [--outdir figures]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urbanclimate import data
from urbanclimate.envs import PollutantDynamicsEnv, fixed_policy

RES = Path("results")
PRED = RES / "predictions"
OFF = data.LAG_OFFSET
PRETTY = {"cmod": "C-Mod", "mlp64": "MLP-64", "mlp128": "MLP-128",
          "lstm": "LSTM", "transformer": "Transformer", "ridge": "Ridge",
          "persistence": "Persistence", "seasonal_naive": "Seasonal-naive",
          "climatology": "Climatology", "sarima": "SARIMA"}


def _load(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def fig_convergence(outdir):
    """Validation-loss curves from the saved checkpoints (seed 0, Delhi h=1)."""
    import torch
    ck = Path("checkpoints/delhi/h1")
    curves = {}
    for f in sorted(ck.glob("*_seed0.pt")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        if d.get("val_history"):
            curves[d["model"]] = d["val_history"]
    if not curves:
        print("  [skip] convergence: no checkpoints with val_history")
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, h in curves.items():
        ax.plot(range(1, len(h) + 1), h, label=PRETTY.get(name, name), lw=1.6)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation loss (standardised MSE)")
    ax.set_title("Validation convergence, Delhi $h=1$ (seed 0)")
    ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outdir / "convergence_analysis.png", dpi=200)
    plt.close(fig); print("  wrote convergence_analysis.png")


def fig_forecast_window(outdir, hours=72):
    z = PRED / f"delhi_h1_off{OFF}.npz"
    if not z.exists():
        print("  [skip] forecast window: no predictions"); return
    d = np.load(z); y = d["y_true"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(y[-hours:], color="k", lw=2.0, label="Observed")
    for name in ("lstm", "cmod", "ridge"):
        if name in d.files:
            ax.plot(d[name][-hours:], lw=1.4, alpha=0.9,
                    label=PRETTY.get(name, name))
    ax.set_xlabel("Hour of final test window"); ax.set_ylabel("AQI")
    ax.set_title(f"One-step-ahead forecasts, final {hours} h of the Delhi test split")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outdir / "forecast_72h_aqi.png", dpi=200)
    plt.close(fig); print("  wrote forecast_72h_aqi.png")


def fig_scatter(outdir):
    z = PRED / f"delhi_h1_off{OFF}.npz"
    if not z.exists():
        print("  [skip] scatter: no predictions"); return
    d = np.load(z); y = d["y_true"]
    pair = [m for m in ("lstm", "cmod") if m in d.files]
    if not pair:
        print("  [skip] scatter: needed models absent"); return
    fig, axes = plt.subplots(1, len(pair), figsize=(5 * len(pair), 4.6),
                             squeeze=False)
    lim = (0, float(max(y.max(), max(d[m].max() for m in pair))) * 1.02)
    for ax, m in zip(axes[0], pair):
        ax.scatter(y, d[m], s=3, alpha=0.2, edgecolors="none")
        ax.plot(lim, lim, "r--", lw=1.2)
        r2 = 1 - np.sum((y - d[m]) ** 2) / np.sum((y - y.mean()) ** 2)
        ax.set_title(f"{PRETTY.get(m, m)}  ($R^2$ = {r2:.3f})")
        ax.set_xlabel("Observed AQI"); ax.set_ylabel("Predicted AQI")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outdir / "scatter_pred_obs.png", dpi=200)
    plt.close(fig); print("  wrote scatter_pred_obs.png")


def fig_multicity(outdir):
    cities = ["Delhi", "Mumbai", "Kolkata"]
    order = ["sarima", "persistence", "transformer", "lstm", "ridge",
             "mlp128", "cmod"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    any_data = False
    for ax, h in zip(axes, (1, 24)):
        present = []
        for m in order:
            col = []
            for c in cities:
                j = _load(RES / f"forecast_{c.lower()}_h{h}_off{OFF}.json")
                v = (j or {}).get("results", {}).get(m)
                col.append(np.nan if v is None
                           else v.get("r2", v.get("r2_mean", np.nan)))
            if not np.all(np.isnan(col)):
                present.append((m, col)); any_data = True
        x = np.arange(len(present)); w = 0.26
        for i, c in enumerate(cities):
            ax.bar(x + (i - 1) * w, [p[1][i] for p in present], w, label=c)
        ax.set_xticks(x)
        ax.set_xticklabels([PRETTY.get(p[0], p[0]) for p in present],
                           rotation=30, ha="right")
        ax.set_ylabel("$R^2$"); ax.set_title(f"$h = {h}$ hour" + ("s" if h > 1 else ""))
        ax.grid(alpha=0.3, axis="y")
        if h == 1:
            ax.legend(fontsize=9)
    if not any_data:
        plt.close(fig); print("  [skip] multicity: no result files"); return
    fig.suptitle("Test-split $R^2$ by model and city at both horizons")
    fig.tight_layout(); fig.savefig(outdir / "multicity_summary.png", dpi=200)
    plt.close(fig); print("  wrote multicity_summary.png")


def fig_ppo_curve(outdir):
    j = _load(RES / "control.json")
    if not j or not j.get("curves"):
        print("  [skip] ppo curve: no control.json"); return
    key = sorted(j["curves"])[0]
    r = np.asarray(j["curves"][key], float)
    ma = np.convolve(r, np.ones(20) / 20, mode="valid")
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(r, lw=0.8, alpha=0.45, label="Episode return")
    ax.plot(np.arange(19, 19 + len(ma)), ma, lw=2.2,
            label="20-episode moving average")
    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"Episodic return ({j['config']['episode_hours']} h)")
    ax.set_title(f"PPO training curve ({key})")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(outdir / "ppo_learning_curve.png", dpi=200)
    plt.close(fig); print("  wrote ppo_learning_curve.png")


def fig_control_regimes(outdir, hours=168, seed=10_000):
    import torch
    from urbanclimate.ppo import ActorCritic, torch_policy_fn
    env = PollutantDynamicsEnv(episode_hours=hours)
    series = {}
    for kind, label in [("none", "No control"), ("random", "Random policy"),
                        ("half", "Fixed rule ($u=0.5$)"),
                        ("max", "Fixed max ($u=1$)")]:
        obs = env.reset(seed=seed); pm, done = [], False
        fn = fixed_policy(kind, rng=np.random.default_rng(0))
        while not done:
            obs, _, done, info = env.step(fn(obs)); pm.append(info["pm25"])
        series[label] = pm
    ck = Path("checkpoints/ppo/ppo_seed0.pt")
    if ck.exists():
        d = torch.load(ck, map_location="cpu", weights_only=False)
        pol = ActorCritic(env.obs_dim, env.act_dim)
        pol.load_state_dict(d["state_dict"]); pol.eval()
        fn = torch_policy_fn(pol)
        obs = env.reset(seed=seed); pm, done = [], False
        while not done:
            obs, _, done, info = env.step(fn(obs)); pm.append(info["pm25"])
        series["PPO controller (ours)"] = pm
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for label, pm in series.items():
        ax.plot(pm, lw=1.4, label=label)
    ax.axhline(15.0, ls=":", color="k", lw=1.0, label="WHO 24-h guideline")
    ax.set_xlabel("Hour of evaluation episode")
    ax.set_ylabel(r"PM$_{2.5}$ ($\mu$g/m$^3$)")
    ax.set_title("Simulated PM$_{2.5}$ under different control regimes "
                 "(matched episode)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(outdir / "control_regimes.png", dpi=200)
    plt.close(fig); print("  wrote control_regimes.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    print(f"regenerating figures into {outdir}/ (lag_offset={OFF})")
    for fn in (fig_convergence, fig_forecast_window, fig_scatter,
               fig_multicity, fig_ppo_curve, fig_control_regimes):
        try:
            fn(outdir)
        except Exception as exc:                       # pragma: no cover
            print(f"  [fail] {fn.__name__}: {exc}")
    print("done")


if __name__ == "__main__":
    main()
