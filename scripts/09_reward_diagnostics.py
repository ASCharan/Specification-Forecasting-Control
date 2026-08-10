#!/usr/bin/env python3
"""Diagnostics on the reward of Eq. (5).

Establishes two properties reported in the Methods:

  1. The w2 forecast-error term is invariant to the action. The weather fields
     are exogenous, so this term contributes a constant offset to the return
     and cannot influence the learned policy.
  2. The quadratic congestion term is a leading-order contribution to
     emission, not a correction.

Usage:  python scripts/09_reward_diagnostics.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urbanclimate.envs import ALPHA1, ALPHA2, PollutantDynamicsEnv
from urbanclimate.utils import save_json


def forecast_term_invariance(levels=(0.0, 0.25, 0.5, 0.75, 1.0), hours=168):
    out = {}
    for u in levels:
        env = PollutantDynamicsEnv(episode_hours=hours)
        env.reset(seed=777)
        errs = []
        for _ in range(hours):
            W_before = env.W.copy()
            env.step(np.full(env.act_dim, u))
            errs.append(float(np.linalg.norm(
                (env.W - W_before) / np.array([40.0, 100.0, 10.0, 1020.0]))))
        out[f"u={u}"] = round(float(np.mean(errs)), 6)
    return out


def emission_decomposition():
    env = PollutantDynamicsEnv()
    traffic = np.array([env._traffic(t) for t in range(24)])
    rows = {}
    for label, T in (("peak", float(traffic.max())),
                     ("mean", float(traffic.mean())),
                     ("trough", float(traffic.min()))):
        lin, quad = ALPHA1[0] * T, ALPHA2[0] * T ** 2
        rows[label] = {"traffic": round(T, 2), "linear": round(lin, 3),
                       "quadratic": round(quad, 3),
                       "quadratic_share_pct": round(100 * quad / (lin + quad), 1)}
    return rows


def main():
    inv = forecast_term_invariance()
    emi = emission_decomposition()

    print("1. Mean ||eps^W|| by action level (w2 term):")
    for k, v in inv.items():
        print(f"     {k:<8} {v}")
    spread = max(inv.values()) - min(inv.values())
    print(f"   spread across action levels: {spread:.2e}")
    print("   -> the w2 term is invariant to the action and cannot shape the "
          "policy.\n" if spread < 1e-9 else
          "   -> WARNING: term is action-dependent; the Methods text is wrong.\n")

    print("2. PM2.5 emission decomposition:")
    print(f"   {'':<8}{'traffic':>10}{'linear':>10}{'quadratic':>12}{'quad %':>9}")
    for k, v in emi.items():
        print(f"   {k:<8}{v['traffic']:>10.2f}{v['linear']:>10.3f}"
              f"{v['quadratic']:>12.3f}{v['quadratic_share_pct']:>8.1f}%")

    save_json({"forecast_term_invariance": inv,
               "forecast_term_spread": spread,
               "emission_decomposition": emi},
              "results/reward_diagnostics.json")
    print("\nwrote results/reward_diagnostics.json")


if __name__ == "__main__":
    main()
